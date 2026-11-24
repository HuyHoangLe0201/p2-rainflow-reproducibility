# Source backup SHA256: b46b186f059c48c3faede096ba7faa0084bac0bb5431c333e6519338f54653c4
# Source viability SHA256: 131d6f8cb36368ccc9dd23a6c437cb98140d421b70eeccf64fc5a95e6726f1a8
# Source damage core SHA256: 4a7226a0f1f069422ad03f62ba895a53e6add7214a2b9fe49ec41ac28d399b7a
# Base core SHA256: c9b62527f992a9121c7f91423a349eb54e053ec65d65a769951e110f648c58ed
"""Generated fast-loop weak-device RF allocation. Physical primitives shared with core.

Candidate cost uses the exact increment of terminalized prefix RF potential
for the weak device, with persistent confirmed reversals. Predicted nominal
impulse/end temperatures are used, never RF of the thermal upper envelope.
This is a soft fast-loop cost. Mission hard budgets still need certified
completion selection; no online recursive feasibility is inferred here.
"""
from core import np,njit,STATES,GATES,PATH,MODULE,SHIFT,losses
from core import envelope_prediction,advance_thermal,rf_push,rf_total
from viability_helpers import successor_exists
from backup_helpers import find_path

@njit(cache=True)
def rf_preview_device(j,peak,end,stack,depth,previous,direction,settled,alpha):
    n=depth[j]
    local=np.empty(n+6)
    local[:n]=stack[j,:n]
    last=previous[j]
    direct=direction[j]
    cost=settled[j]
    for value in (peak,end):
        newdir=direct
        if value>last:newdir=1
        elif value<last:newdir=-1
        if direct and newdir!=direct:
            local[n]=last;n+=1
            while n>=3:
                old=abs(local[n-2]-local[n-3])
                new=abs(local[n-1]-local[n-2])
                if new<old:break
                if n==3:
                    cost+=.5*(old/40.)**alpha
                    local[0]=local[1];local[1]=local[2];n=2
                else:
                    cost+=(old/40.)**alpha
                    local[n-3]=local[n-1];n-=2
        last=value;direct=newdir
    if last!=local[n-1]:local[n]=last;n+=1
    while n>=3:
        old=abs(local[n-2]-local[n-3])
        new=abs(local[n-1]-local[n-2])
        if new<old:break
        if n==3:
            cost+=.5*(old/40.)**alpha
            local[0]=local[1];local[1]=local[2];n=2
        else:
            cost+=(old/40.)**alpha
            local[n-3]=local[n-1];n-=2
    for b in range(n-1):cost+=.5*(abs(local[b+1]-local[b])/40.)**alpha
    return cost

@njit(cache=True)
def simulate_recorded(profile, weights, lut, R, tau, dt=.0002, box_r=.15, box_tau=.2,
             loss_margin=1.1, thermal_limit=150., peak_error_limit=277.2,
             np_limit=90., alpha=4.8, trace_stride=50, damage_lambda=0.,
             damage_reference=1., weak_device=18, damage_start_step=0, lookahead=False, horizon=2, guard_np=60., node_limit=4096, stop_step=-1, forced_actions=np.empty(0,dtype=np.int64)):
    """profile columns: Ipk, voltage utilization m, power factor, fe.

    weights per stage: thermal weight, balance weight, hard margin (C).
    Exact discrete affine RL step with frozen semiconductor voltage drops.
    Thermal pulse is an ideal impulse; use endpoint RF convention declared.
    """
    N=profile.shape[0]
    actions=np.full(N,-1,dtype=np.int64)
    count=(N+trace_stride-1)//trace_stride
    trace=np.zeros((count,12+30))
    y=np.zeros((30,4));U=y.copy()
    my=np.zeros(9);MU=my.copy()
    current=np.zeros(3);old=np.zeros(3,dtype=np.int64)
    edges=np.zeros((50,12),dtype=np.int64);edge_sum=np.zeros(12,dtype=np.int64)
    total_edges=np.zeros(12,dtype=np.int64)
    stack=np.zeros((30,4096));stack[:,0]=40.
    depth=np.ones(30,dtype=np.int64);previous=np.full(30,40.)
    direction=np.zeros(30,dtype=np.int64);settled=np.zeros(30)
    debug=np.zeros(13)
    metrics=np.zeros(15)
    metrics[4]=1e9 # minimum valid vectors
    phase=0.;delta=0.;completed=0
    rl_a=np.exp(-.02*dt/.00093);rl_g=(1-rl_a)/.02
    reference=np.empty(3);emf=np.empty(3)
    edges_out=np.zeros(12,dtype=np.int64)
    temps=np.full(30,40.)
    for k in range(N):
        if k==stop_step:
            metrics[14]=3. # Deliberate checkpoint, not physical failure.
            break
        rf_cost_now=0.
        active_damage=damage_lambda>0. and k>=damage_start_step
        if active_damage:
            rf_cost_now=rf_preview_device(weak_device,previous[weak_device],previous[weak_device],
                stack,depth,previous,direction,settled,alpha)
        ipk,m,pf,fe=profile[k]
        midphase=phase+np.pi*fe*dt
        nextphase=phase+2*np.pi*fe*dt
        phi=np.arccos(max(-1.,min(1.,pf)))
        for ph in range(3):
            x=midphase+SHIFT[ph]
            iref=ipk*np.sin(x)
            emf[ph]=(1800.*m/np.sqrt(3))*np.sin(x+phi)-.02*iref-.00093*ipk*2*np.pi*fe*np.cos(x)
            reference[ph]=ipk*np.sin(nextphase+SHIFT[ph])
        slot=k%50
        available=edge_sum-edges[slot]
        best=1e99;best_state=-1;valid_count=0
        predictions=np.zeros((27,3));np_preds=np.zeros(27)
        base_costs=np.full(27,1e99)
        edge_candidates=np.zeros((27,12),dtype=np.int64)
        for si in range(27):
            if len(forced_actions)>0 and si!=forced_actions[k]:continue
            q=STATES[si]
            valid=True
            edge_count=0
            resource_cost=0.
            for ph in range(3):
                if abs(q[ph]-old[ph])>1: valid=False
                for gate in range(4):
                    gi=4*ph+gate
                    edge=abs(GATES[q[ph]+1,gate]-GATES[old[ph]+1,gate])
                    edge_candidates[si,gi]=edge
                    edge_count+=edge
                    if available[gi]+edge>30: valid=False
                    used=available[gi]+edge
                    resource_cost+=.0001*(used/max(1.,31.-used))**2
            if not valid: continue
            voltage=np.empty(3)
            for ph in range(3):
                level=q[ph]
                ideal=(900.+delta/2) if level==1 else (-900.+delta/2 if level==-1 else 0.)
                sg=0 if current[ph]>=0 else 1
                ix=min(1200,int(abs(current[ph])+.5))
                drop=0.
                for b in range(2):
                    d=PATH[sg,level+1,b]
                    drop+=lut[0 if d<4 else 1,ix]
                voltage[ph]=ideal-(drop if current[ph]>=0 else -drop)-emf[ph]
            avg=np.mean(voltage)
            err=0.;peakerr=0.;maxi=0.;i0=0.
            for ph in range(3):
                predictions[si,ph]=rl_a*current[ph]+rl_g*(voltage[ph]-avg)
                e=predictions[si,ph]-reference[ph]
                err+=e*e
                peakerr=max(peakerr,abs(e))
                maxi=max(maxi,abs(predictions[si,ph]))
                if q[ph]==0:i0+=(current[ph]+predictions[si,ph])/2
            dp=delta+dt*i0/.01
            np_preds[si]=dp
            if abs(dp)>np_limit or maxi>600. or peakerr>peak_error_limit:continue
            cost=err/(462.*462.)+.002*edge_count+2.*(dp/np_limit)**2+resource_cost
            base_costs[si]=cost
        best_base=np.min(base_costs)
        stage=min(weights.shape[0]-1,int(k*dt/10.))
        tw,bw,margin=weights[stage]
        # Exact lower-bound pruning: all health terms are nonnegative.
        # No fixed candidate band that could incorrectly report an empty set.
        for si in np.argsort(base_costs):
            if base_costs[si]>=best or base_costs[si]>1e90:continue
            mid=(current+predictions[si])/2
            cond,energy=losses(STATES[si],old,current,mid,lut)
            caps,trialU,trialMU=envelope_prediction(U,MU,cond,energy,R,tau,dt,box_r,box_tau,loss_margin)
            if np.max(caps)>thermal_limit-margin:continue
            valid_count+=1
            maxT=np.max(caps)
            balance=0.
            for ph in range(3):
                # Compare four IGBTs only; diode models have different Rth.
                mean=0.
                for j in range(4):mean+=caps[10*ph+j]/4.
                for j in range(4):balance+=(caps[10*ph+j]-mean)**2/(4*40.*40.)
            cost=base_costs[si]+tw*((maxT-40.)/110.)**2+bw*balance
            if active_damage:
                pk,te=advance_thermal(y.copy(),my.copy(),cond,energy,R,tau,dt)
                trial=rf_preview_device(weak_device,pk[weak_device],te[weak_device],
                    stack,depth,previous,direction,settled,alpha)
                increment=trial-rf_cost_now
                debug[0]+=1
                if increment < -1e-10:debug[1]+=1
                penalty=damage_lambda*N*max(0.,increment)/damage_reference
                debug[3]=max(debug[3],penalty)
                cost+=penalty
            if cost<best:
                if lookahead and k+1<N:
                    successor_available=available+edge_candidates[si]-edges[(k+1)%50]
                    nextstage=min(weights.shape[0]-1,int((k+1)*dt/10.))
                    debug[6]+=1
                    if horizon>2 and max(abs(delta),abs(np_preds[si]))>guard_np:
                        trialedges=edges.copy();trialedges[slot]=edge_candidates[si]
                        status,path,nodes=find_path(predictions[si],np_preds[si],nextphase,STATES[si],trialedges,
                            available+edge_candidates[si],trialU,trialMU,profile,k+1,horizon-1,
                            lut,R,tau,dt,box_r,box_tau,loss_margin,thermal_limit,peak_error_limit,np_limit,weights,node_limit)
                        debug[8]+=1;debug[9]+=nodes;debug[10]=max(debug[10],nodes)
                        if status==-1:debug[11]+=1
                        if status!=1:
                            debug[7]+=1
                            continue
                    elif not successor_exists(predictions[si],np_preds[si],nextphase,STATES[si],successor_available,
                        profile[k+1],trialU,trialMU,lut,R,tau,dt,box_r,box_tau,loss_margin,
                        thermal_limit,peak_error_limit,np_limit,weights[nextstage,2]):
                        debug[7]+=1
                        continue
                best=cost;best_state=si
        if best_state<0:
            # Empty set is a reported failure, never a silent relaxed fallback.
            metrics[14]=1.
            break
        if active_damage and best_state!=np.argmin(base_costs):debug[2]+=1
        debug[4]=max(debug[4],depth[weak_device])
        if active_damage:debug[5]+=1
        si=best_state;q=STATES[si]
        actions[k]=si
        pred=predictions[si];mid=(current+pred)/2
        cond,energy=losses(q,old,current,mid,lut)
        caps,U,MU=envelope_prediction(U,MU,cond,energy,R,tau,dt,box_r,box_tau,loss_margin)
        peaks,temps=advance_thermal(y,my,cond,energy,R,tau,dt)
        for j in range(30):
            if not rf_push(peaks[j],j,stack,depth,previous,direction,settled,alpha):
                metrics[14]=2.
            if not rf_push(temps[j],j,stack,depth,previous,direction,settled,alpha):
                metrics[14]=2.
        if metrics[14]!=0:break
        current=pred.copy();delta=np_preds[si];phase=nextphase
        edges[slot]=edge_candidates[si]
        edge_sum=available+edges[slot]
        total_edges+=edges[slot]
        old=q.copy();completed=k+1
        err2=np.sum((current-reference)**2)
        metrics[0]+=err2/3.
        if k<damage_start_step:debug[12]+=err2/3.
        metrics[1]=max(metrics[1],np.max(np.abs(current-reference)))
        metrics[2]=max(metrics[2],np.max(peaks),np.max(temps))
        metrics[3]=max(metrics[3],np.max(caps))
        metrics[4]=min(metrics[4],valid_count)
        metrics[5]=max(metrics[5],abs(delta))
        metrics[6]=max(metrics[6],np.max(edge_sum)/(2*50*dt))
        metrics[7]+=np.sum(cond)*dt
        metrics[8]+=np.sum(energy)
        metrics[9]+=np.dot(current,emf)*dt
        metrics[10]=max(metrics[10],np.max(np.abs(current)))
        metrics[11]=max(metrics[11],np.max(peaks-caps),np.max(temps-caps))
        if k%trace_stride==0:
            row=k//trace_stride
            trace[row,0]=k*dt;trace[row,1:4]=current;trace[row,4:7]=reference
            trace[row,7]=delta;trace[row,8]=np.max(caps);trace[row,9]=np.max(peaks)
            trace[row,10]=np.max(edge_sum)/(2*50*dt);trace[row,11]=ipk
            trace[row,12:]=temps
    snapshot=(current,old,edges,edge_sum,U,MU,y,my,stack,depth,previous,direction,settled,np.array([delta,phase,float(completed)]))
    metrics[12]=completed
    metrics[13]=np.max(total_edges)/(2*max(completed,1)*dt)
    return metrics,rf_total(stack,depth,previous,settled,alpha),trace[:(completed+trace_stride-1)//trace_stride],total_edges,debug,snapshot,actions
