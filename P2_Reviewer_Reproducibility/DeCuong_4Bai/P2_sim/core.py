"""27-state NPC, deterministic affine RL/EMF electrical surrogate, pulse thermal model.

JIT execution is PC software simulation, never HIL or real-time qualification.
Temperatures used for electrical/loss tables are fixed at 150 C by design.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "runtime"))
import numpy as np
from numba import njit

STATES = np.array([[a,b,c] for a in (-1,0,1) for b in (-1,0,1)
                   for c in (-1,0,1)], dtype=np.int64)
GATES = np.array([[0,0,1,1],[0,1,1,0],[1,1,0,0]],dtype=np.int64)
# sign+, sign-; P/N/O; device order T1..T4,D1..D4,DcU,DcL.
PATH = np.array([[[6,7],[8,1],[0,1]],[[2,3],[2,9],[4,5]]],dtype=np.int64)
MODULE = np.array([0,0,1,1,0,0,1,1,2,2],dtype=np.int64)
SHIFT = np.array([0.,-2*np.pi/3,2*np.pi/3])


@njit(cache=True)
def rf_push(value, j, stack, depth, prev, direction, settled, alpha):
    """Persistent confirmed-reversal stack; live endpoint is never committed."""
    olddir = direction[j]
    newdir = olddir
    if value > prev[j]: newdir = 1
    elif value < prev[j]: newdir = -1
    if olddir and newdir != olddir:
        n = depth[j]
        if n >= stack.shape[1]:
            return False
        stack[j,n] = prev[j]
        n += 1
        while n >= 3:
            old = abs(stack[j,n-2]-stack[j,n-3])
            new = abs(stack[j,n-1]-stack[j,n-2])
            if new < old: break
            if n == 3:
                settled[j] += .5*(old/40.)**alpha
                stack[j,0] = stack[j,1]
                stack[j,1] = stack[j,2]
                n = 2
            else:
                settled[j] += (old/40.)**alpha
                stack[j,n-3] = stack[j,n-1]
                n -= 2
        depth[j] = n
    direction[j] = newdir
    prev[j] = value
    return True


@njit(cache=True)
def rf_total(stack, depth, prev, settled, alpha):
    result = settled.copy()
    for j in range(30):
        n = depth[j]
        local=np.empty(stack.shape[1]+1)
        local[:n]=stack[j,:n]
        if prev[j] != local[n-1]:
            local[n]=prev[j]
            n+=1
        while n>=3:
            old=abs(local[n-2]-local[n-3])
            new=abs(local[n-1]-local[n-2])
            if new<old:break
            if n==3:
                result[j]+=.5*(old/40.)**alpha
                local[0]=local[1];local[1]=local[2];n=2
            else:
                result[j]+=(old/40.)**alpha
                local[n-3]=local[n-1];n-=2
        for b in range(n-1):
            result[j] += .5*(abs(local[b+1]-local[b])/40.)**alpha
    return result


@njit(cache=True)
def losses(q, old, currents, midpoint, lut):
    cond = np.zeros(30)
    energy = np.zeros(30)
    for ph in range(3):
        sg = 0 if currents[ph] >= 0 else 1
        im = min(1200,max(0,int(abs(midpoint[ph])+.5)))
        isw = min(1200,max(0,int(abs(currents[ph])+.5)))
        for k in range(2):
            d = PATH[sg,q[ph]+1,k]
            cond[10*ph+d] += abs(midpoint[ph])*lut[0 if d<4 else 1,im]
        if q[ph] != old[ph]:
            # P<->O: T1 / DcU for positive current, T3 / D1 for negative.
            # O<->N: T2 / D4 for positive current, T4 / DcL for negative.
            if max(q[ph],old[ph]) == 1:
                tr,di = (0,8) if sg == 0 else (2,4)
                turn_on = q[ph] == (1 if sg == 0 else 0)
            else:
                tr,di = (1,7) if sg == 0 else (3,9)
                turn_on = q[ph] == (0 if sg == 0 else -1)
            energy[10*ph+tr] += lut[2 if turn_on else 3,isw]
            if turn_on: energy[10*ph+di] += lut[4,isw]
    return cond, energy


@njit(cache=True)
def advance_thermal(y, module_y, cond, energy, R, tau, dt):
    mod_cond = np.zeros(9)
    mod_energy = np.zeros(9)
    for j in range(30):
        mo=3*(j//10)+MODULE[j%10]
        mod_cond[mo]+=cond[j]
        mod_energy[mo]+=energy[j]
    peak = np.empty(30)
    end = np.empty(30)
    am = np.exp(-dt/2.)
    # Effective dynamic shared case->sink path, tau=2s (design assumption).
    mj = module_y+mod_energy/2.
    me = am*mj+(1-am)*mod_cond
    aa=np.exp(-dt/tau)
    for j in range(30):
        mo=3*(j//10)+MODULE[j%10]
        peak[j]=40.+.0215*mj[mo]
        end[j]=40.+.0215*me[mo]
        for m in range(4):
            jump=y[j,m]+energy[j]/tau[m]
            a=aa[m]
            y[j,m]=a*jump+(1-a)*cond[j]
            peak[j]+=R[j,m]*jump
            end[j]+=R[j,m]*y[j,m]
    module_y[:] = me
    return peak,end


@njit(cache=True)
def envelope_prediction(U,M,cond,energy,R,tau,dt,box_r,box_tau,loss_margin):
    mod_cond=np.zeros(9)
    mod_energy=np.zeros(9)
    for j in range(30):
        mo=3*(j//10)+MODULE[j%10]
        mod_cond[mo]+=cond[j]*loss_margin
        mod_energy[mo]+=energy[j]*loss_margin
    mj=M+mod_energy/(2.*(1-box_tau))
    ma0=np.exp(-dt/(2.*(1-box_tau)))
    ma1=np.exp(-dt/(2.*(1+box_tau)))
    me=np.maximum(ma0*mj+(1-ma0)*mod_cond,ma1*mj+(1-ma1)*mod_cond)
    nextU=np.empty_like(U)
    cap=np.empty(30)
    aa0=np.exp(-dt/(tau*(1-box_tau)))
    aa1=np.exp(-dt/(tau*(1+box_tau)))
    for j in range(30):
        mo=3*(j//10)+MODULE[j%10]
        cap[j]=40.+.0215*(1+box_r)*max(mj[mo],me[mo])
        for m in range(4):
            jump=U[j,m]+energy[j]*loss_margin/(tau[m]*(1-box_tau))
            a0=aa0[m]
            a1=aa1[m]
            power=cond[j]*loss_margin
            nxt=max(a0*jump+(1-a0)*power,a1*jump+(1-a1)*power)
            nextU[j,m]=nxt
            # Sum of branchwise endpoint maxima bounds every time within ZOH.
            cap[j]+=R[j,m]*(1+box_r)*max(jump,nxt)
    return cap,nextU,me


@njit(cache=True)
def simulate(profile, weights, lut, R, tau, dt=.0002, box_r=.15, box_tau=.2,
             loss_margin=1.1, thermal_limit=150., peak_error_limit=277.2,
             np_limit=90., alpha=4.8, trace_stride=50):
    """profile columns: Ipk, voltage utilization m, power factor, fe.

    weights per stage: thermal weight, balance weight, hard margin (C).
    Exact discrete affine RL step with frozen semiconductor voltage drops.
    Thermal pulse is an ideal impulse; use endpoint RF convention declared.
    """
    N=profile.shape[0]
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
    metrics=np.zeros(15)
    metrics[4]=1e9 # minimum valid vectors
    phase=0.;delta=0.;completed=0
    rl_a=np.exp(-.02*dt/.00093);rl_g=(1-rl_a)/.02
    reference=np.empty(3);emf=np.empty(3)
    edges_out=np.zeros(12,dtype=np.int64)
    temps=np.full(30,40.)
    for k in range(N):
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
            caps,_,_=envelope_prediction(U,MU,cond,energy,R,tau,dt,box_r,box_tau,loss_margin)
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
            if cost<best:
                best=cost;best_state=si
        if best_state<0:
            # Empty set is a reported failure, never a silent relaxed fallback.
            metrics[14]=1.
            break
        si=best_state;q=STATES[si]
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
    metrics[12]=completed
    metrics[13]=np.max(total_edges)/(2*max(completed,1)*dt)
    return metrics,rf_total(stack,depth,previous,settled,alpha),trace[:(completed+trace_stride-1)//trace_stride],total_edges
