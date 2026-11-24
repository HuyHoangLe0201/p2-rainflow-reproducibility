"""Exact one-step admissibility for the unchanged digital plant.

Two-step existence is a finite-lookahead heuristic, not an invariant-set proof.
Mask bits: adjacency1, switching-window2, NP4, current8, tracking16, thermal32.
"""
from core import np,njit,STATES,GATES,SHIFT,PATH,losses,envelope_prediction

@njit(cache=True)
def assess_vector(si,current,delta,phase,old,available,row,U,MU,lut,R,tau,
                  dt,box_r,box_tau,loss_margin,thermal_limit,peak_error_limit,np_limit,margin):
    q=STATES[si];mask=0;edge=np.zeros(12,dtype=np.int64)
    for ph in range(3):
        if abs(q[ph]-old[ph])>1:mask|=1
        for gate in range(4):
            gi=4*ph+gate;edge[gi]=abs(GATES[q[ph]+1,gate]-GATES[old[ph]+1,gate])
            if available[gi]+edge[gi]>30:mask|=2
    ipk,m,pf,fe=row;midphase=phase+np.pi*fe*dt;nextphase=phase+2*np.pi*fe*dt
    phi=np.arccos(max(-1.,min(1.,pf)));voltage=np.empty(3);ref=np.empty(3)
    for ph in range(3):
        x=midphase+SHIFT[ph];iref=ipk*np.sin(x)
        emf=(1800.*m/np.sqrt(3))*np.sin(x+phi)-.02*iref-.00093*ipk*2*np.pi*fe*np.cos(x)
        ref[ph]=ipk*np.sin(nextphase+SHIFT[ph])
        level=q[ph];ideal=(900.+delta/2) if level==1 else (-900.+delta/2 if level==-1 else 0.)
        sg=0 if current[ph]>=0 else 1;ix=min(1200,int(abs(current[ph])+.5));drop=0.
        for b in range(2):
            d=PATH[sg,level+1,b];drop+=lut[0 if d<4 else 1,ix]
        voltage[ph]=ideal-(drop if current[ph]>=0 else -drop)-emf
    avg=np.mean(voltage);rl_a=np.exp(-.02*dt/.00093);rl_g=(1-rl_a)/.02
    pred=np.empty(3);maxi=0.;peakerr=0.;i0=0.
    for ph in range(3):
        pred[ph]=rl_a*current[ph]+rl_g*(voltage[ph]-avg)
        maxi=max(maxi,abs(pred[ph]));peakerr=max(peakerr,abs(pred[ph]-ref[ph]))
        if q[ph]==0:i0+=(current[ph]+pred[ph])/2
    dp=delta+dt*i0/.01
    if abs(dp)>np_limit:mask|=4
    if maxi>600.:mask|=8
    if peakerr>peak_error_limit:mask|=16
    return mask,pred,dp,edge,nextphase,maxi,peakerr

@njit(cache=True)
def successor_exists(current,delta,phase,old,available,row,U,MU,lut,R,tau,
                     dt,box_r,box_tau,loss_margin,thermal_limit,peak_error_limit,np_limit,margin):
    for si in range(27):
        mask,pred,dp,edge,nextphase,maxi,peakerr=assess_vector(si,current,delta,phase,old,available,row,U,MU,lut,R,tau,
            dt,box_r,box_tau,loss_margin,thermal_limit,peak_error_limit,np_limit,margin)
        if mask:continue
        cond,energy=losses(STATES[si],old,current,(current+pred)/2,lut)
        caps,_,_=envelope_prediction(U,MU,cond,energy,R,tau,dt,box_r,box_tau,loss_margin)
        if np.max(caps)<=thermal_limit-margin:return True
    return False

@njit(cache=True)
def all_masks(current,delta,phase,old,available,row,U,MU,lut,R,tau,
              dt,box_r,box_tau,loss_margin,thermal_limit,peak_error_limit,np_limit,margin):
    masks=np.zeros(27,dtype=np.int64);values=np.zeros((27,4))
    for si in range(27):
        mask,pred,dp,edge,_,maxi,peakerr=assess_vector(si,current,delta,phase,old,available,row,U,MU,lut,R,tau,
            dt,box_r,box_tau,loss_margin,thermal_limit,peak_error_limit,np_limit,margin)
        cond,energy=losses(STATES[si],old,current,(current+pred)/2,lut)
        caps,_,_=envelope_prediction(U,MU,cond,energy,R,tau,dt,box_r,box_tau,loss_margin)
        Tmax=np.max(caps)
        if Tmax>thermal_limit-margin:mask|=32
        masks[si]=mask;values[si]=[dp,maxi,peakerr,Tmax]
    return masks,values
