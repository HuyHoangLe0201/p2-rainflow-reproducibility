"""Bounded exact DFS witness for a finite deterministic preview, no terminal set."""
from core import np,njit,STATES,losses,envelope_prediction
from viability_helpers import assess_vector

@njit(cache=True)
def find_path(current,delta,phase,old,edges,edge_sum,U,MU,profile,start,depth,
              lut,R,tau,dt,box_r,box_tau,loss_margin,thermal_limit,peak_error_limit,np_limit,
              weights,node_limit):
    # Explicit DFS stack avoids recursive JIT and records the witness states.
    horizon=min(depth,profile.shape[0]-start)
    path=np.full(depth,-1,dtype=np.int64)
    if horizon<=0:return 1,path,0
    currents=np.zeros((depth+1,3));deltas=np.zeros(depth+1);phases=np.zeros(depth+1)
    olds=np.zeros((depth+1,3),dtype=np.int64);hist=np.zeros((depth+1,50,12),dtype=np.int64)
    sums=np.zeros((depth+1,12),dtype=np.int64);Us=np.zeros((depth+1,30,4));Ms=np.zeros((depth+1,9))
    cursor=np.zeros(depth+1,dtype=np.int64)
    currents[0]=current;deltas[0]=delta;phases[0]=phase;olds[0]=old
    hist[0]=edges;sums[0]=edge_sum;Us[0]=U;Ms[0]=MU
    level=0;nodes=0
    while level>=0:
        if level==horizon:return 1,path,nodes
        if cursor[level]>=27:
            cursor[level]=0;level-=1;continue
        if nodes>=node_limit:return -1,path,nodes
        si=cursor[level];cursor[level]+=1;nodes+=1;k=start+level
        available=sums[level]-hist[level,k%50]
        margin=weights[min(weights.shape[0]-1,int(k*dt/10.)),2]
        mask,pred,dp,edge,nextphase,maxi,error=assess_vector(si,currents[level],deltas[level],phases[level],olds[level],
            available,profile[k],Us[level],Ms[level],lut,R,tau,dt,box_r,box_tau,loss_margin,
            thermal_limit,peak_error_limit,np_limit,margin)
        if mask:continue
        cond,energy=losses(STATES[si],olds[level],currents[level],(currents[level]+pred)/2,lut)
        caps,nextU,nextM=envelope_prediction(Us[level],Ms[level],cond,energy,R,tau,dt,box_r,box_tau,loss_margin)
        if np.max(caps)>thermal_limit-margin:continue
        path[level]=si;currents[level+1]=pred;deltas[level+1]=dp;phases[level+1]=nextphase;olds[level+1]=STATES[si]
        hist[level+1]=hist[level];hist[level+1,k%50]=edge;sums[level+1]=available+edge
        Us[level+1]=nextU;Ms[level+1]=nextM;cursor[level+1]=0;level+=1
    return 0,path,nodes
