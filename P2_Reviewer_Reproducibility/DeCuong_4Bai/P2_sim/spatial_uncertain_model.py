"""136 coupled states with fixed per-realization uncertainty; sealed baseline preserved.

Time unit is one second: xdot=rate*(-G*x+q)/(1 s), rate>0 scales both terms, x in W, e=.02*W*x in K.
Columns 0:59 preserve inherited integrand meanings; 59:89 external chip TV,
89:91 spatial-state TV. All temperatures feed the electrical slopes and heat.
"""
from heterogeneous_bv_model import (np,njit,BASIS,TMAP,ACTIVE_CHIPS,BASE,SIGNS,DS,LOW,HIGH,B0,DB,G0,DG)
from identified_sink_bv_model import SPARSITY as OLD_SPARSITY
from scipy.sparse import lil_matrix
from scipy.integrate import solve_ivp
from thermal_handoff_bv_model import ledger
from bv_feedback_model import P

_jac=lil_matrix((136,136),dtype=int)
_jac[:134,:134]=OLD_SPARSITY
_jac[:134,134:136]=1
_jac[134:136,134:136]=1
SPARSITY=_jac.tocsr()

@njit(cache=True,boundscheck=True)
def rhs_native(t,state,z,kappa,R,taus,case_R,drift,sink_R,mode_taus,W,qext,rate):
    y=state[2:122].reshape((30,4));m=state[122:131]
    current=BASIS@state[:2]
    temp=np.empty(30)
    for j in range(30): temp[j]=40.+np.dot(R[j],y[j])+case_R*m[TMAP[j]]+np.dot(sink_R,state[131:134])+.02*np.dot(W[j],state[134:136])
    ff=-np.expm1(-t/.2) if drift else 0.
    slope=np.empty(3)
    for ph in range(3):
        theta=0.
        for chip in (1,2,8,9): theta+=temp[10*ph+chip]/4
        slope[ph]=min(HIGH,max(LOW,BASE[ph]+kappa[ph]*SIGNS[ph]*(theta-40.)+DS[ph]*ff))
    source=B0+DB*ff+(G0+DG*ff)*z/4096.
    source-=source.mean()
    chip_power=np.zeros(30);module_power=np.zeros(9)
    for ph in range(3):
        positive=.55*slope[ph]*max(current[ph],0.)**2
        negative=.55*slope[ph]*max(-current[ph],0.)**2
        chip_power[10*ph+1]=chip_power[10*ph+8]=positive
        chip_power[10*ph+2]=chip_power[10*ph+9]=negative
        module_power[3*ph]=positive;module_power[3*ph+1]=negative
        module_power[3*ph+2]=positive+negative
    out=np.empty(136)
    out[:2]=BASIS.T@(source-.02*current-slope*current)/.00093
    for j in range(30):
        for b in range(4): out[2+4*j+b]=(chip_power[j]-y[j,b])/taus[b]
    out[122:131]=(module_power-m)/taus[4]
    for mode in range(3):out[131+mode]=(module_power.sum()-state[131+mode])/mode_taus[mode]
    out[134]=rate*(qext[0]-2*state[134]+state[135])
    out[135]=rate*(qext[1]+state[134]-2*state[135])
    return out

def observables(times,states,z,kappa,R,taus,case_R,drift,sink_R,mode_taus,W,qext,rate):
    tt=np.asarray(times);ss=np.asarray(states)
    if ss.ndim==1: ss=ss[:,None]
    tt=np.broadcast_to(tt,ss.shape[1]);n=len(tt)
    current=ss[:2].T@BASIS.T;y=ss[2:122].T.reshape(n,30,4);m=ss[122:131].T
    temp=40.+np.einsum('njb,jb->nj',y,R)+case_R*m[:,TMAP]+(ss[131:134].T@sink_R)[:,None]
    spatial=ss[134:136].T@W.T*.02
    temp+=spatial
    theta=np.stack([temp[:,10*ph+np.array(ACTIVE_CHIPS)].mean(axis=1) for ph in range(3)],axis=1)
    ff=-np.expm1(-tt/.2) if drift else np.zeros(n)
    fp=np.exp(-tt/.2)/.2 if drift else np.zeros(n)
    raw=BASE+kappa*SIGNS*(theta-40)+ff[:,None]*DS
    slope=np.clip(raw,LOW,HIGH)
    bias=B0+ff[:,None]*DB;gain=G0+ff[:,None]*DG
    source=bias+gain*np.asarray(z)/4096.;source-=source.mean(axis=1,keepdims=True)
    pos=np.maximum(current,0.);neg=np.maximum(-current,0.)
    power=.55*slope[:,:,None]*np.stack([pos**2,neg**2],axis=2)
    chip=np.zeros((n,30));module=np.empty((n,9))
    for ph in range(3):
        chip[:,10*ph+1]=chip[:,10*ph+8]=power[:,ph,0]
        chip[:,10*ph+2]=chip[:,10*ph+9]=power[:,ph,1]
        module[:,3*ph]=power[:,ph,0];module[:,3*ph+1]=power[:,ph,1]
        module[:,3*ph+2]=power[:,ph].sum(axis=1)
    idot=source-.02*current-slope*current;idot-=idot.mean(axis=1,keepdims=True);idot/=.00093
    ydot=(chip[:,:,None]-y)/taus[:4];mdot=(module-m)/taus[4]
    mode_dot=(module.sum(axis=1)[:,None]-ss[131:134].T)/mode_taus
    tdot=np.einsum('njb,jb->nj',ydot,R)+case_R*mdot[:,TMAP]+(mode_dot@sink_R)[:,None]
    xdot=rate*(ss[134:136].T@np.array([[-2.,1.],[1.,-2.]])+np.asarray(qext))
    spatial_dot=xdot@W.T*.02
    tdot+=spatial_dot
    theta_dot=np.stack([tdot[:,10*ph+np.array(ACTIVE_CHIPS)].mean(axis=1) for ph in range(3)],axis=1)
    slope_dot=np.where((raw>LOW)&(raw<HIGH),kappa*SIGNS*theta_dot+fp[:,None]*DS,0.)
    source_dot=fp[:,None]*(DB+DG*np.asarray(z)/4096.);source_dot-=source_dot.mean(axis=1,keepdims=True)
    power_dot=.55*(slope_dot[:,:,None]*np.stack([pos**2,neg**2],axis=2)+
        2*slope[:,:,None]*np.stack([pos*idot,-neg*idot],axis=2))
    module_dot=np.empty((n,9))
    for ph in range(3):
        module_dot[:,3*ph]=power_dot[:,ph,0];module_dot[:,3*ph+1]=power_dot[:,ph,1]
        module_dot[:,3*ph+2]=power_dot[:,ph].sum(axis=1)
    integrands=np.c_[np.linalg.norm(idot,axis=1),np.abs(theta_dot).sum(axis=1),
        np.abs(slope_dot).sum(axis=1),np.linalg.norm(source_dot,axis=1),np.abs(tdot),
        np.abs(power_dot).reshape(n,6),np.abs(module_dot),2*power.sum(axis=(1,2)),np.abs(idot).sum(axis=1),np.abs(idot),np.abs(mode_dot@sink_R),np.abs(2*power_dot.sum(axis=(1,2))),np.abs(mode_dot),np.abs(spatial_dot),np.abs(xdot)]
    return dict(current=current,temperature=temp,theta=theta,slopes=slope,raw_slopes=raw,
        spatial_rise=spatial,spatial_states=ss[134:136].T.copy(),source=source,bias=bias,gain=gain,chip_power=chip,module_power=module,integrands=integrands,mode_states=ss[131:134].T.copy(),common_rise=ss[131:134].T@sink_R,
        state_derivative=np.c_[idot@BASIS,ydot.reshape(n,120),mdot,mode_dot,xdot])

def adaptive_dense_integrals(solution,z,kappa,R,taus,case_R,drift,sink_R,mode_taus,W,qext,rate):
    # Absolute-value derivatives and clipping give nonsmooth integrands even
    # when the ODE trajectory is well resolved. Compare each accepted solver
    # interval separately and bisect only intervals failing Gauss8/16 agreement.
    left=solution.sol.ts[:-1].copy();right=solution.sol.ts[1:].copy()
    totals8=np.zeros(91);totals16=np.zeros(91);splits=0;maxdepth=0
    for depth in range(41):
        width=right-left;values=[]
        for order in (8,16):
            nodes,weights=np.polynomial.legendre.leggauss(order)
            times=(left[:,None]+width[:,None]*(nodes+1)/2).ravel()
            obs=observables(times,solution.sol(times),z,kappa,R,taus,case_R,drift,sink_R,mode_taus,W,qext,rate)
            integral=np.einsum('iqk,q,i->ik',obs['integrands'].reshape(len(left),order,91),weights,width/2)
            values.append(integral)
        discrepancy=np.max(np.abs(values[0]-values[1]),axis=1)
        good=discrepancy<=1e-11+1e-9*width
        totals8+=values[0][good].sum(axis=0);totals16+=values[1][good].sum(axis=0)
        if np.all(good):return totals8,totals16,splits,maxdepth
        assert depth<40,'dense_variation_quadrature_not_resolved'
        lo=left[~good];hi=right[~good];mid=(lo+hi)/2
        assert np.all(mid>lo) and np.all(mid<hi)
        left=np.r_[lo,mid];right=np.r_[mid,hi];splits+=len(lo);maxdepth=depth+1
    raise AssertionError('unreachable_quadrature_exit')


def external_input(times,profile):
    tt=np.asarray(times)
    q=np.zeros(tt.shape+(2,))
    if profile!='zero':
        q[...,0]=10.
        if profile in ('ripple10','dual10'):
            q[...,0]*=np.searchsorted(np.array([.2,.4,.6,.8]),tt,side='right')%2==0
    if profile=='dual10':q[...,1]=10.-q[...,0]
    return q

def episode(start,R,taus,rc,kappa,gains,mt,W,capacity,profile,rate=1.,rtol=2e-9,atol=2e-11):
    book=ledger('alternating_drift',capacity);codes=book['delivered'];n=len(codes);dt=P['dt_s']
    edges=np.array([1000,2000,3000,4000],dtype=int) if profile in ('ripple10','dual10') else np.empty(0,dtype=int)
    changes=np.unique(np.r_[0,1+np.flatnonzero(np.any(codes[1:]!=codes[:-1],axis=1)),edges,n])
    state=np.array(start).copy();states=np.empty((136,n+1));states[:,0]=state
    i8=np.zeros(91);i16=np.zeros(91);jumps=0.;prev=np.zeros(3,dtype=np.int16)
    for a,b in zip(changes[:-1],changes[1:]):
        ta,tb=a*dt,b*dt;z=codes[a].astype(float);q=external_input(np.array([(ta+tb)/2]),profile)[0]
        before=observables([ta],state,prev,kappa,R,taus,rc,True,gains,mt,W,q,rate)['source'][0]
        after=observables([ta],state,z,kappa,R,taus,rc,True,gains,mt,W,q,rate)['source'][0]
        jumps+=np.linalg.norm(after-before);prev=z.copy()
        sol=solve_ivp(rhs_native,(ta,tb),state,method='Radau',args=(z,kappa,R,taus,rc,True,gains,mt,W,q,rate),
            jac_sparsity=SPARSITY,rtol=rtol,atol=atol,dense_output=True,max_step=.02)
        assert sol.success,sol.message
        states[:,a+1:b+1]=sol.sol(np.arange(a+1,b+1)*dt);state=sol.y[:,-1]
        aa,bb,_,_=adaptive_dense_integrals(sol,z,kappa,R,taus,rc,True,gains,mt,W,q,rate);i8+=aa;i16+=bb
    times=np.arange(n+1)*dt
    obs=observables(times,states,np.vstack([codes,codes[-1]]),kappa,R,taus,rc,True,gains,mt,W,external_input(times,profile),rate)
    i8[3]+=jumps;i16[3]+=jumps
    return dict(time_s=times,initial_state=np.array(start),final_state=state,states=states,
        current=obs['current'],temperature=obs['temperature'],slopes=obs['slopes'],
        spatial_rise=obs['spatial_rise'],spatial_states=obs['spatial_states'],common_rise=obs['common_rise'],
        integrals_order8=i8,integrals_order16=i16,external_input_W=external_input(times,profile),**book)
