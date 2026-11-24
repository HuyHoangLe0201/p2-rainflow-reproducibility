"""Immutable certificate replacement and model-specific terminal conditions."""
from dataclasses import dataclass
from hashlib import sha256
import numpy as np
from core import rf_push,rf_total,MODULE

def terminal_check(snap,lut,R):
 current,old,edges,edge_sum,U,MU,*_=snap;delta=snap[-1][0]
 a=float(np.exp(-.02*.0002/.00093));c=.0002*(1+a)/(2*.01)
 infinity=float(delta+c*sum(current)/(1-a))
 cap=np.array([40.+.0215*1.15*MU[3*(j//10)+MODULE[j%10]]+np.sum(1.15*R[j]*U[j]) for j in range(30)])
 checks={'zero_drop_at_lookup_index0':bool(np.all(lut[:2,0]==0)),
  'current_in_deadband':bool(np.max(np.abs(current))<=.49),'old_switch000':bool(np.all(old==0)),
  'window_valid':bool(np.all(edge_sum<=30) and np.all(edge_sum>=0) and np.array_equal(edge_sum,edges.sum(axis=0))),
  'NP_current_and_idle_limit':bool(max(abs(delta),abs(infinity))<=90),
  'thermal_nonnegative_and_safe':bool(np.all(U>=0) and np.all(MU>=0) and np.max(cap)<=150)}
 return dict(pass_terminal=all(checks.values()),checks=checks,idle_NP_limit_V=infinity,
  initial_peak_bound_C=float(cap.max()),max_current_A=float(np.max(np.abs(current))),decay_a=a)

def cooldown_damage(snap):
 stack=snap[8].copy();depth=snap[9].copy();prev=snap[10].copy();direction=snap[11].copy();settled=snap[12].copy()
 for j in range(30):assert rf_push(40.,j,stack,depth,prev,direction,settled,4.8)
 return rf_total(stack,depth,prev,settled,4.8)

@dataclass(frozen=True)
class Certificate:
 actions: np.ndarray
 profile_hash: str
 model_hash: str
 certified: bool
 label: str
 def __post_init__(self):
  actions=np.array(self.actions,dtype=np.int64,copy=True);actions.flags.writeable=False
  object.__setattr__(self,'actions',actions)
 def digest(self):return sha256(self.actions.tobytes()).hexdigest()

class StoredManager:
 def __init__(self,certificate):
  if not certificate.certified:raise ValueError('No initial certified backup')
  self.backup=certificate;self.position=0;self.executed=[];self.events=[]
 def replace(self,candidate,reason='planner_unavailable'):
  before=self.backup.digest();accepted=False
  if candidate is not None:
   if candidate.profile_hash!=self.backup.profile_hash:reason='profile_hash_mismatch'
   elif candidate.model_hash!=self.backup.model_hash:reason='model_hash_mismatch'
   elif not candidate.certified:reason='certificate_failed'
   elif len(candidate.actions)!=len(self.backup.actions):reason='horizon_mismatch'
   elif not np.array_equal(candidate.actions[:self.position],np.array(self.executed,dtype=np.int64)):reason='executed_prefix_mismatch'
   else:self.backup=candidate;accepted=True;reason='certificate_committed'
  after=self.backup.digest()
  if not accepted:assert before==after
  self.events.append(dict(step=self.position,accepted=accepted,reason=reason,backup_before=before,backup_after=after))
  return accepted
 def next_action(self):
  if self.position>=len(self.backup.actions):
   self.position+=1;return 13 # STATES[13]==000, valid only under the certified idle assumptions.
  action=int(self.backup.actions[self.position]);self.executed.append(action);self.position+=1;return action
