import sys
import os
sys.path.append('/users/csmith36/shadows')
import numpy as np
import jax.numpy as jnp
import shadow_evolution_learning
from shadow_evolution_learning.utils.operations import denseXP, denseXM, denseYP, denseYM, denseZP, denseZM, zero_state, one_state, ket0, ket1, hadamard, phase_z, pX,pY, pZ
from shadow_evolution_learning.utils.time_evolution_simulator import timeEvolution
from shadow_evolution_learning.utils.utils import commutator, makeRho
from shadow_evolution_learning.shadow_sampling.shadow_obs import classicalShadowCalc, estimate_shadow_observable
from shadow_evolution_learning.shadow_sampling import shadow_loader

import matplotlib.pyplot as plt
import numpy as np

import jax

from shadow_evolution_learning.utils.network_utils import process_mat, partial_trace,construct_exact_vals, construct_test_vals
from shadow_evolution_learning.networks import DAE
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('--index', '-ind', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--start', type=float, default=0.02)
parser.add_argument('--dt', type=float, default=0.01)
parser.add_argument('--steps',type=int,default=20)
parser.add_argument('--measurements','-m',type=int,default=10000)
parser.add_argument('--iters','-i',type=int,default=2000)
parser.add_argument('--path',type=str,default='results')
parser.add_argument('--k',type=int,default=10)
parser.add_argument('--max_hid_u','-max_h',type=int,default=20)
parser.add_argument('--min_hid_u','-min_h',type=int,default=5)
parser.add_argument('--lr',type=float,default=1e-2)



args = parser.parse_args()
seed = int(args.seed//10)
index = int(args.seed%4)+6
start = args.start
dt = args.dt
steps = args.steps
meas = args.measurements
path = args.path
iters = args.iters
k = args.k
max_hid = args.max_hid_u
min_hid = args.min_hid_u
lr = args.lr

### Make save path
try:
    if not os.path.exists(path):
        os.mkdir(path)
except:
    pass

try:
    path=path+'/measurements='+str(meas)+'_k='+str(k)
    if not os.path.exists(path):
        os.mkdir(path)
except:
    pass

path=path+'/start='+str(start)+'_dt='+str(dt)+'_steps='+str(steps)
try:
    if not os.path.exists(path):
        os.mkdir(path)
except:
    pass

path=path+'/index='+str(index)
try:
    if not os.path.exists(path):
        os.mkdir(path)
except:
    pass
path=path+'/max_hid='+str(max_hid) + '_min_hid='+str(min_hid)
try:
    if not os.path.exists(path):
        os.mkdir(path)
except:
    pass
path=path+'/lr='+str(lr)
try:
    if not os.path.exists(path):
        os.mkdir(path)
except:
    pass
path=path+'/seed='+str(seed)
try:
    if not os.path.exists(path):
        os.mkdir(path)
except:
    pass

### Define Hamiltonian
vals = np.load('/users/csmith36/shadows/NNSHL/ising/ising_shadow_data/params.npy',allow_pickle=True)
val = vals[index]
pXX = jnp.kron(pX,pX)
hamil = val[0]*pXX + val[1]*jnp.kron(pZ,jnp.eye(2)) + val[2]*jnp.kron(jnp.eye(2),pZ)

### Define Observables (full and shadow)
obs_0 = [-jnp.kron(pZ,jnp.eye(2)),jnp.kron(pY,pX)]
obs_1 = [jnp.kron(jnp.eye(2),pX),jnp.kron(jnp.eye(2),pY)]
obs_2 = [jnp.kron(pX,jnp.eye(2)),jnp.kron(pY,jnp.eye(2))]
p_0 = jnp.kron(obs_0[0],obs_0[1])
p_1 = jnp.kron(obs_1[0],obs_1[1])
p_2 = jnp.kron(obs_2[0],obs_2[1])

q_0 = jnp.kron(obs_0[1],obs_0[1])
q_1 = jnp.kron(obs_1[1],obs_1[1])
q_2 = jnp.kron(obs_2[1],obs_2[1])

full_obs = [[p_0,q_0],[p_1,q_1],[p_2,q_2]]

p_0_s = np.array([2,1,0]), np.array([0,2,3])
p_1_s = np.array([0,1]), np.array([1,3])
p_2_s = np.array([0,1]), np.array([0,2])

q_0_s = np.array([1,0,1,0]), np.array([0,1,2,3])
q_1_s = np.array([1,1]), np.array([1,3])
q_2_s = np.array([1,1]), np.array([0,2])

full_shadow_obs = [[p_0_s,q_0_s],[p_1_s,q_1_s],[p_2_s,q_2_s]]

### Load Data
ts = np.array([round(start + i*dt,4) for i in range(steps)])
shadow_obs_p_ts, shadow_obs_q_ts = shadow_loader.data_loader(ts,meas,index,full_shadow_obs,k=k)
full_data = jnp.concatenate([shadow_obs_p_ts,shadow_obs_q_ts],axis=-1)
print(full_data.shape)
train_data = full_data[::2]
train_times = ts[::2]
val_data = full_data[1::2]
val_times = ts[1::2]

data = [[train_data,train_times],[val_data,val_times]]

print(val_data.shape,val_times.shape)
print(train_data.shape,train_times.shape)


### Define network
net = (max_hid,max_hid//2,min_hid,max_hid//2)
model = DAE(layers=net,out=6)

import shadow_evolution_learning.train
from shadow_evolution_learning.train import main
full_predicts, val_shadows, coeff_shadow,test_times, val_losses, train_loss = main(data,model,hamil,full_obs,val,lr=0.01,seed=seed)

np.save(path+'/full_predicts.npy',full_predicts)
np.save(path+'/val_predicts.npy',val_shadows)
np.save(path+'/coeff_predicts.npy',coeff_shadow)
np.save(path+'/test_times.npy',test_times)
np.save(path+'/val_losses.npy',val_losses)
np.save(path+'/train_losses.npy',train_loss)


