import sys
import os
sys.path.append('/users/csmith36/shadows')
import numpy as np
import jax.numpy as jnp
from shadow_evolution_learning.operations import denseXP, denseXM, denseYP, denseYM, denseZP, denseZM, zero_state, one_state, ket0, ket1, hadamard, phase_z, pX,pY, pZ
from shadow_evolution_learning.time_evolution_simulator import timeEvolution
from shadow_evolution_learning.shadow_obs import classicalShadowCalc, estimate_shadow_observable
import matplotlib.pyplot as plt
import numpy as np

from shadow_evolution_learning.utils import commutator, makeRho
import jax
from jax.scipy.linalg import expm

from shadow_evolution_learning.network_utils import process_mat, partial_trace,construct_exact_vals, construct_test_vals, createTrainState, trainStep
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


vals = np.load('/users/csmith36/shadows/NNSHL/ising/ising_shadow_data/params.npy',allow_pickle=True)
val = vals[index]
pXX = jnp.kron(pX,pX)
hamil = val[0]*pXX + val[1]*jnp.kron(pZ,jnp.eye(2)) + val[2]*jnp.kron(jnp.eye(2),pZ)

### Define Observables

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

shadow_obs_p_ts = []
shadow_obs_q_ts = []
val = vals[index]
print(val)
hamil = val[0]*pXX + val[1]*jnp.kron(pZ,jnp.eye(2)) + val[2]*jnp.kron(jnp.eye(2),pZ)
for t in ts:
    phi_rho_t = process_mat(t,hamil)
    shadow = np.load('/users/csmith36/shadows/NNSHL/ising/ising_shadow_data/ket_00/index='+str(index)+'/t='+str(round(t,3))+'/measurements='+str(meas)+'/seed=0/shadow.npy',allow_pickle=True)
    temp_p_ts_shad = []
    temp_q_ts_shad = []
    for os in full_shadow_obs:
        shadow_p = estimate_shadow_observable(shadow,os[0],k=k)
        shadow_q = estimate_shadow_observable(shadow,os[1],k=k)
        temp_p_ts_shad.append(shadow_p)
        temp_q_ts_shad.append(shadow_q)
    shadow_obs_p_ts.append(temp_p_ts_shad)
    shadow_obs_q_ts.append(temp_q_ts_shad)
shadow_obs_p_ts = np.array(shadow_obs_p_ts)
shadow_obs_q_ts = np.array(shadow_obs_q_ts)

full_data = np.concatenate([shadow_obs_p_ts,shadow_obs_q_ts],axis=-1)
train_data = full_data[::2]
train_times = ts[::2]
val_data = full_data[1::2]
val_times = ts[1::2]

def extract_params(predict_vals,time_shift,intervals):
    predict_vals = predict_vals.reshape((intervals,2,3))
    dt_shadow = (predict_vals[:-1,0,:] - predict_vals[1:,0,:])/time_shift
    coeff_shadow = dt_shadow/predict_vals[:-1,1,:]
    return coeff_shadow

net = (max_hid,max_hid//2,min_hid,max_hid//2)

def periodic_actv_fn(x):
    return x + jnp.sin(x)**2

def main():
    train_loss = []
    full_predicts = []
    val_shadows = []
    val_losses = []
    full_coeffs = []

    key = jax.random.PRNGKey(seed)
    model = DAE(layers=net,out=6,act=periodic_actv_fn)
    state = createTrainState(key,lr,model,train_times[...,None])
    test_times = np.linspace(train_times[0],train_times[-1],100)
    time_shift = test_times[11]-test_times[10]
    for i in range(iters):
       state, loss, preds = trainStep(state,train_data,train_times[...,None],model)
       #print('step:',i)
       if i%1==0:
        print('i:',i,loss)

        predict,exact_obs = construct_test_vals(test_times,model,state,hamil,full_obs)
        exact_obs = exact_obs.reshape((len(test_times),-1))
        predict_val, exact_val = construct_test_vals(val_times,model,state,hamil,full_obs)

        exact_val = exact_val.reshape((len(train_data),-1))
        #print("shapes:",finOutVal[:-1,0].shape,finOutVal[:-1,1].shape,obsQval.shape,obsPval.shape)
        val_loss = jnp.abs(predict_val - val_data)
        print('val_loss:', jnp.mean(val_loss))
        val_losses.append(val_loss)
        val_shadows.append(predict_val)

        full_predicts.append(predict)

        plot=False
        if plot:
            if i%20 == 0:
                for ii in range(6):
                    plt.plot(test_times,predict[:,ii],color='blue',label='predict '+str(ii))
                    plt.plot(test_times,exact_obs[:,ii],color='green',label='exact '+str(ii))
                plt.legend()
                plt.savefig('figs/test_plot_'+str(i)+'.png')
                plt.clf()
        
        coeff_shadow = extract_params(predict,time_shift,len(test_times))
        full_coeffs.append(coeff_shadow)
        #print(val)
        print('median diff:',np.median(coeff_shadow,axis=-2)[np.array([0,2,1])]/2-val)
        #print('mean diff:',np.mean(np.sort(coeff_shadow)[30:-30],axis=-2)[np.array([0,2,1])]/2-val)

       train_loss.append(float(loss))
    return full_predicts, val_shadows, full_coeffs ,test_times, val_losses, train_loss
full_predicts, val_shadows, coeff_shadow,test_times, val_losses, train_loss = main()

np.save(path+'/full_predicts.npy',full_predicts)
np.save(path+'/val_predicts.npy',val_shadows)
np.save(path+'/coeff_predicts.npy',coeff_shadow)
np.save(path+'/test_times.npy',test_times)
np.save(path+'/val_losses.npy',val_losses)
np.save(path+'/train_losses.npy',train_loss)


