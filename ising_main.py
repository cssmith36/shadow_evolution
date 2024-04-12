import sys
import os
import numpy as np
import jax.numpy as jnp
from shadow_evolution_learning.operations import denseXP, denseXM, denseYP, denseYM, denseZP, denseZM, zero_state, one_state, ket0, hadamard, phase_z, pX,pY, pZ
from shadow_evolution_learning.time_evolution_simulator import timeEvolution
from shadow_evolution_learning.shadow_obs import classicalShadowCalc, estimate_shadow_obervable
import matplotlib.pyplot as plt
import numpy as np

from shadow_evolution_learning.utils import commutator, makeRho
import jax
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--index',type=int,default=0)
parser.add_argument('--measurements','-m',type=int,default=5000)
parser.add_argument('--t_init',type=float,default=0.0)
parser.add_argument('--steps',type=int,defualt=20)
parser.add_argument('--time_step',type=float,default=0.01)
parser.add_argument('--max_hid_units',type=int,default=10)
parser.add_argument('--k',type=int,default=30)

args = parser.parse_args()
index = args.index

### Load Value
val_path = '/users/csmith36/shadows/ising_shadow_data/ket_00/param_vals.npy'
vals = np.load(val_path, allow_pickle=True)
val = vals[index]

### Hamiltonian
hamil = val[0]*pXX + val[1]*jnp.kron(pZ,jnp.eye(2)) + val[2]*jnp.kron(jnp.eye(2),pZ)

measurements  = args.measurements
startTime = round(args.t_init,4)
steps = args.steps
tStep = round(args.time_step,4)
netStruct = args.max_hid_units
k = args.k

### Observables
oQ = np.array([1,2]),np.array([0,1])
oP = np.array([0]),np.array([0])
oQo = np.array(jnp.kron(pY,pZ))
oPo = np.array(jnp.kron(pX,jnp.eye(2)))
psi_init = jnp.kron(ket0,ket0)

shadow_obsVals = []

obsQ = oQ
obsP = oP
### Build Network Structure
net = (int(netStruct),int(netStruct/2),1,int(netStruct/2),int(netStruct))


trainTimes = np.array([round(startTime + i*tStep,4) for i in range(steps)])
times = trainTimes
for t in times:
    path_t = '/users/csmith36/shadows/shadow_data/ket_00_errZZ='+str(errZZ)+'/t='+str(round(t,4))+'/meausurements='+str(measurments)+'/shadow.npy'
    rhoShadow = np.load(path_t,allow_pickle=True)
    shadow_obsVals.append([estimate_shadow_obervable(rhoShadow,obsP,k),estimate_shadow_obervable(rhoShadow,obsQ,k)])

from shadow_evolution_learning.networks import DAE
from shadow_evolution_learning.networkUtils import createTrainState, trainStep

def testData(test_times,model,state):
    finOut = model.apply(state.params,test_times[...,None])
    obsQ,obsP = construct_exact(test_times)
    return finOut,obsQ,obsP

def construct_exact(test_times):
    obsValsP = []
    obsValsQ = []
    for t in test_times:
        rho = makeRho(timeEvolution(psi_init,hamil,t))
        obsValsP.append(np.trace(rho@oPo))
        obsValsQ.append(np.trace(rho@oQo))
    obsValsP = np.array(obsValsP)
    obsValsQ = np.array(obsValsQ)
    return obsValsQ,obsValsP

iters = 2000
trainData = jnp.array(shadow_obsVals)
seed = np.random.randint(1,10000)

def main():
    trainLoss = []
    testLoss = []
    fullPreds = []
    truePreds = []
    coeffPreds = []
    key = jax.random.PRNGKey(seed)
    lr = 0.01
    model = DAE(layers=net)
    state = createTrainState(key,lr,model,trainTimes[...,None])
    time_diff = trainTimes[-1] - trainTimes[0]
    test_times = np.linspace(trainTimes[0],trainTimes[-1],100)
    tShift = test_times[11]-test_times[10]
    for i in range(iters):
       state, loss, preds = trainStep(state,trainData,trainTimes[...,None],model)
       #print('step:',i)
       if i%50==0:
        print('i:',i,loss)
        finOut,obsQ,obsP = testData(test_times,model,state)
        fullPreds.append(finOut)
        dt_shadow = (finOut[:-1,0] - finOut[1:,0])/tShift
        coeff_shadow = dt_shadow/finOut[:-1,1]
        print("predicted median:",np.median(np.sort(coeff_shadow)[1:-1]))
        dt = (obsP[:-1] - obsP[1:])/tShift
        coeff = dt/obsQ[:-1]
        print('exact median:',np.median(coeff))
        coeffPreds.append(coeff_shadow)
       trainLoss.append(float(loss))
    truePreds.append([obsQ,obsP])
    return preds,fullPreds,truePreds,obsQ,obsP,coeffPreds,trainTimes,test_times,trainLoss, testLoss, model, state
preds,fullPreds,truePreds,obsQ,obsP,coeffPreds,trainTimes,test_times,trainLoss, testLoss, model, state = main()

### Analyze Data

path = '/users/csmith36/shadows/NNSHL/results'
if not os.path.exists(path):
    os.mkdir(path)
path = path + '/DAE_extensive'
if not os.path.exists(path):
    os.mkdir(path)
path = path + '/errZZ='+str(errZZ)
if not os.path.exists(path):
    os.mkdir(path)
path = path + '/net='+str(net)
if not os.path.exists(path):
    os.mkdir(path)
path = path + '/measurements='+str(measurments)
if not os.path.exists(path):
    os.mkdir(path)
path = path + '/k='+str(k)
if not os.path.exists(path):
    os.mkdir(path)
path = path + '/start='+str(startTime) + '_tStep='+str(tStep)+'_steps='+str(steps)
if not os.path.exists(path):
    os.mkdir(path)
path = path + '/seed='+str(seed)  
if not os.path.exists(path):
    os.mkdir(path)

preds,fullPreds,coeffPreds,trainTimes,test_times,trainLoss, testLoss, model, state
np.save(path + '/fullPreds.npy',fullPreds)
np.save(path + '/trueObsP.npy',obsP)
np.save(path + '/trueObsQ.npy',obsQ)
np.save(path + '/coeff_shadow.npy',coeffPreds)
np.save(path + '/trainLoss.npy',trainLoss)
np.save(path + '/trainTimes.npy',trainTimes)
np.save(path + '/test_times.npy',test_times)



#print(np.sort(coeff_shadow))