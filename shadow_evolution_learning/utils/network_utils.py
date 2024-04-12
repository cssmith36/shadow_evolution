import flax.linen as nn
from typing import Sequence, Callable
import optax
import jax
import jax.numpy as jnp
import numpy as np
from .utils import makeRho
from flax.training import train_state
from .time_evolution_simulator import timeEvolution
from .operations import denseXP, denseXM, denseYP, denseYM, denseZP, denseZM, zero_state, one_state, ket0, ket1, hadamard, phase_z, pX,pY, pZ
from ..shadow_sampling.shadow_obs import estimate_shadow_observable

from jax.scipy.linalg import expm


def create_train_state(key,lr,model,obsVal):
    '''Create Training State'''
    key1,key2 = jax.random.split(key,2)
    params = model.init(key1,obsVal)
    sgdOpt = optax.sgd(lr,momentum=0.9)
    sgdOpt = optax.adam(lr)
    return train_state.TrainState.create(apply_fn=model.apply,params=params,tx=sgdOpt)

def lossfn(truth,pred):
    print(pred,truth)
    return jnp.mean(jnp.abs(pred-truth)**2)

def train_step(state,lossData,trainInp,model):

    def loss_fn(params):        
        preds = model.apply(params,trainInp)
        return jnp.mean(jnp.abs(preds-lossData)**2), preds

    (loss,preds), grads = jax.value_and_grad(loss_fn, has_aux=True, argnums=0)(state.params)

    state = state.apply_gradients(grads = grads)
    return state, loss, preds

def dataLoader(path_t,oP,oQ,k):
    rhoShadow = np.load(path_t,allow_pickle=True)
    return [estimate_shadow_observable(rhoShadow,oP,k), estimate_shadow_observable(rhoShadow,oQ,k)]

def construct_init_rho():
    ket00 = jnp.kron(ket0,ket0)
    ket11 = jnp.kron(ket1,ket1)
    ket01 = jnp.kron(ket0,ket1)
    ket10 = jnp.kron(ket1,ket0)
    alpha = 1/np.sqrt(4) * (jnp.kron(ket00,ket00) + jnp.kron(ket11,ket11) + jnp.kron(ket01,ket01) + jnp.kron(ket10,ket10))
    return alpha

def process_mat(t,hamil):
    process = expm(-1.j*hamil*t)
    alpha = construct_init_rho()
    phi = jnp.kron(jnp.eye(4),process)@alpha
    phi_rho = jnp.outer(phi,phi.conj().T)
    return phi_rho

def partial_trace_plus_construct(t,hamil,init_rho):
    phi_rho = process_mat(t,hamil)
    init_rho = jnp.kron(init_rho.T,jnp.eye(4))

    full = init_rho@phi_rho

    full = full.reshape((4,4,4,4))
    return jnp.trace(full,axis1=0,axis2=2)*4
def partial_trace(rho):
    rho = rho.reshape((4,4,4,4))
    return jnp.trace(rho,axis1=0,axis2=2)

def construct_exact_vals(test_times,hamil,obs):
    full_obs = []
    for t in test_times:
        phi_rho_t = process_mat(t,hamil)

        obsValsP = []
        obsValsQ = []
        for o in obs:
            obsValsP.append(jnp.trace(partial_trace(o[0]@phi_rho_t)))
            obsValsQ.append(np.trace(partial_trace(o[1]@phi_rho_t)))
        full_obs.append([obsValsP,obsValsQ])
    full_obs = np.array(full_obs)
    return full_obs

def construct_test_vals(test_times,model,state,hamil,obs):
    finOut = model.apply(state.params,test_times[...,None])
    exact_obs = construct_exact_vals(test_times,hamil,obs)
    return finOut,exact_obs

def periodic_actv_fn(x):
    return x + jnp.sin(x)**2