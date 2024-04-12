import sys
import os
import numpy as np
import jax.numpy as jnp
from operations import denseXP, denseXM, denseYP, denseYM, denseZP, denseZM, zero_state, one_state, ket0, ket1, hadamard, phase_z, pX,pY, pZ
from time_evolution_simulator import timeEvolution
from shadowObs import classicalShadowCalc, estimate_shadow_obervable
import matplotlib.pyplot as plt
import numpy as np

from utils import commutator, makeRho
import jax
from jax.scipy.linalg import expm

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
    phi_psi = jnp.kron(jnp.eye(4),process)@alpha
    phi_rho = jnp.outer(phi_psi,phi_psi.conj().T)
    return phi_rho

def partial_trace(t,hamil,init_rho):
    phi_rho = process_mat(t,hamil)
    init_rho = jnp.kron(init_rho.T,jnp.eye(4))

    full = init_rho@phi_rho

    full = full.reshape((4,4,4,4))
    return jnp.trace(full,axis1=0,axis2=2)*4

