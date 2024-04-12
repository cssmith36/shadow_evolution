import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
from operations import denseXP, denseXM, denseYP, denseYM, denseZP, denseZM, zero_state, one_state, ket0, hadamard, phase_z, pX,pY, pZ
from time_evolution_simulator import timeEvolution
from shadowsObs_gen import classicalShadowCalc, estimate_shadow_obervable
from choi_construction import process_mat

vals = np.load('/users/csmith36/shadows/NNSHL/ising/ising_shadow_data/params.npy',allow_pickle=True)
vals = vals.round(2)

times = [round(0.005 + i*0.005,5) for i in range(400)]
measurements = [1000,2500,5000,10000,15000,20000,25000]

ind = int(sys.argv[1])
seed = int(sys.argv[2])
m = 10000
vals = [vals[ind]]
print('here!',vals)
#quit()
pZZ = jnp.kron(pZ,pZ)
pXX = jnp.kron(pX,pX)


#hamil = 2*jnp.pi*0.222 *(jnp.kron(pX,jnp.eye(2)) + jnp.kron(jnp.eye(2),pX)) + errZZ*pZZ

num_qubits = 4
index = ind
psi_init = jnp.kron(ket0,ket0)
for val in vals:
    for t in times:
        #hamil = 2*jnp.pi*0.222 *(jnp.kron(pX,jnp.eye(2)) + jnp.kron(jnp.eye(2),pX)) + errZZ*pZZ
        hamil = val[0]*pXX + val[1]*jnp.kron(pZ,jnp.eye(2)) + val[2]*jnp.kron(jnp.eye(2),pZ)
        phi_rho_t = process_mat(t,hamil)
        
        rhoShadow_t1 = classicalShadowCalc(phi_rho_t,int(m),num_qubits,seed=seed)
        path = '/users/csmith36/shadows/NNSHL/ising/ising_shadow_data'
        if not os.path.exists(path):
            os.mkdir(path)
        path += '/ket_00'
        if not os.path.exists(path):
            os.mkdir(path)

        path += '/index='+str(index)
        if not os.path.exists(path):
            os.mkdir(path)
        path = path + '/t='+str(round(t,4))
        if not os.path.exists(path):
            os.mkdir(path)
        path = path + '/measurements='+str(m)
        if not os.path.exists(path):
            os.mkdir(path)
        path = path + '/seed='+str(seed)
        if not os.path.exists(path):
            os.mkdir(path)
        np.save(path+'/shadow.npy',rhoShadow_t1)
    index += 1

    print(index)
    print(val)
    print(path)
