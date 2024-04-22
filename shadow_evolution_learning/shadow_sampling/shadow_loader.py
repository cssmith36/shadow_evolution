import numpy as np
from .shadow_obs import estimate_shadow_observable
import jax.numpy as jnp
import jax

def data_loader(times,measurements,index,full_shadow_obs,k=10,base_path='/users/csmith36/shadows/NNSHL/ising/ising_shadow_data/ket_00'):
    shadow_obs_p_ts = []
    shadow_obs_q_ts = []
    for t in times:
        shadow = np.load(base_path+'/index='+str(index)+'/t='+str(round(t,3))+'/measurements='+str(measurements)+'/seed=0/shadow.npy',allow_pickle=True)
        temp_p_ts_shad = []
        temp_q_ts_shad = []
        for o in full_shadow_obs:
            shadow_p = estimate_shadow_observable(shadow,o[0],k=k)
            shadow_q = estimate_shadow_observable(shadow,o[1],k=k)
            temp_p_ts_shad.append(shadow_p)
            temp_q_ts_shad.append(shadow_q)
        shadow_obs_p_ts.append(temp_p_ts_shad)
        shadow_obs_q_ts.append(temp_q_ts_shad)

    shadow_obs_p_ts = np.array(shadow_obs_p_ts)
    shadow_obs_q_ts = np.array(shadow_obs_q_ts)
    return shadow_obs_p_ts, shadow_obs_q_ts

def partition_function(shadow,groups=500):
    shp = shadow.shape
    return shadow.reshape((shp[0],groups,-1,4)).transpose((1,0,2,3))

def load_and_partition(key,groups,times,measurements,index,full_shadow_obs,perms=10,k=10,base_path='/users/csmith36/shadows/NNSHL/ising/ising_shadow_data/ket_00'):
    full_p = []
    full_q = []
    for t in times:
        shadow = np.load(base_path+'/index='+str(index)+'/t='+str(round(t,3))+'/measurements='+str(measurements)+'/seed=0/shadow.npy',allow_pickle=True)
        perm_vals_p = []
        perm_vals_q = []
        for p in range(perms):
            shadow = jax.random.permutation(key,shadow,axis=1)
            part_shadow = partition_function(shadow,groups=groups)
            part_shad_p = []
            part_shad_q = []
            for s in part_shadow:
                q_vals = []
                p_vals = []
                for o in full_shadow_obs:
                    shadow_p = estimate_shadow_observable(s,o[0],k=k)
                    shadow_q = estimate_shadow_observable(s,o[1],k=k)
                    p_vals.append(shadow_p)
                    q_vals.append(shadow_q)
                part_shad_p.append(p_vals)
                part_shad_q.append(q_vals)
            perm_vals_p.append(part_shad_p)
            perm_vals_q.append(part_shad_q)
        full_p.append(perm_vals_p)
        full_q.append(perm_vals_q)
    return jnp.array(full_p), jnp.array(full_q)
    #print(jnp.array(full_p),jnp.array(full_q))
    #return jnp.array(full_p).transpose(1,0,2), jnp.array(full_q).transpose(1,0,2)
