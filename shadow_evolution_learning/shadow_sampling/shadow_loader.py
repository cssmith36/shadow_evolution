import numpy as np
from .shadow_obs import estimate_shadow_observable

def data_loader(times,measurements,index,full_shadow_obs,k=10):
    shadow_obs_p_ts = []
    shadow_obs_q_ts = []
    for t in times:
        shadow = np.load('/users/csmith36/shadows/NNSHL/ising/ising_shadow_data/ket_00/index='+str(index)+'/t='+str(round(t,3))+'/measurements='+str(measurements)+'/seed=0/shadow.npy',allow_pickle=True)
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