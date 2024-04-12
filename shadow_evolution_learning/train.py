import jax
import jax.numpy as jnp
import numpy as np
from . import utils
from shadow_evolution_learning.utils.network_utils import construct_test_vals, create_train_state, train_step, periodic_actv_fn

def extract_params(predict_vals,time_shift,intervals):
    predict_vals = predict_vals.reshape((intervals,2,3))
    dt_shadow = (predict_vals[:-1,0,:] - predict_vals[1:,0,:])/time_shift
    coeff_shadow = dt_shadow/predict_vals[:-1,1,:]
    return coeff_shadow

def main(data,model,hamil,full_obs,val,lr=0.01,seed=0,iters=2000):

    train_data, train_times = data[0]
    val_data, val_times = data[1]

    train_loss = []
    full_predicts = []
    val_shadows = []
    val_losses = []
    full_coeffs = []

    key = jax.random.PRNGKey(seed)
    state = create_train_state(key,lr,model,train_times[...,None])
    test_times = np.linspace(train_times[0],train_times[-1],100)
    time_shift = test_times[11]-test_times[10]
    for i in range(iters):
       state, loss, preds = train_step(state,train_data,train_times[...,None],model)
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