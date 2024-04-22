import jax
import jax.numpy as jnp
import flax.linen as nn
from .utils.network_utils import create_train_state, train_step, construct_exact_vals
from .utils.time_evolution_simulator import timeEvolution


def construct_batch(key,train_data,batch_size):
    order = jnp.array([i for i in range(len(train_data))])
    key = jax.random.split(key,num=1)[0]
    order = jax.random.permutation(key,order)
    return train_data[order[:batch_size]], key

def test_accuracy(model,hamil,train_times):
    return
def main(noisy_data,clean_data,model,train_times,hamil,full_obs,lr=0.01,seed=0,iters=2000,batch_size=3):
    train_data = noisy_data

    clean_loss = []
    exact_loss = []
    
    key = jax.random.PRNGKey(seed)

    state = create_train_state(key,lr,model,train_data) 

    for i in range(iters):
        batch_data, key = construct_batch(key,train_data,batch_size)
        state, loss, preds = train_step(state,clean_data,batch_data,model)
        exact_obs = construct_exact_vals(train_times,hamil,full_obs).transpose(1,0)[None,...]
        #print('exact_obs.shape',exact_obs.shape)
        #print('preds shape',preds.shape)

        pred_clean = model.apply(state.params,clean_data)
        clean = jnp.sum(jnp.abs(pred_clean - clean_data))
        exact = jnp.sum(jnp.abs(exact_obs - pred_clean))
        clean_loss.append(clean)
        exact_loss.append(exact)
        print('clean loss:',clean)
        print('exact loss:',exact)
        if i ==0:
            exact_clean = jnp.sum(jnp.abs(exact_obs - clean_data))
        print('exact-clean loss:',exact_clean)
    return clean_loss, exact_loss, exact_clean


        

'''def create_train_state(key,lr,model,obsVal):
    ###Create Training State
    key1,key2 = jax.random.split(key,2)
    params = model.init(key1,obsVal)
    #sgdOpt = optax.sgd(lr,momentum=0.9)
    sgdOpt = optax.adam(lr)
    return train_state.TrainState.create(apply_fn=model.apply,params=params,tx=sgdOpt)'''
