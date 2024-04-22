import numpy as np
import jax
import jax.numpy as jnp
from ..network_utils import create_train_state


def lossfn(truth,pred):
    return jnp.mean(jnp.abs(pred-truth)**2)

def train_step(state,lossData,trainInp,model):

    def loss_fn(params):        
        preds = model.apply(params,trainInp)
        return jnp.mean(jnp.abs(preds-lossData)**2), preds

    (loss,preds), grads = jax.value_and_grad(loss_fn, has_aux=True, argnums=0)(state.params)

    state = state.apply_gradients(grads = grads)
    return state, loss, preds