import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Callable
import optax
from flax.training import train_state

def periodic_actv_fn(x):
    return x + jnp.sin(x)**2

class FFN(nn.Module):
    layers: Sequence = (30,20,10)
    act: Callable = nn.activation.gelu

    @nn.compact
    def __call__(self,x):
        for l in self.layers:
            x = self.act(nn.Dense(l)(x)) + self.act(1.j*nn.Dense(l)(x))
        return nn.Dense(4)(x)

class DAE(nn.Module):
    out: int = 6
    layers: Sequence = (50,10,5,10,30)
    act: Callable = nn.activation.gelu

    @nn.compact
    def __call__(self,x):
        for l in self.layers:
            x = self.act(nn.Dense(l)(x))
        return nn.Dense(self.out)(x)

class DAE_0(nn.Module):
    layers: Sequence = (20,10,5,10,20)
    act: Callable = nn.activation.gelu

    @nn.compact
    def __call__(self,x):
        for l in self.layers:
            x = self.act(nn.Dense(l)(x))
        return nn.Dense(x.shape[-1])

class DAE_periodic(nn.Module):
    out: int = 6
    layers: Sequence = (50,10,5,10,30)
    act: Callable = periodic_actv_fn

    @nn.compact
    def __call__(self,x):
        for l in self.layers:
            x = self.act(nn.Dense(l)(x))
        return nn.Dense(self.out)(x)