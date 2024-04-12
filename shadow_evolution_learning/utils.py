import jax.numpy as jnp
import numpy as np

def construct_hermitian_matrix(rho):
    indU = jnp.triu_indices(4,k=0)
    indL = jnp.tril_indices(4,k=0)
    indD = jnp.diag_indices(4)
    
    matRho = jnp.zeros((4,4),dtype=complex)

    matRho = matRho.at[indU].set(rho)
    matRho = matRho.at[indL].set(rho)
    matRho = matRho.at[indD].set(jnp.real(jnp.diag(matRho)))
    matRho = matRho/jnp.trace(matRho)
    return matRho

def commutator(A,B):
    return A@B - B@A

def makeRho(psi):
    return np.outer(psi,psi.conj().T)