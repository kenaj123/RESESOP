import numpy as np

"""
    This file contains functionality for hyperplain and stripe geometry, e.g.
        
        orthogonal projections,
        finding argmax <z, a> for z belonging to hyperplane/stripe.
"""

def projection_hyperplane(x, u, alpha):
    '''
        Projection on single hyplerplane.
    '''

    x_shape = x.shape
    x = x.flatten()
    u = u.flatten()

    t = (np.dot(u, x) - alpha) / np.dot(u,u)
    return (x - t * u).reshape(x_shape)


def projection_intersection_two_hyperplanes(f, u1, u2, a1, a2):
    '''
    Assumption: f belongs already to the hyperplane H(u1, a1). 
    Computes projection of f onto intersection of H(u1,a1) and H(u2, a2)
    '''
    nominator = (u1*u1).sum() * (u2 * u2).sum() - (u1 * u2).sum()**2
    if np.linalg.norm(nominator) == 0:
        print('Search directions are linear dependent!')
        return f
    T    = ((f * u2).sum() - a2) / nominator # ((u1*u1).sum() * (u2 * u2).sum() - (u1 * u2).sum()**2)
    t1   = - (u1 * u2).sum() * T
    t2   = (u1 * u1).sum() * T

    return f - t1*u1 - t2*u2


def projection_stripe(x, u, alpha, zeta):
    '''
    Projection on single stripe.
    '''

    x_shape = x.shape
    x = x.flatten()
    u = u.flatten()

    classifier = np.dot(u, x)
    if classifier > alpha + zeta:
        return projection_hyperplane(x, u, alpha + zeta).reshape(x_shape)
    elif classifier < alpha - zeta:
        return projection_hyperplane(x, u, alpha - zeta).reshape(x_shape)
    else: return x.reshape(x_shape)

    
def argmax_hyperplane(u, alpha, rho, a):
    """
    Consider the hyperplane H defined by

        z\in H iff <z, u> = alpha.

    This function returns that z in H such that

        ||z|| <= rho and <z, a> is maximal.
    """
    a_proj = projection_hyperplane(a, u, alpha)
    u_proj = alpha / np.linalg.norm(u)**2 * u  
    au_norm = np.linalg.norm(a_proj - u_proj)

    if au_norm == 0:
        return u_proj

    k      = (rho**2 - np.linalg.norm(u_proj)**2) / au_norm**2
    if np.sign(k) < 0:
        # Hyperplane is outside of B_rho(0)
        return None
    return u_proj + np.sqrt(k) * (a_proj - u_proj)


def is_in_stripe(x, u, alpha, zeta):
    c = np.abs( np.dot(x, u) - alpha )
    return (c<=zeta)


def argmax_stripe(u, alpha, zeta, rho, a):
    """
    Consider the stripe H defined by

        z\in H iff | <z, u> - alpha | <= zeta.

    This function returns that z in H such that

        ||z|| <= rho and <z, a> is maximal.
    """

    a_twiddle = rho / np.linalg.norm(a) * a
    if is_in_stripe(a_twiddle, u, alpha, zeta):
        return a_twiddle

    candidate_1 = argmax_hyperplane(u, alpha + zeta, rho, a)
    candidate_2 = argmax_hyperplane(u, alpha - zeta, rho, a)

    if candidate_1 is None:
        if candidate_2 is None:
            print("Stripe outside of B_rho(0)!")
            return None
        return candidate_2

    if candidate_2 is None:
        return candidate_1

    if np.dot(candidate_1, a) > np.dot(candidate_2, a):
        return candidate_1
    return candidate_2