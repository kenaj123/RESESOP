import numpy as np

from .geometry import (
    projection_hyperplane,
    projection_intersection_two_hyperplanes,
    projection_stripe,
    argmax_stripe
)



class resesop_regularizer():
    def __init__(self, operators, adjoints, number_operators, g, etas, deltas, f0 = None, groundtruth=None):
        self.operators = operators
        self.adjoints  = adjoints
        self.g         = g
        self.etas      = etas
        self.deltas    = deltas
        self.number_operators = number_operators
        self.f0          = f0 # current iterate or start iterate
        self.number_of_sweeps = 0
        self.errors      = []
        self.groundtruth = groundtruth 
        
    def reset(self):
        self.f0 = None
        self.errors = []
        
    def set_up_subproblems(self, operators, adjoints, number_operators, g, etas, deltas):
        """
            operators(k, x) yields the kth operator evaluated at x,
            g[k] is the right-hand-side of the respective subproblem.
        """
        self.operators = operators
        self.adjoints  = adjoints
        self.g         = g
        self.etas      = etas
        self.deltas    = deltas
        self.number_operators = number_operators
   

    def resesop_one_search_direction(self, rho, tau=1.00001, sweeps=50, f0=None, compute_errors=False):

        n = self.number_operators
        
        if f0 is None:
            if self.f0 is None:
                raise ValueError('f0 must be set!')
            f_n = self.f0
        else: 
            self.number_of_iterations = 0
            f_n = f0

        counter = 0
        number_updates = 0
        for j in range(sweeps):
            perm = np.random.permutation(np.arange(self.number_operators))
            for op_index in perm:
                g_noisy     = self.g[op_index]
                eta         = self.etas[op_index]
                delta       = self.deltas[op_index]
                w           = self.operators(op_index, f_n) - g_noisy
                discrepancy = np.linalg.norm(w)
                if discrepancy > tau * (eta * rho + delta):
                    number_updates += 1
                    counter         = 0
                    alpha           = (w * g_noisy).sum()
                    zeta            = (eta * rho + delta) * discrepancy
                    u               = self.adjoints(op_index, w)
                    if np.linalg.norm(u)==0:
                        f_n = f_n
                        #return f_n
                    else:
                        f_n             = projection_stripe(f_n, u, alpha, zeta)
                    
                else:
                    counter += 1
                    if counter >= self.number_operators:
                        print("Algorithm terminated succesfully!")
                        print(f"Number of non-stationary iteration steps: {number_updates}.")
                        return f_n
            if compute_errors: 
                self.errors.append(np.linalg.norm(f_n - self.groundtruth)/np.linalg.norm(self.groundtruth))
        self.f0 = f_n
        return f_n
    
    
    def resesop_two_search_directions(self, rho, tau=1.000001, sweeps=50, f0=None, compute_errors=False):

        if f0 is None:
            if self.f0 is None:
                raise ValueError('f0 must be set!')
            f_n = self.f0
        else: 
            self.number_of_iterations = 0
            f_n = f0
        
        # Set zeta_previous to infinity. Then H(u_prev, alpha_prev, zeta_prev) is the full space.
        alpha_previous = 0
        zeta_previous  = np.inf
        u_previous     = f_n
        
        counter = 0
        number_updates = 0
        for j in range(sweeps):
            perm = np.random.permutation(np.arange(self.number_operators))
            for op_index in perm:
                g_noisy     = self.g[op_index]
                eta         = self.etas[op_index]
                delta       = self.deltas[op_index]
                w           = self.operators(op_index, f_n) - g_noisy
                discrepancy = np.linalg.norm(w)

                if discrepancy > tau * (eta * rho + delta):
                    # Step (a)
                    number_updates += 1
                    counter         = 0
                    alpha           = (w * g_noisy).sum()
                    zeta            = (eta * rho + delta) * discrepancy
                    u               = self.adjoints(op_index, w)
                    f_n_twiddle     = projection_hyperplane(f_n, u, alpha + zeta) #projection_stripe(f_n, u, alpha, zeta)
                    # Step (b)
                    c               = (f_n_twiddle * u_previous).sum()
                    # Next update
                    if c > alpha_previous + zeta_previous:
                        f_n = projection_intersection_two_hyperplanes(f_n_twiddle, u, u_previous, alpha + zeta, alpha_previous + zeta_previous)
                    elif c < alpha_previous - zeta_previous:
                        f_n = projection_intersection_two_hyperplanes(f_n_twiddle, u, u_previous, alpha + zeta, alpha_previous - zeta_previous)
                    else: 
                        f_n = f_n_twiddle
                        
                    alpha_previous        = alpha
                    zeta_previous         = zeta
                    u_previous            = u 
                else:
                    counter += 1
                    if counter >= self.number_operators:
                        print("Algorithm terminated succesfully!")
                        print(f"Number of non-stationary iteration steps: {number_updates}.")
                        return f_n
            if compute_errors: 
                self.errors.append(np.linalg.norm(f_n - self.groundtruth))
            #print("Algorithm has not yet terminated, use more sweeps!")
            #print(f"Number of non-stationary iteration steps: {number_updates}.")
            
        self.number_of_sweeps += sweeps
        self.f0 = f_n
        return f_n