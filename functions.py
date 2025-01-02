##### Dropbox Problem

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import scipy
import os
import time
import pickle
import cProfile
import pstats
import io
import numba
import dill

from numpy import logaddexp
from scipy.stats import norm, gumbel_r
from scipy.integrate import quad
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
from scipy.stats import binom
from cobyqa import minimize as cob_min
from joblib import Parallel, delayed, parallel_backend
from concurrent.futures import ProcessPoolExecutor, as_completed
# from multiprocessing import Pool
from line_profiler import LineProfiler
from memory_profiler import memory_usage
# from tqdm import tqdm
# from tqdm_joblib import tqdm_joblib
from functools import partial
from scalene import scalene_profiler
# from scalene import profile 

import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    

#### Preparatory Functions

## Utility function
def utility(x, d, nu, r, y, params, P):
    theta, alpha, alpha0, rho1, rho2, gamma = params
    return (theta * x) + (alpha * d**2) + (alpha0 * d * nu) + (rho1 * r) + (rho2 * r**2) + (gamma * P * y)


## Logsumexp
def logsumexp(xvec):
    xvec = np.array(xvec)
    if np.sum(np.isfinite(xvec)) > 0:
        xmax = np.max(xvec)
        xnorm = xvec - xmax
        rval = xmax + np.log(np.sum(np.exp(xnorm)))
    else:
        rval = -np.inf
    return rval


## Probabilities for referral acceptance
def binomial_probabilities(n, p):
    k_values = np.arange(0, n+1)
    probabilities = binom.pmf(k_values, n, p)
    return list(probabilities)


# Addition as a function of x and z
def a_probs(g_avalues, x, z, x_threshold, k1=0.1, k2=0.2, b=0.05, p1=2, p2=2, q=2):
    """
    Calculate probabilities for each a in g_avalues based on x, z, and x_threshold.
    
    Parameters:
    - g_avalues: list or array of a values
    - x: total memory used
    - z: days left until the end of the subscription
    - x_threshold: memory threshold
    - k1: decay rate before threshold
    - k2: decay rate after threshold
    - b: growth rate with respect to z
    - p1: power for x_total before threshold
    - p2: power for x_total after threshold
    - q: power for z
    
    Returns:
    - probs: normalized probabilities corresponding to each a in g_avalues
    """
    x_total = x + np.array(g_avalues)  # Calculate x_total for each a in g_avalues
    
    # Apply growth/decay based on x_total and threshold
    s = np.where(
        x_total <= x_threshold,
        k1 * x_total**p1,  # Before threshold
        k1 * x_threshold**p1 + k2 * (x_total - x_threshold)**p2  # After threshold
    )
    
    # Calculate unnormalized probabilities
    P = np.exp(-s + b * z**q)
    
    # Normalize probabilities
    total_P = np.sum(P)
    if total_P > 0:
        probs = P / total_P
    else:
        # Handle cases where sum of probs is zero
        probs = np.full(len(g_avalues), 1 / len(g_avalues))
    
    return probs


#### Value Function Iteration

## Main Function
def value_function_iteration2(x, z, R, Ra, g_avalues, d, y, r, params, g_nuvalues, g_nuprobs, ra_probs, beta, P, m_bonus, xbase_threshold, max_ref, g_x_interval, max_x, tol):
    
    V = np.zeros((len(x), len(z), len(R), len(Ra)))
    V_new = np.full_like(V, -1e-10)
    iteration = 0

    x_indices = np.arange(len(x))
    z_indices = np.arange(len(z))
    R_indices = np.arange(len(R))
    Ra_indices = np.arange(len(Ra))

    while True:
        iteration += 1
        for i, xt in enumerate(x):
            for l, zt in enumerate(z):
                for m, Rt in enumerate(R):
                    for n, Rat in enumerate(Ra):
                        if Rt - Rat < 0:
                            continue
                        
                        ### Precompute values that don’t change in inner loops
                        ra_probs_val = ra_probs[Rt - Rat][1]
                        value_list = []
                        
                        for rat, rap in enumerate(ra_probs_val):
                            Ra_prime = min(max_ref, Rat + rat)
                            x_threshold = xbase_threshold + (Ra_prime * g_x_interval) * m_bonus
                            g_aprobs = a_probs(g_avalues, xt, zt, x_threshold, k1=0.1, k2=0.2, b=0.05, p1=2, p2=2, q=2)
                            
                            for at, ap in zip(g_avalues, g_aprobs):
                                for nut, nup in zip(g_nuvalues, g_nuprobs):
                                    for yt in y:
                                        if yt == 1 and zt != 0:
                                            continue
                                        for rt in r:
                                            R_prime = min(max_ref, Rt + rt)
                                            R_prime_idx = np.searchsorted(R, R_prime)
                                            Ra_prime_idx = np.searchsorted(Ra, Ra_prime)
                                            for dt in d:
                                                x_prime = xt + at - dt
                                                if x_prime < 0 or x_prime > max_x:
                                                    continue
                                                if x_prime > x_threshold and zt == 0 and yt == 0:
                                                    continue
                                                x_prime_idx = np.searchsorted(x, x_prime)
                                                
                                                if yt == 1:
                                                    future_value = beta * V[x_prime_idx, min(z), R_prime_idx, Ra_prime_idx]
                                                else:
                                                    future_value = beta * V[x_prime_idx, min(len(z) - 1, max(z) - zt + 1), R_prime_idx, Ra_prime_idx]
                                                
                                                current_utility = utility(xt, dt, nut, rt, yt, params, P)
                                                value = current_utility + future_value
                                                value_list.append(value * rap * nup)
                        
                        if value_list:
                            V_new[i, l, m, n] = logsumexp(value_list) * ap
                        
        diff = np.abs(V_new - V)
        if np.max(diff) < tol:
            break
        V = V_new.copy()
    
    Vn = np.where(V == -1e-10, -np.inf, V)
    return Vn


#### Problem Functions

## Data generation
def generate_simulation_data(
    V, x, z, R, Ra, g_avalues, d, y, r, params, g_nuvalues, g_nuprobs, ra_probs, p,
    beta, P, m_bonus, xbase_threshold, max_ref, g_x_interval, max_x, tol, N):
    x_sim = [0]  # Starting value of x
    z_sim = [0] 
    R_sim = [0]
    Ra_sim = [0]
    d_sim = []     
    y_sim = []  
    r_sim = []
    ra_sim = []
    at_sim = []  
    nu_sim = []
    
    
    nu = np.random.choice(g_nuvalues, size=N, p=g_nuprobs)
    e = gumbel_r.rvs(loc=0, size=N, scale=1)
    
    for t in range(N):
        xt = x_sim[-1]
        zt = z_sim[-1]
        Rt = R_sim[-1]
        Rat = Ra_sim[-1]
        
        if Rt-Rat < 0: ### Inside loop or here?
            continue
                    
        rat = np.random.binomial(Rt-Rat, p)
        Ra_prime = min(max_ref, Rat+rat)
        x_threshold = xbase_threshold + Ra_prime*g_x_interval*m_bonus
        
        g_aprobs = a_probs(g_avalues, xt, zt, x_threshold, k1=0.1, k2=0.2, b=0.05, p1=2, p2=2, q=2)
        a = np.random.choice(g_avalues, size=N, p=g_aprobs)
        at = a[t]
        nut = nu[t]
        et = e[t]
        
        # Find the optimal y and d that maximizes the value function
        best_action_value = -np.inf
        best_yt = None
        best_dt = None
        best_rt = None
        
        for dt in d:
            for yt in y:
                for rt in r:
                    
                    R_prime = min(max_ref, Rt+rt)

                    R_prime_idx = np.searchsorted(R, R_prime)
                    Ra_prime_idx = np.searchsorted(Ra, Ra_prime)

                    ## -Constrain 1
                    x_prime = xt + at - dt
                    if x_prime < 0 or x_prime > max_x:
                        continue

                    ## -Constrain 2, y=1 -> z=0: SKIP VALUES WHEN yt=1, z!=0 (z must be 0)
                    if yt == 1 and zt != 0:
                        continue

                    ## -Constrain 3, x'>Qfree -> y=1: SKIP VALUES WHEN x'>0.5, z=0, yt=0 (yt must be 1)
                    if x_prime > x_threshold and zt == 0 and yt == 0:
                        continue

                    x_prime_idx = np.searchsorted(x, x_prime)
                   # x_prime_idx = min(max(x_prime_idx, 0), len(x) - 1)

                    # Condition y=1    
                    if yt == 1:
                        future_value = beta * V[x_prime_idx, min(z), R_prime_idx, Ra_prime_idx]
                    # Conditions y=0   
                    else:# yt = 0
                        if zt == 0:
                            future_value = beta * V[x_prime_idx, max(z), R_prime_idx, Ra_prime_idx] #z' = 0
                        else:
                            future_value = beta * V[x_prime_idx, (max(z) - zt) + 1, R_prime_idx, Ra_prime_idx] 

                    current_utility = utility(xt, dt, nut, rt, yt, params, P)
                    total_value = current_utility + future_value + et
                    if total_value > best_action_value:
                        best_action_value = total_value
                        best_yt = yt
                        best_dt = dt
                        best_rt = rt
            
        # Apply the best action
        yt = best_yt
        dt = best_dt
        rt = best_rt
        if rt is None:
            continue
        
        d_sim.append(dt)
        y_sim.append(yt)
        r_sim.append(rt)
        ra_sim.append(rat)
        at_sim.append(at)
        nu_sim.append(nut)
        
        # Constraints for data generation
        
        R_prime = min(max_ref, Rt+rt)
        R_sim.append(R_prime)
            
        Ra_prime = min(max_ref, Rat+rat)
        Ra_sim.append(Ra_prime)
        
        x_prime = xt + at - dt
        if x_prime > max_x or x_prime < 0:
            continue
        x_sim.append(max(0, x_prime))  

        if yt == 1: 
            zt = max(z)  # Set to maximum value of z
        else:
            if zt == 0:
                zt = 0
            else:
                zt = zt - 1
        z_sim.append(zt)
    
    # Create a DataFrame for the simulated data
    sim_data = pd.DataFrame({
        'x': x_sim[:-1],
        'a': at_sim, 
        'd': d_sim,
        'y': y_sim,
        'z': z_sim[:-1],
        'nu': nu_sim,
        'R': R_sim[:-1],
        'Ra': Ra_sim[:-1],
        'r': r_sim,
        'ra': ra_sim
    })
    
    sim_data = sim_data.round({'x': 1, 'a': 1, 'd': 1})
    
    return sim_data


## Exponed Values (Matrix of VF = u + V')
def compute_v_matrix(Va, x, z, R, Ra, g_avalues, d, y, r, params, g_nuvalues, g_nuprobs, ra_probs, 
                     beta, P, m_bonus, xbase_threshold, max_ref, g_x_interval, max_x, tol):
     
    max_x = max(x)
    
    zi = np.flip(z)[:-1]
    z_semi = z[:-1]
    
    # Initialize results matrices for y=0, z=!0 and y=0, z=0
    results_matrix03 = np.full((len(d), len(r), len(x), len(zi), len(R), len(Ra)), -np.inf)
    results_matrix01 = np.full((len(d), len(r), len(x), len(R), len(Ra)), -np.inf)
    results_matrix11 = np.full((len(d), len(r), len(x), len(R), len(Ra)), -np.inf)
    results_matrix1 = np.full((len(d), len(r), len(x), len(z), len(R), len(Ra)), -np.inf)

    
    yt = 0  # Given y = 0
           
    # Calculate results_matrix03 (y=0, z=3, z=2, z=1)
    for i, xt in enumerate(x):
        for m, Rt in enumerate(R):
            for n, Rat in enumerate(Ra):
                for k, l in zip(zi, z_semi):
                
                    if Rt-Rat < 0:
                        continue

                    for j, dt in enumerate(d):
                        for s, rt in enumerate(r):
                
                            total_value = 0
                            for rat, rap in enumerate(ra_probs[Rt - Rat][1]):
                                
                                x_threshold = xbase_threshold + (min(max_ref, Rat+rat)*g_x_interval) * m_bonus
                                g_aprobs = a_probs(g_avalues, xt, l, x_threshold, k1=0.1, k2=0.2, b=0.05, p1=2, p2=2, q=2)
                                for a_val, a_prob in zip(g_avalues, g_aprobs):
                                   
                                    x_prime = xt + a_val - dt
                                    x_idx = np.searchsorted(x, x_prime)
                                   
                                    if x_prime <= max_x and x_prime >= 0:
                                        for nu_idx, nu_val in enumerate(g_nuvalues):
                                            try:
                                                v = logsumexp((utility(xt, dt, nu_val, rt, yt, params, P) +
                                                     beta * Va[x_idx, k + 1, min(max_ref, Rt+rt), min(max_ref, Rat+rat)]) * rap * g_nuprobs[nu_idx]) * a_prob
                                            except IndexError:
                                                v = 0
                                            total_value += v
                            if total_value == 0:
                                total_value = -np.inf
                            results_matrix03[j, s, i, k, m, n] = total_value


    # Calculate results_matrix01 (y=0, z=0)
    for i, xt in enumerate(x):
        for m, Rt in enumerate(R):
            for n, Rat in enumerate(Ra):
                
                if Rt-Rat < 0:
                    continue

                for j, dt in enumerate(d):
                    for s, rt in enumerate(r):
            
                        total_value = 0
                        for rat, rap in enumerate(ra_probs[Rt - Rat][1]):
                            
                            x_threshold = xbase_threshold + (min(max_ref, Rat+rat)*g_x_interval) * m_bonus
                            g_aprobs = a_probs(g_avalues, xt, min(z), x_threshold, k1=0.1, k2=0.2, b=0.05, p1=2, p2=2, q=2)
                            for a_val, a_prob in zip(g_avalues, g_aprobs):
                                
                                x_prime = xt + a_val - dt
                                x_idx = np.searchsorted(x, x_prime)
                                
                                if x_prime <= max_x and x_prime >= 0:
                                    if x_prime <= x_threshold:
                                        for nu_idx, nu_val in enumerate(g_nuvalues):
                                            try:
                                                v = logsumexp((utility(xt, dt, nu_val, rt, yt, params, P) +
                                                     beta * Va[x_idx, max(z), min(max_ref, Rt+rt), min(max_ref, Rat+rat)]) * rap * g_nuprobs[nu_idx]) * a_prob
                                            except IndexError:
                                                v = 0
                                            total_value += v
                       
                        if total_value == 0:
                            total_value = -np.inf
                        results_matrix01[j, s, i, m, n] = total_value
                
    results_matrix0 = np.concatenate((results_matrix03, results_matrix01[:, :, :, np.newaxis, :, :]), axis=3)

    ###y = 1
    yt = 1
    for i, xt in enumerate(x):
        for m, Rt in enumerate(R):
            for n, Rat in enumerate(Ra):
                
                if Rt-Rat < 0:
                    continue
                
                for j, dt in enumerate(d):
                    for s, rt in enumerate(r):
            
                        total_value = 0
                        for rat, rap in enumerate(ra_probs[Rt - Rat][1]):
                            
                            x_threshold = xbase_threshold + (min(max_ref, Rat+rat)*g_x_interval) * m_bonus
                            g_aprobs = a_probs(g_avalues, xt, max(z), x_threshold, k1=0.1, k2=0.2, b=0.05, p1=2, p2=2, q=2)
                            for a_val, a_prob in zip(g_avalues, g_aprobs):
                                
                                x_prime = xt + a_val - dt
                                x_idx = np.searchsorted(x, x_prime)
                                
                                if x_prime <= max_x and x_prime >= 0:
                                    for nu_idx, nu_val in enumerate(g_nuvalues):
                                        try:
                                            v = logsumexp((utility(xt, dt, nu_val, rt, yt, params, P) +
                                                 beta * Va[x_idx, min(z), min(max_ref, Rt+rt), min(max_ref, Rat+rat)]) * rap * g_nuprobs[nu_idx]) * a_prob
                                        except IndexError:
                                            v = 0
                                        total_value += v
                            if total_value == 0:
                                total_value = -np.inf
                            results_matrix11[j, s, i, m, n] = total_value

    for j in range(len(d)):
        for s in range(len(r)):
            for i in range(len(x)):
                for m in range(len(R)):
                    for n in range(len(Ra)):
                        results_matrix1[j, s, i, -1, m, n] = results_matrix11[j, s, i, m, n]
    
    combined_array = np.concatenate((results_matrix0, results_matrix1))
    
    combined_dict = {}
    for yt in range(2):  # yt = 0 or 1
        for j, dt in enumerate(d):
            for s, rt in enumerate(r):
                key = (yt, dt, rt)
                if yt == 0:
                    combined_dict[key] = results_matrix0[j, s, :, :, :, :]
                else:
                    combined_dict[key] = results_matrix1[j, s, :, :, :, :]

    return combined_dict


## Model-based Probabilities
def normalize_exponentiated_values(Va, x, z, R, Ra, g_avalues, d, y, r, params, g_nuvalues, g_nuprobs, ra_probs, beta, P, m_bonus, xbase_threshold, max_ref, g_x_interval, max_x, tol):
    normalized_dict = {}
    total_sum = 0
    
    # Compute v_matrix_dict (assuming the function is defined elsewhere)
    v_matrix_dict = compute_v_matrix(Va, x, z, R, Ra, g_avalues, d, y, r, params, g_nuvalues, g_nuprobs, ra_probs, 
                     beta, P, m_bonus, xbase_threshold, max_ref, g_x_interval, max_x, tol)

    # Calculate the sum of all exponentiated values
    for key, matrix in v_matrix_dict.items():
        exp_matrix = np.exp(matrix)
        total_sum += exp_matrix

    # Normalize each matrix by the total sum
    for key, matrix in v_matrix_dict.items():
        exp_matrix = np.exp(matrix)
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized_matrix = np.true_divide(exp_matrix, total_sum)
            # Where total_sum is 0, set the result to 0
            normalized_matrix = np.where(np.isnan(normalized_matrix), 0, normalized_matrix)
        normalized_dict[key] = normalized_matrix

    return normalized_dict


## Log-likelihood
def log_like(df, Va, x, z, R, Ra, g_avalues, d, y, r, params, g_nuvalues, g_nuprobs, ra_probs, beta, P, m_bonus, xbase_threshold, max_ref, g_x_interval, max_x, tol):
    logL = 0

    Pr_dt_yt = normalize_exponentiated_values(Va, x, z, R, Ra, g_avalues, d, y, r, params, g_nuvalues, g_nuprobs, ra_probs, beta, P, m_bonus, xbase_threshold, max_ref, g_x_interval, max_x, tol)
    #print(Pr_dt_yt)
    
    for decisiond, decisionr, decisiony, statex, statez, stateR, stateRa in zip(df['d'], df['r'], df['y'], df['x'], df['z'], df['R'], df['Ra']):
        statex_index = int(np.searchsorted(x, statex))  
        #print(statex_index)
        statez_index = int(max(z) - statez) 
        stateR_index = int(stateR)
        stateRa_index = int(stateRa)
        
        prob_matrix = Pr_dt_yt.get((decisiony, decisiond, decisionr), None)
        #print(prob_matrix)
        
        if prob_matrix is not None and prob_matrix[statex_index, statez_index, stateR_index, stateRa_index] > 0:
            logL += np.log(prob_matrix[statex_index, statez_index, stateR_index, stateRa_index]) # add indexes
    #print('a')
    return -logL


#### Recover

## Optimization
def optimize_parameters(df, Va, x, z, R, Ra, g_avalues, d, y, r, params, g_nuvalues, g_nuprobs, ra_probs, beta, P, m_bonus, xbase_threshold, max_ref, g_x_interval, max_x, tol, bounds, initial_guess):
    def objective(params):
        #print(params)
        return log_like(df, Va, x, z, R, Ra, g_avalues, d, y, r, params, g_nuvalues, g_nuprobs, ra_probs, beta, P, m_bonus, xbase_threshold, max_ref, g_x_interval, max_x, tol)
    options = None
    method = 'Powell'
    result = minimize(objective, initial_guess,  method=method, bounds=bounds, options=options)
    return result.x


## SE
def standard_errors(params, iterations):
    sd = np.std(params, axis=0)
    return sd / np.sqrt(iterations)


## Bootstrap
def params_iteration(user, theta):
    results_list = []
    
    # Constants
    iterations = 100
    N = 1000
    tol = 1e-6
    g_x_interval = 0.1
    xbase_threshold = 0.5
    max_x = 1
    max_ref = 2
    P = 0.4
    beta = 0.1
    m_bonus = 1
    
    # States
    R = np.arange(0, max_ref+1)
    Ra = np.arange(0, max_ref+1)
    x = np.arange(0, max_x+g_x_interval, g_x_interval)
    z = np.arange(3, -1, -1)

    # Decisions
    d = np.arange(0, max_x+g_x_interval, g_x_interval) 
    y = [0,1]
    r = np.arange(0, max_ref+1)

    # Shocks
    g_avalues = np.arange(0, 2.5 * g_x_interval, g_x_interval)
    g_nuvalues = np.array([0.1, 0.2, 0.25])
    g_nuprobs = np.array([1, 1, 1], dtype=float)
    g_nuprobs = g_nuprobs / np.sum(g_nuprobs)

    n_values = np.arange(0, max_ref + 1)
    p = 0.1
    ra_probs = [(n, binomial_probabilities(n, p)) for n in n_values]
    
    # Parameters
    alpha, alpha0, rho1, rho2, gamma = -120, 100, -12, 6, -7

    user_results_file = f"{user}_results_m.csv"
    
    # If a user CSV already exists, load and return it
    if os.path.exists(user_results_file):
        print(f"Results for {user} already exist. Loading existing results.")
        existing_df = pd.read_csv(user_results_file)
        return existing_df

    # Checkpoints
    checkpoint_filename = f"{user}_checkpoint_pow_m.pkl"
    params = [theta, alpha, alpha0, rho1, rho2, gamma]
    initial_guess = params
    bounds = [[0, None], [None, 0], [0, None], [None, 0], [0, None], [None, 0]]
    
    # Set up result storage
    results = np.zeros((iterations, len(params)))
    start_iteration = 0

    # Check if checkpoint file exists (resume if it does)
    if os.path.exists(checkpoint_filename):
        with open(checkpoint_filename, 'rb') as f:
            checkpoint = pickle.load(f)
        results = checkpoint['results']
        start_iteration = checkpoint['iteration']
        print(f"Resuming {user} from iteration {start_iteration}")
    else:
        print(f"No checkpoint found for {user}, starting from iteration 0")
    
    # Run value function iteration
    Va = value_function_iteration2(
        x, z, R, Ra, g_avalues, d, y, r, params, g_nuvalues, g_nuprobs, 
        ra_probs, beta, P, m_bonus, xbase_threshold, max_ref, g_x_interval, max_x, tol
    )
    
    for i in range(start_iteration, iterations):
        # Keep trying the current iteration until it succeeds
        while True:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    df1 = generate_simulation_data(
                        Va, x, z, R, Ra, g_avalues, d, y, r, params, g_nuvalues, g_nuprobs, 
                        ra_probs, p, beta, P, m_bonus, xbase_threshold, max_ref, g_x_interval, max_x, tol, N
                    )
                    optimized_params = optimize_parameters(
                        df1, Va, x, z, R, Ra, g_avalues, d, y, r, params, g_nuvalues, g_nuprobs, ra_probs, 
                        beta, P, m_bonus, xbase_threshold, max_ref, g_x_interval, max_x, tol, bounds, initial_guess
                    )
                results[i, :] = optimized_params
                break
            except RuntimeWarning as e:
                print(f"RuntimeWarning encountered at iteration {i+1} for {user}: {e}. Retrying this iteration...")

        # Save checkpoint after each iteration
        with open(checkpoint_filename, 'wb') as f:
            pickle.dump({'results': results, 'iteration': i+1}, f)
        print(f"Iteration {i+1} for {user} completed and checkpoint saved.")

    # After completing all iterations, remove the checkpoint file
    if os.path.exists(checkpoint_filename):
        os.remove(checkpoint_filename)
        print(f"All iterations completed for {user}. Checkpoint file removed.")

    # Save final CSV for this user
    user_results = pd.DataFrame(
        results,
        columns=["theta", "alpha", "alpha0", "rho1", "rho2", "gamma"]
    )
    user_results["user"] = user
    user_results.to_csv(user_results_file, index=False)
    print(f"Final results for {user} saved to {user_results_file}.")
    
    return user_results
