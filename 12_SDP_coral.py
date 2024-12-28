"""
Semi-definite Programming (SDP) for the optimal solution
Author    : Khin Thandar Kyaw
Last Modified: 28 Dec 2024
"""


import cvxpy as cp
import numpy as np
import logging
from nn_utils import *
from timer import *

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("Plotting/optimization_log_copy_coral_2.log", mode="w"),  # Log to a file
        logging.StreamHandler()  # Also print to console
    ]
)

time_list = []

with Timer() as timer:
    total_users = total_users()[0]
    logging.info(f"Total number of users = {total_users}")
    
    SNR = [-10, -5, 0, 5, 10, 15, 20, 25] # threshold
    Gamma = [10 ** (np.array(gamma_db) / 10) for gamma_db in SNR]
    size_gamma = len(Gamma)
    logging.info(f"Gamma in linear scale = {Gamma}")
    
    # covariance after ADMM
    # Rh = np.load(f'test/{total_users}users/cov_test_ADMM.npy')[5:7, :, :, :]
    Rh = np.load(f'test/{total_users}users/cov_test_ADMM.npy')[:100, :, :, :]
    # Rh_1 = np.load(f'test/{total_users}users/cov_test_ADMM.npy')[1:2, :, :, :]
    # Rh_2 = np.load(f'test/{total_users}users/cov_test_ADMM.npy')[4:5, :, :, :]
    
    # Rh = np.concatenate((Rh_1, Rh_2), axis=0)
    logging.info(f'Shape of Rh = {Rh.shape}')
    Nt, N, _, _, _, _, _ = parameters(total_users)
    
    sample_size = Rh.shape[0]
    
    # all direct and indirect users
    sum_rate = []
    transmit_power = []
    for gamma_idx, gamma in enumerate(Gamma):
        logging.info("=" * 70)
        logging.info(f"At SNR threshold = {SNR[gamma_idx]} dB")
        
        sum_rate_per_gamma = []
        transmit_power_per_gamma = []
        for sample in range(sample_size):
            Rh_sample = Rh[sample]
            gamma_sample = np.full((total_users), gamma) 
            # https://www.cvxpy.org/tutorial/constraints/index.html
            # W matrix
            W_sample = [cp.Variable((Nt, Nt), hermitian=True) for _ in range(total_users)]
            objective = cp.Minimize(cp.sum([cp.trace(W_k) for W_k in W_sample]))
            constraints = []
            
            for k in range(total_users):
                signal = cp.trace(Rh_sample[k] @ W_sample[k])
                interference = cp.sum([cp.trace(Rh_sample[k] @ W_sample[n]) for n in range(total_users) if n!=k])
                # no complex number in the constraints of cvxpy
                constraints.append(cp.real(signal - gamma_sample[k] * interference) >= gamma_sample[k])
                # positive semi-definite constraint
                constraints.append(W_sample[k] >> 0)
            # https://www.cvxpy.org/examples/basic/sdp.html
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            # check if the problem is solved
            problem_status = problem.status
            # logging.info(f"Problem status: {problem_status}")
            
            test_constraint_result = []
            for k in range(total_users):
                signal = np.trace(Rh_sample[k] @ W_sample[k].value)
                interference = np.sum(np.trace(Rh_sample[k] @ W_sample[n].value) for n in range(total_users) if n!=k)
                test_constraint_result.append(np.real(signal - gamma_sample[k] * interference))
                # logging.info(W_sample[k].value)
            # logging.info(f"Test constraints: {test_constraint_result} must be greater than or equal to {gamma_sample}")
            
            # only for optimal solutions by the solver
            if problem_status != cp.OPTIMAL:
                logging.info(f"No optimal solution for Sample {sample + 1}")
                logging.info("-" * 50)
                continue
            
            else:
                # check the eigenvalues of W
                logging.info(f"Optimal solution for Sample {sample + 1}")
                opt_w_sample = []
                mu_sample = []
                pos_eig_sample = []
                sorted_eigenvalues_sample = []
                sorted_eigenvectors_sample = []
                zeros_eig_sample = []
                for k in range(total_users):
                    logging.info(f"Sample {sample + 1} - User {k + 1}")
                    eigenvalues, eigenvectors = np.linalg.eigh(W_sample[k].value)
                    sorted_indices = np.argsort(eigenvalues)[::-1]
                    sorted_eigenvalues = eigenvalues[sorted_indices]
                    sorted_eigenvectors = eigenvectors[:, sorted_indices]
                    sorted_eigenvalues_sample.append(sorted_eigenvalues)
                    sorted_eigenvectors_sample.append(sorted_eigenvectors)
                    mu_sample.append(sorted_eigenvectors[:, 0].reshape(-1, 1))
                    # logging.info(f"Eigenvalues: {eigenvalues}")
                    # logging.info(f"Sorted Eigenvalues: {sorted_eigenvalues}")
                    # logging.info(f"Sorted Eigenvectors: {sorted_eigenvectors}")
                    
                    # one positive eigenvalue and the rest are zeros
                    # greater than or equal to 1e-6 is considered as positive
                    pos_eig = [eig for eig in eigenvalues if eig >= 1e-6]
                    zeros_eig = [eig for eig in eigenvalues if eig < 1e-6]
                    pos_eig_count = len(pos_eig)
                    zeros_eig_count = len(zeros_eig)
                    pos_eig_sample.append(pos_eig_count)
                    zeros_eig_sample.append(zeros_eig_count)
                    # logging.info(f"Positive eigenvalues: {pos_eig}")    
                    # logging.info(f"Zero eigenvalues: {zeros_eig}")
                    logging.info(f"Number of positive eigenvalues: {pos_eig_count}")
                    logging.info(f"Number of zero eigenvalues: {zeros_eig_count}")
                logging.info("-" * 50)
                logging.info(f"Positve eigenvalue counts: {pos_eig_sample}")
                logging.info(f"Zero eigenvalue counts: {zeros_eig_sample}")
                
                F = np.zeros((total_users, total_users), dtype=complex)
                for l in range(total_users):
                    for i in range(total_users):
                        if l == i:
                            F[l, i] = mu_sample[l].conj().T @ Rh_sample[l] @ mu_sample[l]
                        else:
                            F[l, i] = - gamma_sample[i] * mu_sample[i].conj().T @ Rh_sample[l] @ mu_sample[i]   
                # optimal_power
                p_star = np.linalg.inv(F) @ gamma_sample.reshape(-1, 1)
                    
    
                # calculate the optimal beamforming vectors
                ############################################
                # the first approach
                ############################################
                # -------------------------------------------
                for k in range(total_users):
                    if pos_eig_sample[k] == 1 and zeros_eig_sample[k] == Nt - 1:
                        logging.info("Switching to the first approach")
                        logging.info(f"Using the first approach for user {k+1} (rank one matrix)")
                        opt_w_sample.append(np.sqrt(sorted_eigenvalues_sample[k][0]) * sorted_eigenvectors_sample[k][:, 0].reshape(-1, 1))
                        logging.info(f"Length of opt_w_sample after having finished the first approach = {len(opt_w_sample)}")
                    else:
                        logging.info("Switching to the second approach")
                        logging.info(f"Using the second approach for user {k+1} (rank greater than one matrix)")
                        # logging.info(f"F = {F}")
                        opt_w_sample.append(np.sqrt(p_star[k]) * mu_sample[k])
                        logging.info(f"Length of opt_w_sample after having finished the second approach = {len(opt_w_sample)}")
                        logging.info(f"Final length of opt_w_sample {len(opt_w_sample)} should be equal to {total_users}")
                ############################################
                # Compute the sum transmit power
                ############################################
                transmit_power_sample = np.sum([np.linalg.norm(opt_w_sample[k]) ** 2 for k in range(total_users)])
                transmit_power_sample_db = 10 * np.log10(transmit_power_sample)
                transmit_power_per_gamma.append(transmit_power_sample_db)
                # logging.info(f"Transmit power = {transmit_power_sample_db}")
                #- -------------------------------------------
                
                ############################################
                # Compute the sum rate
                ############################################
                # -------------------------------------------
                # one tab forward (so you need to take one tab back)
                logging.info(f"Optimal solution for sample {sample + 1}")
                logging.info(f"The number of optimal beams in each sample = {len(opt_w_sample)} at Sample = {sample + 1} - SNR = {SNR[gamma_idx]} dB")
                sinr_sample = []
                for k in range(total_users):
                    signal = opt_w_sample[k].conj().T @ Rh_sample[k] @ opt_w_sample[k]
                    interference = np.sum([opt_w_sample[i].conj().T @ Rh_sample[k] @ opt_w_sample[i] for i in range(total_users) if i!=k]) + 1
                    sinr_sample.append(np.real(signal / interference))
                    
                sum_rate_sample = np.sum([np.log2(1 + sinr_k) for sinr_k in sinr_sample])
                sum_rate_per_gamma.append(sum_rate_sample)
                 #- -------------------------------------------
                    # logging.info(f"Sum rate = {sum_rate_sample}")
                logging.info("-" * 50)
            
        ############################################
        # averaging over multiple samples
        ############################################
        sum_rate.append(np.mean(sum_rate_per_gamma))
        transmit_power.append(np.mean(transmit_power_per_gamma))
        
    logging.info(f"Sum rate = {sum_rate}")
    np.save(f'Plotting/{total_users}users/sum_rate_optimal_solution_coral_2_plot_7.npy', np.array(sum_rate))
    logging.info(f"Transmit power = {transmit_power}")
    np.save(f'Plotting/{total_users}users/transmit_power_optimal_solution_coral_2_plot_7.npy', np.array(transmit_power))
logging.info(f"Time elapsed = {timer.elapsed_time}")
np.save(f'test/{total_users}users/timeArrayOptimalSolution_coral_2_plot_7.npy', np.array(timer.elapsed_time))
                
                
                  
                                
                        
                        