import numpy as np
from nn_utils import *
from scipy.linalg import eigh
from matplotlib import pyplot as plt
from timer import *

# parameters

def calculate_optimal_beam_sum_rate(filename, type, total_users):
    with Timer() as timer:
        SNR = [-10, -5, 0, 5, 10, 15, 20, 25]

        Nt, N, _, _, _, _, _ = parameters(total_users[0])
            
        for K in total_users:
            
            Rh = np.load(f'test/{K}users/{filename}')

            print(f'Shape of Rh = {Rh.shape}')


            sample_size = Rh.shape[0]
            
            optimal_beam_all_snr = []
            # p_star_all_snr = []
            snr_from_p_star_all_rate_threshold = []
            norm_sq_opt_beam_all_snr = []
            x_coordinate = []
            y_coordinate_my_way = []
            y_coordinate = []
            for gamma_db in SNR:
                print('=' * 50)
                print(f'SNR = {gamma_db} dB')
                gamma = 10 ** (np.array(gamma_db) / 10)
                optimal_beam_per_snr = []
                snr_from_p_star_per_rate_threshold = []
                norm_sq_opt_beam_per_snr = []
                x_coordinate_all_sample = []
                y_coordinate_all_sample = []
                y_coordinate_my_way_all_sample = []
                for sample in range(sample_size): # 100 samples
                    if sample % 10 == 0:
                        print("Sample = ", sample)
                        print('Loading...')
                    n_max = 1000
                    diff_p = np.zeros(n_max)
                    epsilon = 1e-17
                    # epsilon = 1e-4
                    
                    p_prev = np.zeros(K)
                    p = np.ones(K)
                    Gamma = np.full((K), gamma)
                    
                    n = 1
                    
                    while (np.linalg.norm(p - p_prev) > epsilon) and (n < n_max):
                        diff_p[n - 1] = np.linalg.norm(p - p_prev)
                        mu_per_sample = []
                        lambdas_per_sample = []
                        p_prev = p.copy()
                        for k in range(K):
                            A = p[k] * Rh[sample, k, :, :]
                            B = sum(p[i] * Gamma[i] * Rh[sample, i, :, :] for i in range(K) if i != k) + np.eye(Nt)
                            eigenValues, eigenVectors = eigh(A, B)
                            sorted_indices = np.argsort(eigenValues)[::-1]
                            sorted_eigenValues = eigenValues[sorted_indices]
                            sorted_eigenVectors = eigenVectors[:, sorted_indices]
                            lambda_max = sorted_eigenValues[0]
                            mu_k = sorted_eigenVectors[:, 0].reshape(-1, 1)
                            lambdas_per_sample.append(lambda_max)
                            mu_per_sample.append(mu_k)
                            
                        n += 1

                        for l in range(K):
                            p[l] = (Gamma[l] / lambdas_per_sample[l]) * p_prev[l]

                    F = np.zeros((K, K), dtype=complex)
                    
                    for l in range(K):
                        for i in range(K):
                            if l == i:
                                F[l, i] = mu_per_sample[l].conj().T @ Rh[sample, l, :, :] @ mu_per_sample[l]
                            else:
                                F[l, i] = - Gamma[i] * (mu_per_sample[i].conj().T @ Rh[sample, l, :, :] @ mu_per_sample[i])
                                
                    Gamma = Gamma.reshape(-1, 1) # (user, 1)
                    p_star_per_sample = np.linalg.inv(F) @ Gamma # (user, 1)
                    # print(f'p_star_per_sample = {p_star_per_sample}')
                    snr_from_p_star_per_rate_threshold = list(snr_from_p_star_all_rate_threshold)
                    # print(f'snr_from_p_star_per_rate_threshold = {snr_from_p_star_per_rate_threshold}')
                    snr_from_p_star_per_rate_threshold.append(sum(np.array(np.real(p_star_per_sample))))
                    # print(f'snr_from_p_star_per_rate_threshold = {snr_from_p_star_per_rate_threshold}')
                    w_per_sample = []
                    norm_sq_opt_beam_per_sample = []
                    expected_sinr_per_sample = np.zeros(K)
                    for k in range(K):
                        w = np.sqrt(p_star_per_sample[k]) * mu_per_sample[k]
                        w_per_sample.append(w)
                        norm_sq_opt_beam_per_sample.append(np.linalg.norm(w)**2)
                        
                    #---- Calculate SINR from p_star ----#
                    for k in range(K):
                        expected_sinr_per_sample[k] = np.real(w_per_sample[k].conj().T @ Rh[sample, k, :, :] @ w_per_sample[k]) / \
                                                        (sum(np.real(w_per_sample[i].conj().T @ Rh[sample, k, :, :] @ w_per_sample[i]) for i in range(K) if i != k) + 1)
                    # print(f'expected_sinr_per_sample = {expected_sinr_per_sample}')
                        
                    sum_norm_sq_opt_beam_per_sample = sum(norm_sq_opt_beam_per_sample)
                    optimal_beam_per_snr.append(np.array(w_per_sample))
                    norm_sq_opt_beam_per_snr.append(sum_norm_sq_opt_beam_per_sample)
                    
                    # print(f'optimal beam per sample = {np.array(w_per_sample).shape}')
                    
                    # print(f'snr_from_p_star_per_rate_threshold = {snr_from_p_star_per_rate_threshold}')
                    snr_from_p_star_per_rate_threshold = 10 * np.log10(snr_from_p_star_per_rate_threshold)
                    # print(f'snr_from_p_star_per_rate_threshold: x_coordinate = {snr_from_p_star_per_rate_threshold}')
                    print(f'SNR (db) for sample {sample}= {snr_from_p_star_per_rate_threshold}')
                    x_coordinate_all_sample.append(snr_from_p_star_per_rate_threshold)
                    sum_rate_y_coordinate = K * np.log2(1 + 10 ** (0.1 * gamma_db))
                    # print(f'y_coordinate = {sum_rate_y_coordinate}')
                    y_coordinate_all_sample.append(sum_rate_y_coordinate)
                
                    # print('-' * 50)
                    # print('My way of y_coordinate')
                    sum_rate_y_coordinate_my_way = np.sum(np.log2(1 + np.array(expected_sinr_per_sample)))
                    # print(f'y_coordinate_my_way = {sum_rate_y_coordinate_my_way}')
                    y_coordinate_my_way_all_sample.append(sum_rate_y_coordinate_my_way)
                    # print('-' * 50)
                
                x_coordinate.append(np.mean(np.array(x_coordinate_all_sample), axis=0))
                y_coordinate.append(np.mean(np.array(y_coordinate_all_sample), axis=0))
                y_coordinate_my_way.append(np.mean(np.array(y_coordinate_my_way_all_sample), axis=0))
                
                # p_star_all_snr.append(np.array(p_star_per_snr))
                optimal_beam_all_snr.append(np.array(optimal_beam_per_snr))
                norm_sq_opt_beam_all_snr.append(np.mean(np.array(norm_sq_opt_beam_per_snr)))
                # snr_from_p_star_all_rate_threshold.append(snr_from_p_star_per_rate_threshold)
                print(f'Optimal beam per SNR = {np.array(optimal_beam_per_snr).shape}')
                print(f'Done for {gamma_db}dB')
            
            np.save(f'test/{K}users/optimal_beam_{type}.npy', np.array(optimal_beam_all_snr))
            np.save(f'Plotting/{K}users/snr_from_p_star_x_coordinate_{type}.npy', np.array(x_coordinate).flatten())
            np.save(f'Plotting/{K}users/snr_from_p_star_y_coordinate_{type}.npy', np.array(y_coordinate).flatten())
            np.save(f'Plotting/{K}users/snr_from_p_star_y_coordinate_my_way_{type}.npy', np.array(y_coordinate_my_way).flatten())
            
            print(f'SNR from p_star y_coordinate = {np.array(y_coordinate).flatten()}')
            print(f'SNR from p_star y_coordinate my way = {np.array(y_coordinate_my_way).flatten()}')
            
            print(f'diff_p = {diff_p}')

            # Plotting
            plt.semilogy(np.arange(1, n_max+1), diff_p, 'b-', label='||p(n) - p(n-1)||', linewidth=2)
            plt.xlabel('Iteration')
            plt.ylabel('Norm difference')
            plt.legend()
            plt.grid(True)
            plt.axis([0, 100, epsilon, 10])
            plt.title(r'$N_t$ = %d, $N$ = %d, $\gamma$ = %.4g, $\epsilon_p$ = %.4g' % (Nt, N, gamma, epsilon))
            plt.show()
            plt.close()
            print('=' * 50)
            
    eplased_time = timer.elapsed_time
    np.save(f'test/{total_users[0]}users/timeArray_opt_BF{type}.npy', np.array(eplased_time))
            

total_users = total_users()

# calculate_optimal_beam_sum_rate('cov_test.npy', 'random_theta', total_users)
calculate_optimal_beam_sum_rate('cov_test_ADMM.npy', 'ADMM_theta_NN', total_users)