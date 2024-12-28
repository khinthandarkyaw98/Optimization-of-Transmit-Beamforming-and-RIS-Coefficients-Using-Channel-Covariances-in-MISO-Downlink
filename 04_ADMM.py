"""
Build a unsupervised model for indiviudal power and individual beta optimization
Author  : Khin Thandar Kyaw
Reference : C. G. Tsinos and B. Ottersten, "An Efficient Algorithm for Unit-
Modulus Quadratic Programs With Application in Beamforming for Wireless Sensor Networks
Last Modified : 28 DEC 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from nn_utils import *

def calculate_theta_ADMM(G_data, Rg_data, user_size): #filename, pic_name):
    # Compute Q
    print(f'G_data shape: {np.array(G_data).shape}')
    for user_size in total_users:
        print(f'Total # of Users: {user_size}')
        print_line()
        
        print('Loading...')
        Nt, N, _, K, _, _, _ = parameters(user_size)
        
        # sample_size = len(G)
        
        theta_all_samples = []
    
        for G_matrix, Rg_matrix in zip(G_data, Rg_data):
            print(f'G_matrix shape: {G_matrix.shape}')
            theta_per_sample = []
            G_user_size = G_matrix.shape[0]
            # print(f'G_matrix shape: {G_matrix.shape}') # (user, row, col)
            for i in range(G_user_size):

                G = G_matrix[i]
                Rg = Rg_matrix[i]
            
                Q = (Rg.T) * (G.conj().T @ G)
                # print(f'Rg shape: {Rg.shape}')
                # print(f'G shape: {G.shape}')
                # print(f'Q shape: {Q.shape}')

                # Iteration parameters
                mu = 2000 
                epp = 1e-5  # tolerance
                epz = 1e-5  # tolerance
                k_max = 1000  # max. no. of loop iterations

                # Tracking the diffs
                diffz = np.zeros(k_max)
                diffzw = np.zeros(k_max)
                diffLamb = np.zeros(k_max)
                progress_obj = np.zeros(k_max)

                # Initialize
                B = np.linalg.inv(Q - mu * np.eye(N))
                z = np.random.randn(N, 1) + 1j * np.random.randn(N, 1)
                z = z / np.linalg.norm(z)
                Lamb = np.zeros((N, 1), dtype=complex)
                w = np.exp(1j * 2 * np.pi * np.random.rand(N, 1)) / N
                zprev = np.random.randn(N, 1) + 1j * np.random.randn(N, 1)
                zprev = zprev / np.linalg.norm(zprev)

                k = 0

                while not ((np.linalg.norm(z - w) <= epp) and (np.linalg.norm(z - zprev) <= epz)) and (k < k_max):
                    k += 1
                    zprev = z
                    Lamb_prev = Lamb

                    z = B @ (Lamb - mu * w)
                    
                    z_H = z.conj().T
                    
                    obj = - 0.5 * z_H @ Q @ z
                    # print(f'obj = {obj}')
                    
                    progress_obj[k-1] = obj.real
                    
                    a = z + Lamb_prev / mu
                    
                    # print(f' a = {a}')
                    
                    for n in range(N):
                        if a[n] != 0:
                            # print(f' w[n] before = {w[n]}')
                            w[n] = a[n] / np.abs(a[n])
                            # print(f' w[n] after = {w[n]}')
                            # print(f'Norm of w[n] = {np.linalg.norm(w[n])}')
                        else:
                            w[n] = 0

                    # break
                    Lamb = Lamb + mu * (z - w)

                    diffz[k-1] = np.linalg.norm(z - zprev)
                    diffzw[k-1] = np.linalg.norm(z - w)
                    diffLamb[k-1] = np.linalg.norm(Lamb - Lamb_prev)

                Theta = w / np.sqrt(N)
                theta_per_sample.append(Theta)
                print(f'Shape of theta_per_sample: {np.array(theta_per_sample).shape}')
            theta_all_samples.append(theta_per_sample)
            print(f'Shape of theta_all_samples: {np.array(theta_all_samples).shape}')
        return theta_all_samples # in this case we return batches
        
    
total_users = total_users()
###########################################################
# G_train shape: (8500, 3, 16, 30)
# G_test shape: (1500, 3, 16, 30)

for total_user in total_users:
    
    G_train = np.load(f'train/{total_user}users/G_train.npy')
    Rg_train = np.load(f'train/{total_user}users/Rg_train.npy')
    
    # loop through 100 samples around the dataset till finish
    train_shape = G_train.shape[0]
    
    theta_train_samples = []
    
    print(f'Calculating theta [TRAINING] for {total_user} users...')
    
    for batch in range(0, train_shape, 100):
        print(f'Training Batch : {batch}')
        G_train_batch = G_train[batch:batch+100]
        Rg_train_batch = Rg_train[batch:batch+100]
        filename_train = f'train/{total_user}users/theta_trainADMM.npy'
        pic_train_name = f'theta_trainADMM{batch}.png'
        
        
        theta_train_batches = calculate_theta_ADMM(G_train_batch, Rg_train_batch, total_user)
        print('Shape of theta_train_batches: ', np.array(theta_train_batches).shape)
        theta_train_samples.extend(theta_train_batches)
    np.save(f'train/{total_user}users/theta_trainADMM.npy', theta_train_samples)
        
     
    G_test = np.load(f'test/{total_user}users/G_test.npy')
    Rg_test = np.load(f'test/{total_user}users/Rg_test.npy')   

    test_shape = G_test.shape[0]
    
    theta_test_samples = []
    
    print(f'Calculating theta [TESTING] for {total_user} users...')
    
    for batch in range(0, test_shape, 100):
        print(f'Testing Batch : {batch}')
        G_test_batch = G_test[batch:batch+100]
        Rg_test_batch = Rg_test[batch:batch+100]
        filename_test = f'test/{total_user}users/theta_testADMM.npy'
        pic_test_name = f'theta_testADMM{batch}.png'
        
        theta_test_batches = calculate_theta_ADMM(G_test_batch, Rg_test_batch, total_user)
        theta_test_samples.extend(theta_test_batches)
    np.save(f'test/{total_user}users/theta_testADMM.npy', theta_test_samples)
    print("====================================")
    print("Theta saved!")