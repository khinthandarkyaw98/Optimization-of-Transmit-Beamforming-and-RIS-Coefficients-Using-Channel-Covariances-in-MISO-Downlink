"""
Compare Theta 
Author    : Khin Thandar Kyaw
Date      : 21 Jul 2024
"""

import numpy as np
import math
from nn_utils import *

total_users = total_users()

def generate_random_theta(size):
    phi = np.random.uniform(0, 2 * np.pi, (size, K, N, 1))
    theta = np.cos(phi) + 1j * np.sin(phi)
    normalized_theta = theta / math.sqrt(N)
    return normalized_theta

def compare_theta(Rg_data, G_data, theta_data):
    obj_all_samples = []
    for Rg, G, theta in zip(Rg_data, G_data, theta_data):
        obj_per_sample = []
        user_size = Rg.shape[0]
        for i in range(user_size):
            Rg_i = Rg[i]
            G_i = G[i]
            theta_i = theta[i]
            
            # compute the objective function
            Rg_i_transpose = np.transpose(Rg_i)
            G_i_H = np.transpose(G_i.conj())
            G_i_H_G = np.matmul(G_i_H, G_i)
            # element-wise multiplication
            Rg_i_transpose_G_i_H_G = np.multiply(Rg_i_transpose, G_i_H_G)
            
            theta_i_H = np.transpose(theta_i.conj())
            
            obj = np.matmul(np.matmul(theta_i_H, Rg_i_transpose_G_i_H_G), theta_i)
            obj_per_sample.append(obj)
        obj_all_samples.append(obj_per_sample)
        
    avg_obj = np.mean(obj_all_samples)
    return avg_obj

for total_user in total_users:
    
    Rg_train = np.load(f'train/{total_user}users/Rg_train.npy')[0:3]
    G_train = np.load(f'train/{total_user}users/G_train.npy')[0:3]
    
    Rg_test = np.load(f'test/{total_user}users/Rg_test.npy')[0:3]
    G_test = np.load(f'test/{total_user}users/G_test.npy')[0:3]
    
    # random theta
    Nt, N, M, K, Lm, Lk, Ltotal = parameters(total_user)
    print(f"Total No. of Users: {total_user}")
    print(f'K = {K}')
    
    theta_random = generate_random_theta(3)
    print(f'Theta_random shape: {theta_random.shape}')
    print(f'Theta_random: {theta_random[0]}')
    
    # ADMM case
    theta_train_ADMM = np.load(f'train/{total_user}users/theta_trainADMM.npy')[0:3]
    theta_test_ADMM = np.load(f'test/{total_user}users/theta_testADMM.npy')[0:3]
    
    print(f'Theta_train_ADMM shape: {theta_train_ADMM.shape}')
    print(f'Theta_test_ADMM shape: {theta_test_ADMM.shape}')
    print(f'Theta_train_ADMM: {theta_train_ADMM[0]}')
    
    # Supervised NN
    theta_NN = np.load(f'test/{total_user}users/theta_test_NN.npy')[0:3]
    print(f'Theta_NN shape: {theta_NN.shape}')
    print(f'Theta_NN: {theta_NN[0]}')
    
    ############################################################
    # compare the objective function
    
    # Random theta
    obj_train_random = compare_theta(Rg_train, G_train, theta_random)
    
    # ADMM case
    obj_train_ADMM = compare_theta(Rg_train, G_train, theta_train_ADMM)
    obj_test_ADMM = compare_theta(Rg_test, G_test, theta_test_ADMM)
    
    # Supervised NN
    obj_test_NN = compare_theta(Rg_test, G_test, theta_NN)
    
    
    print(f"User size: {total_user}")
    print("----------------------")
    print("Random theta")
    print(f"Train: {obj_train_random}")
    print("ADMM case")
    print(f"Train: {obj_train_ADMM}")
    print(f"Test: {obj_test_ADMM}")
    print("----------------------")   
    print("Supervised NN")
    print(f"Test: {obj_test_NN}") 
    print("----------------------")

    