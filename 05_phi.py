import numpy as np
from nn_utils import *

total_users = total_users()

# user size means total users *****
# user in this case is the number of RIS users which is just 3
# total users = 6 users 

def compute_phi(theta, phi_filename):
    theta_shape = theta.shape
    
    phi = np.ones((theta_shape))
    # print(f'phi shape = {phi.shape}')
    
    sample_size = phi.shape[0]
    # print(f'sample size = {sample_size}')
    row_size = phi.shape[2]
    # print(f'row size = {row_size}')
    
    RIS_user_size = theta.shape[1]
    
    for sample_id in range(sample_size):
        # print(f'sample_id = {sample_id}')
        for RIS_user_id in range(RIS_user_size):
            # print(f'sample_id, user_id = {sample_id, RIS_user_id}')
            for row_id in range(row_size):
                # print(f'sample_id, user_id, row_id = {sample_id, RIS_user_id, row_id}')
                
                # print(f'theta = {theta[sample_id, RIS_user_id, row_id]}')
                
                real_part_theta = theta[sample_id, RIS_user_id, row_id].real
                imag_part_theta = theta[sample_id, RIS_user_id, row_id].imag
                
                # print(f'real_part_theta = {real_part_theta}')
                # print(f'imag_part_theta = {imag_part_theta}')
                # print(f'type of real_part_theta = {type(real_part_theta)}')
                
                if real_part_theta > 0:
                    phi[sample_id, RIS_user_id, row_id] = np.arctan(imag_part_theta / real_part_theta)
                elif real_part_theta < 0 and imag_part_theta >= 0:
                    phi[sample_id, RIS_user_id, row_id] = np.arctan(imag_part_theta /real_part_theta) + np.pi
                elif real_part_theta < 0 and imag_part_theta < 0:
                    phi[sample_id, RIS_user_id, row_id] = np.arctan(imag_part_theta / real_part_theta) - np.pi
                elif real_part_theta == 0 and imag_part_theta > 0:
                    phi[sample_id, RIS_user_id, row_id] = np.pi/2
                elif real_part_theta == 0 and imag_part_theta < 0:
                    phi[sample_id, RIS_user_id, row_id] = -np.pi/2
                
                # print(f'phi = {phi[sample_id, user_id, row_id]}')

    print(f'Phi shape: {phi.shape}')
    np.save(phi_filename, phi)
    print(f'Phi saved to {phi_filename}')
        
for total_user in total_users:    
    theta_train = np.load(f'train/{total_user}users/theta_trainADMM.npy')
    
    phi_train_filename = f'train/{total_user}users/phi_trainADMM.npy'
    
    print(f'Theta_train shape: {theta_train.shape}')
    compute_phi(theta_train, phi_train_filename)
    
    theta_test = np.load(f'test/{total_user}users/theta_testADMM.npy')
    
    print(f'Theta_test shape: {theta_test.shape}')
    
    phi_test_filename = f'test/{total_user}users/phi_testADMM.npy'
    compute_phi(theta_test, phi_test_filename)
    
    
                
                
        