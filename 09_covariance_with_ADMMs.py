"""
Generate covariance 
Author    : Khin Thandar Kyaw
Date      : 31 AUG 2023
Last Modified  : 12 JAN 2024
"""

import time
from nn_utils import *
from covariance_utils import *
from timer import *
  
########################################################################
print('Generating covariance matrix...')
print('Loading...')


total_users = total_users()

for total_user in total_users:
  GSample = []
  RgSample = []
  covariance_sample = []
  emax_sample = []
  beam_sample = []
  abs_ZFBF_sample = []
  total_sample_size = 50000
  train_size = int(0.85 * total_sample_size)
  time_list = []
  sample_size = total_sample_size - train_size
  G_test = np.load(f'test/{total_user}users/G_test.npy')
  big_theta_test_NN = np.load(f'test/{total_user}users/big_theta.npy')
  Rg_test = np.load(f'test/{total_user}users/Rg_test.npy')
  print(f'Total # of Users: {total_user}')
  print_line()
  for sample in range(sample_size):
    if sample % 1000 == 0:
      print(f'Generating {sample}th sample...')
      print_line()
    # --------------------------------------------
    # new parameters for M and K for each sample
    # -------------------------------------------
    Nt, N, M, K, Lm, Lk, Ltotal = parameters(total_user)
    covariance_class = Covariance(Nt, N, total_user, M, K, Lm, Lk, Ltotal)
    
    # ------------------------------------
    # direct user channel
    # ------------------------------------
    theta = covariance_class.generate_theta()
    steering_vectors = covariance_class.generate_steering_vectors(theta)
    channel_covariance = covariance_class.generate_channel_covariance(steering_vectors)
    
    # ------------------------------------
    # IRS-assisted user channel
    # ------------------------------------
    channel_BS_IRS = G_test[sample]
    big_theta = big_theta_test_NN[sample]
    channel_covariance_irs = Rg_test[sample]

    # ------------------------------------
    # composite channel covariance
    # ------------------------------------
    channel_covariance_all = covariance_class.generate_composite_channel_covariance(channel_BS_IRS, big_theta, channel_covariance_irs, channel_covariance)
    #print(f'Rank of channel_covariance_irs_compostite: {np.linalg.matrix_rank(channel_covariance_irs_compostite)}')
    covariance_sample.append(channel_covariance_all)
    
    # save eigenvectors corresponding to the largest eigenvalues
    e_max = covariance_class.e_max(channel_covariance_all)
    emax_sample.append(e_max)
    
    # save for zero-forcing
    # ------------------------------------
    # Count the time for ZFBF
    # ------------------------------------
    if sample >= train_size:
      with Timer() as timer:
        U_tilde, W = perform_calculations(covariance_class, channel_covariance_all)
        beam_sample.append(W)
      time_list.append(timer.elapsed_time)
    # ------------------------------------
    else: 
      U_tilde, W = perform_calculations(covariance_class, channel_covariance_all)
      beam_sample.append(W)
      
    abs_ZFBF_res = covariance_class.check_ZFBF_condition(U_tilde, W)
    abs_ZFBF_sample.append(abs_ZFBF_res)
      
  covariance_sample= np.array(covariance_sample)
  print(f'channel_covariance.shape: {covariance_sample.shape}')

  emax_sample = np.array(emax_sample)
  print(f'eMaxSample.shape: {emax_sample.shape}')

  beam_sample = np.array(beam_sample)
  print(f'beamSample.shape: {beam_sample.shape}')

  abs_ZFBF_sample = np.array(abs_ZFBF_sample)
  #print(f'absZFBFSample.shape: {absZFBFSample.shape}')
  #print(f'absZFBFSample[0]: {absZFBFSample[0]}')

  print('Saving...')
          
  # Before saving, ensure the dircetory exists
  ensure_dir(f'train/{total_user}users/')
  ensure_dir(f'test/{total_user}users/')
  # Now, save the data.
  np.save(f'test/{total_user}users/cov_test_ADMM.npy', covariance_sample)
  np.save(f'test/{total_user}users/eMax_test_ADMM.npy', emax_sample)
  np.save(f'test/{total_user}users/beamZF_ADMM.npy', beam_sample) # ZF beamforming
  np.save(f'test/{total_user}users/timeArrayZWF_ADMM.npy', time_list) # W Time
  np.save(f'train/{total_user}users/absZFBFSample_ADMM.npy', abs_ZFBF_sample) # ZF condition
  print(f'Data saved successfully!')
  print_line()
  
  # ------------------------------------
  # Calculate the sum rate of Zero-Forcing
  # ------------------------------------
  print("Calculating the sum rate of Zero-Forcing...")
  print("Loading...")
  
  rateZ = []
  _, _, _, _, NoiseVarTotal, _  = data_preparation(covariance_sample)
  for snr in range(-5, 25, 5):
    SNR = np.power(10, np.ones([covariance_sample.shape[0], 1]) * snr / 10)
    Power = SNR * NoiseVarTotal
    
    # sum rate formulat for wZF is different in noise part
    # K / P_total
    scaledFactor = np.squeeze(np.sqrt(Power/ total_user))
    sumRateZ = np.mean(compute_sum_rate(beam_sample, covariance_sample, scaledFactor))
    rateZ.append(sumRateZ)
    
  ensure_dir(f'Plotting/{total_user}users/')
  np.save(f'Plotting/{total_user}users/sumRateZF_ADMM.npy', np.array(rateZ))
  print(f'Saved sumRateZ successfully for {total_user} users!')
  print_line()