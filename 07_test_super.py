"""
Test a supervised model to calculate RIS coefficients
Author  : Khin Thandar Kyaw
Reference : DL Framework for Optimization of MISO Downlink Beamforming, TCOM,
            TianLin0509/BF-design-with-DL
Date    : 1 Aug 2024
Last Modified : 
"""

from super_unsuper_utils import *
from nn_utils import *
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.optimizers.schedules import ExponentialDecay


# ------------------------------------
# load and generate simulation data
# ------------------------------------
total_users = total_users()

for total_user in total_users:
    print(f'Total # of Users: {total_user}')
    print_line()
    
    # (sample, user, row, col)
    Nt, N, M, K, Lm, Lk, Ltotal = parameters(total_user)
    batch_size = 32
    
    print(f'Loading training data...')
    
    G_test = np.load(f'test/{total_user}users/G_test.npy')
    Rg_test = np.load(f'test/{total_user}users/Rg_test.npy')
    phi_test = np.load(f'test/{total_user}users/phi_testADMM.npy')
    
    print("Training data loaded.")
    print("====================================")
    print("Preprocessing the data...")
    
    G_stacked = stacking(G_test)
    Rg_stacked = stacking(Rg_test)
    
    print("====================================")
    print("Building the model...")
    
    # ------------------------------------
    # Construct the Supervised Model
    # ------------------------------------
    def build_branch(input_layer):
        x = layers.BatchNormalization()(input_layer)
        if (len(input_layer.shape) == 5):
            x = layers.Reshape((input_layer.shape[1] * input_layer.shape[2], input_layer.shape[3], input_layer.shape[4]))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Flatten()(x)
        return x

    G_stacked_input = layers.Input(name='GStackedInput', shape=(G_stacked.shape[1:]), dtype=tf.float32)
    Rg_stacked_input = layers.Input(name='RgStackedInput', shape=(Rg_stacked.shape[1:]), dtype=tf.float32)
    phi_input = layers.Input(name='PhiInput', shape=(phi_test.shape[1:]), dtype=tf.float32)
    
    G_branch = build_branch(G_stacked_input)
    Rg_branch = build_branch(Rg_stacked_input)
    phi_flat = layers.Flatten()(phi_input)
    
    concatenated = layers.Concatenate()([G_branch, Rg_branch, phi_flat])
    x = layers.BatchNormalization()(concatenated)
    x = layers.Dense(512, activation='relu')(x)
    # x = layers.Dropout(0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    # x = layers.Dropout(0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    # x = layers.Dropout(0.1)(x)
    
    phi = layers.Dense(K * N, activation='linear')(x)
    
    phi_pred = layers.Lambda(reshape_phi,
                             dtype=tf.float32,
                             output_shape=(phi_input.shape[0], K, N, 1))([K, N, phi, phi_input])
    
    loss = layers.Lambda(loss_phi, 
                         dtype=tf.float32, 
                         output_shape=(1),
                         name="loss")([phi_input, phi_pred, N])
    
    model = keras.Model(inputs=[G_stacked_input, Rg_stacked_input, phi_input], outputs=loss)
    
    lr_schedule = ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=10000,
        decay_rate=0.9)
    
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
    
    # Compile the model
    model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: y_pred)
    model.summary()
    
    ensure_dir(f'test/{total_user}users/')
    keras.utils.plot_model(model, to_file=f'test/{total_user}users/supervised_phi_model.png', show_shapes=True, show_layer_names=True, dpi=300)
    
    class SavePhi(keras.callbacks.Callback):
        def __init__(self, save_path):
            super(SavePhi, self).__init__()
            self.save_path = save_path
            
        def on_test_end(self, logs=None):
            phi_model = keras.Model(inputs=[G_stacked_input, Rg_stacked_input, phi_input], outputs=phi_pred)
            phik = phi_model.predict([G_stacked, Rg_stacked, phi_test])
            np.save(self.save_path, phik)
            print(f"Phi saved at {self.save_path}")
            
    # ------------------------------------
    # Load the model
    # ------------------------------------
    model.load_weights(f'train/{total_user}users/phi_trainADMM.h5')
    save_on_eval = SavePhi(f'test/{total_user}users/phi_test_NN.npy')
    model.evaluate(x=[G_stacked, Rg_stacked, phi_test], y = phi_test, # dummy y
                   batch_size=batch_size, 
                   verbose= 0,
                   callbacks=[save_on_eval])
    
    phi_test_NN = np.load(f'test/{total_user}users/phi_test_NN.npy')
    print(f'Phi_test_NN shape: {phi_test_NN.shape}')
    
    print("====================================")
    print("Calculating theta...")
    sample_size = phi_test_NN.shape[0]
    user_size = K
    row_size = N
    theta_test_NN = np.zeros((sample_size, user_size, row_size), dtype=np.complex64)
    big_theta = np.zeros((sample_size, user_size, row_size, row_size), dtype=np.complex64)

    for sample_id in range(sample_size):
        for user_id in range(user_size):
            for row_id in range(row_size):
                phi = phi_test_NN[sample_id, user_id, row_id]
                theta_test_NN[sample_id, user_id, row_id] = (1 / np.sqrt(N)) * np.exp(1j * phi)
            big_theta[sample_id, user_id] = np.diagflat(theta_test_NN[sample_id, user_id])
                
    theta_test_NN = np.expand_dims(theta_test_NN, axis=-1)
    print(f'Theta_test_NN shape: {theta_test_NN.shape}')
                
    np.save(f'test/{total_user}users/theta_test_NN.npy', theta_test_NN)
    print(f'Theta saved at test/{total_user}users/theta_test_NN.npy')
    
    print("====================================")
    print(f'theta_test_NN = {theta_test_NN[0]}')
    
    np.save(f'test/{total_user}users/big_theta.npy', big_theta)
    print(f'big_theta saved at test/{total_user}users/big_theta.npy')
    
    print("====================================")
    print("big_theta shape: ", big_theta.shape)
    print("big_theta = ", big_theta[0])
    
    
    
    
                
    
    
                                                 

    
    
    
    
    
    