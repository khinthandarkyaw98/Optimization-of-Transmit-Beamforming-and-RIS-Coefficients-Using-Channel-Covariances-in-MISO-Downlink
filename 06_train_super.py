"""
Build a supervised model to calculate RIS coefficients
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
# Load and generate simulation data
# ------------------------------------
total_users = total_users()

for total_user in total_users:
    print(f'Total # of Users: {total_user}')
    print_line()
    
    # (sample, user, row, col)
    Nt, N, M, K, Lm, Lk, Ltotal = parameters(total_user)
    batch_size = 32
    
    print(f'Loading training data...')
    
    G_train = np.load(f'train/{total_user}users/G_train.npy')
    Rg_train = np.load(f'train/{total_user}users/Rg_train.npy')
    phi_train = np.load(f'train/{total_user}users/phi_trainADMM.npy')
    
    print("Training data loaded.")
    print("====================================")
    print("Preprocessing the data...")
    
    G_stacked = stacking(G_train)
    Rg_stacked = stacking(Rg_train)
    
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
    phi_input = layers.Input(name='PhiInput', shape=(phi_train.shape[1:]), dtype=tf.float32)
    
    G_branch = build_branch(G_stacked_input)
    Rg_branch = build_branch(Rg_stacked_input)
    phi_flat = layers.Flatten()(phi_input)
    
    concatenated = layers.Concatenate()([G_branch, Rg_branch, phi_flat])
    x = layers.BatchNormalization()(concatenated)
    x = layers.Dense(512, activation='relu')(x)
    #x = layers.Dropout(0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    #x = layers.Dropout(0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    #x = layers.Dropout(0.1)(x)
    
    phi = layers.Dense(K * N, activation='linear')(x)
    
    phi_pred = layers.Lambda(reshape_phi,
                             dtype=tf.float32,
                             output_shape=(phi_input.shape[0], K, N, 1))([K, N, phi, phi_input])
    
    loss = layers.Lambda(loss_phi, 
                         dtype=tf.float32, 
                         output_shape=(1),
                         name="loss")([phi_input, phi_pred, N])
    
    model = keras.Model(inputs=[G_stacked_input, Rg_stacked_input, phi_input], outputs=loss)
    
    # ------------------------------------
    # Define Learning Rate Schedule
    # ------------------------------------
    lr_schedule = ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=10000,
        decay_rate=0.9)
    
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule,clipnorm=1.0)
    
    # Compile the model
    model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: y_pred)
    model.summary()
    
    # ------------------------------------
    # Define Callbacks (None of them should include lr_schedule)
    # ------------------------------------
    checkpoint = keras.callbacks.ModelCheckpoint(f'train/{total_user}users/phi_trainADMM.h5',
                                                 monitor='loss',
                                                 verbose=0,
                                                 save_best_only=True,
                                                 mode='min')
    early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=5, mode='min')
    reduced_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, min_delta=0.01, min_lr=1e-7)
    
    # ------------------------------------
    # Train the Model
    # ------------------------------------
    history = model.fit([G_stacked, Rg_stacked, phi_train],
                        y=phi_train,  # dummy variable
                        batch_size=batch_size,
                        epochs=100,
                        verbose=2,
                        validation_split=0.2,
                        callbacks=[checkpoint, early_stopping])

    # ------------------------------------
    # plot the loss curve
    # ------------------------------------
    loss_curve_phi(history, K, 'Supervised model for RIS coefficients')
    plt.savefig(f'train/{total_user}users/phi_loss_curve.png')
    plt.close()
                                                