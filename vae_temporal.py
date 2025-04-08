import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt

class TemporalVAE(Model):
    def __init__(self, input_shape, latent_dim=64, filters=32):
        super(TemporalVAE, self).__init__()
        self.input_shape_ = input_shape
        self.latent_dim = latent_dim
        self.filters = filters
        
        self.encoder = self._build_encoder()
        
        self.decoder = self._build_decoder()
        
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        
    def _build_encoder(self):
        encoder_inputs = layers.Input(shape=self.input_shape_)
        
        x = layers.ConvLSTM2D(
            self.filters, (3,3), padding='same',
            return_sequences=True,
            kernel_regularizer=regularizers.l2(1e-4)
        )(encoder_inputs)
        x = layers.BatchNormalization()(x)
        
        x = layers.ConvLSTM2D(
            self.filters*2, (3,3), padding='same',
            return_sequences=False
        )(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2D(
            self.filters*4, (3,3), strides=2, padding='same',
            activation='relu'
        )(x)
        x = layers.Conv2D(
            self.filters*8, (3,3), strides=2, padding='same',
            activation='relu'
        )(x)
        
        x = layers.Flatten()(x)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        
        return Model(encoder_inputs, [z_mean, z_log_var], name="encoder")
    
    def _build_decoder(self):
        latent_inputs = layers.Input(shape=(self.latent_dim,))
        
        x = layers.Dense(self.filters*8 * (self.input_shape_[1]//4) * (self.input_shape_[2]//4))(latent_inputs)
        x = layers.Reshape((self.input_shape_[1]//4, self.input_shape_[2]//4, self.filters*8))(x)
        
        x = layers.Conv2DTranspose(
            self.filters*4, (3,3), strides=2, padding='same',
            activation='relu'
        )(x)
        x = layers.Conv2DTranspose(
            self.filters*2, (3,3), strides=2, padding='same',
            activation='relu'
        )(x)
        
        x = layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(x)
        x = layers.Lambda(lambda x: tf.tile(x, [1, self.input_shape_[0], 1, 1, 1]))(x)
        
        x = layers.ConvLSTM2D(
            self.filters*2, (3,3), padding='same',
            return_sequences=True
        )(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.ConvLSTM2D(
            self.filters, (3,3), padding='same',
            return_sequences=True
        )(x)
        x = layers.BatchNormalization()(x)
        
        decoder_outputs = layers.TimeDistributed(
            layers.Conv2D(
                self.input_shape_[-1], (3,3), padding='same',
                activation='tanh'
            )
        )(x)
        
        return Model(latent_inputs, decoder_outputs, name="decoder")
    
    @tf.function
    def sample(self, z_mean, z_log_var):
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = self.sample(z_mean, z_log_var)
        reconstructions = self.decoder(z)
        return reconstructions
    
    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
            
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(data)
            z = self.sample(z_mean, z_log_var)
            reconstructions = self.decoder(z)
            
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.mse(data, reconstructions),
                    axis=(1,2,3)
                )
            )
            
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                    axis=1
                )
            )
            
            total_loss = reconstruction_loss + kl_loss
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

def train_temporal_vae(X_train, X_val, seq_len=10, patch_size=64, channels=1,
                      latent_dim=64, filters=32, epochs=100, batch_size=8):
    input_shape = (seq_len, patch_size, patch_size, channels)
    
    vae = TemporalVAE(input_shape, latent_dim, filters)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
    
    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5, verbose=1),
        ModelCheckpoint('temporal_vae_best.keras', save_best_only=True)
    ]
    
    history = vae.fit(
        X_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val,),
        callbacks=callbacks,
        verbose=1
    )
    
    return vae, history

if 'X_train' in locals() and 'X_test' in locals():
    seq_len = X_train.shape[1]
    patch_size = X_train.shape[2]
    channels = X_train.shape[4]
    
    vae, history = train_temporal_vae(
        X_train, X_test,
        seq_len=seq_len,
        patch_size=patch_size,
        channels=channels,
        latent_dim=64,
        filters=32,
        epochs=20,
        batch_size=8
    )
    
    vae.save('ocean_current_temporal_vae.keras')
    print("Training completed and model saved!")
    
    reconstructions = vae.predict(X_test[:3])
    plt.figure(figsize=(15, 9))
    for i in range(3):
        for j in range(3):
            plt.subplot(3, 6, i*6 + j + 1)
            plt.imshow(X_test[i,j,...,0], cmap='Blues', vmin=-1, vmax=1)
            plt.title(f'Sample {i+1}\nOriginal t={j}')
            plt.subplot(3, 6, i*6 + j + 4)
            plt.imshow(reconstructions[i,j,...,0], cmap='Blues', vmin=-1, vmax=1)
            plt.title(f'Reconstructed t={j}')
    plt.tight_layout()
    plt.show()
    
else:
    print("Training data not found by some reason")