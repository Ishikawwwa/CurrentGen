import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def build_generator(latent_dim, seq_length=9, patch_size=64):
    noise_input = layers.Input(shape=(latent_dim,))
    x = layers.Dense(8 * 8 * 256, use_bias=False)(noise_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((8, 8, 256))(x)
    
    def create_timestep_model():
        model = Sequential([
            layers.Conv2DTranspose(128, (5,5), strides=(2,2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            
            layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            
            layers.Conv2DTranspose(32, (5,5), strides=(2,2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            
            layers.Conv2D(1, (5,5), padding='same', activation='tanh')
        ])
        return model
    
    timestep_models = [create_timestep_model() for _ in range(seq_length)]
    outputs = [model(x) for model in timestep_models]
    
    output_sequence = layers.Concatenate(axis=1)([layers.Reshape((1, patch_size, patch_size, 1))(out) for out in outputs])
    
    return Model(noise_input, output_sequence)

def build_discriminator(seq_length=9, patch_size=64):
    model = Sequential([
        layers.Input(shape=(seq_length, patch_size, patch_size, 1)),
        
        layers.Conv3D(32, (3,5,5), strides=(1,2,2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        
        layers.Conv3D(64, (3,5,5), strides=(1,2,2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        
        layers.Conv3D(128, (3,5,5), strides=(1,2,2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

class TGAN(Model):
    def __init__(self, generator, discriminator, latent_dim):
        super(TGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        
    def compile(self, g_optimizer, d_optimizer, loss_fn):
        super(TGAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn
        self.g_loss_metric = tf.keras.metrics.Mean(name='g_loss')
        self.d_loss_metric = tf.keras.metrics.Mean(name='d_loss')
        
    @property
    def metrics(self):
        return [self.g_loss_metric, self.d_loss_metric]
        
    def train_step(self, real_sequences):
        batch_size = tf.shape(real_sequences)[0]
        
        noise = tf.random.normal([batch_size, self.latent_dim])
        with tf.GradientTape() as d_tape:
            generated_sequences = self.generator(noise, training=True)
            real_output = self.discriminator(real_sequences, training=True)
            fake_output = self.discriminator(generated_sequences, training=True)
            
            real_loss = self.loss_fn(tf.ones_like(real_output), real_output)
            fake_loss = self.loss_fn(tf.zeros_like(fake_output), fake_output)
            d_loss = (real_loss + fake_loss) / 2
            
        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(
            zip(d_gradients, self.discriminator.trainable_variables))
        
        noise = tf.random.normal([batch_size, self.latent_dim])
        with tf.GradientTape() as g_tape:
            generated_sequences = self.generator(noise, training=True)
            fake_output = self.discriminator(generated_sequences, training=True)
            g_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)
            
        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(g_gradients, self.generator.trainable_variables))
        
        self.g_loss_metric.update_state(g_loss)
        self.d_loss_metric.update_state(d_loss)
        
        return {
            "g_loss": self.g_loss_metric.result(),
            "d_loss": self.d_loss_metric.result()
        }

def train_tgan(X_train, epochs=10, batch_size=32, latent_dim=128):
    generator = build_generator(latent_dim)
    discriminator = build_discriminator()
    
    tgan = TGAN(generator, discriminator, latent_dim)
    
    g_optimizer = Adam(learning_rate=1e-4, beta_1=0.5)
    d_optimizer = Adam(learning_rate=1e-4, beta_1=0.5)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    
    tgan.compile(g_optimizer, d_optimizer, loss_fn)
    
    dataset = tf.data.Dataset.from_tensor_slices(X_train)
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        for batch in tqdm(dataset):
            tgan.train_step(batch)
        
        if (epoch + 1) % 10 == 0:
            generate_and_save_samples(generator, epoch+1, latent_dim)
    
    return tgan

def generate_and_save_samples(generator, epoch, latent_dim, n_samples=3):
    noise = tf.random.normal([n_samples, latent_dim])
    generated = generator(noise, training=False)
    
    plt.figure(figsize=(15, 5*n_samples))
    for i in range(n_samples):
        plt.subplot(n_samples, 3, i*3+1)
        plt.imshow(generated[i,0,...,0], cmap='Blues', vmin=-1, vmax=1)
        plt.title(f"Sample {i+1}\nFirst Frame")
        
        plt.subplot(n_samples, 3, i*3+2)
        plt.imshow(generated[i,len(generated[i])//2,...,0], cmap='Blues')
        plt.title("Middle Frame")
        
        plt.subplot(n_samples, 3, i*3+3)
        plt.imshow(generated[i,-1,...,0], cmap='Blues')
        plt.title(f"Last Frame\nEpoch {epoch}")
    
    plt.tight_layout()
    plt.savefig(f"generated_samples_epoch_{epoch}.png")
    plt.show()

try:
    print(f"Training data shape: {X_train.shape}")
    
    tgan = train_tgan(X_train, epochs=30, batch_size=32)
    
    tgan.generator.save("tgan_generator.h5")
    tgan.discriminator.save("tgan_discriminator.h5")
    print("Training completed and models saved!")
    
except Exception as e:
    print(f"Training failed: {str(e)}")