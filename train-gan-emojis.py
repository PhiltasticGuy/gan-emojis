import os

from matplotlib import pyplot

from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint

from keras.initializers import RandomNormal
from keras.optimizers import Adam
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout

# Display number of available GPUs on the local machine
import tensorflow as tf; print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))

# List locals devices available for TensorFlow
from tensorflow.python.client import device_lib; print(device_lib.list_local_devices())

# Enable debug logs for TensorFlow
# tf.debugging.set_log_device_placement(True)

def define_discriminator(in_shape=(48, 48, 3), kernel_size=(3, 3)):
    model = Sequential()
    # Normal
    model.add(Conv2D(64, kernel_size, padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    # Downsample to 24x24
    model.add(Conv2D(128, kernel_size, strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # Downsample to 12x12
    model.add(Conv2D(256, kernel_size, strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # Downsample to 6x6
    model.add(Conv2D(512, kernel_size, strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # Classifier
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    # Compile the model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def define_generator(latent_dim, kernel_size=(5, 5)):
    # weight initialization
    model = Sequential()
    # Foundation for 6x6 feature maps
    n_nodes = 512 * 6 * 6
    model.add(Dense(n_nodes, activation='relu', input_dim=latent_dim))
    # model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((6, 6, 512)))
    # Upsample to 12x12
    model.add(Conv2DTranspose(256, kernel_size, activation='relu', strides=(2, 2), padding='same'))
    # model.add(LeakyReLU(alpha=0.2))
    # Upsample to 24x24
    model.add(Conv2DTranspose(128, kernel_size, activation='relu', strides=(2, 2), padding='same'))
    # model.add(LeakyReLU(alpha=0.2))
    # Upsample to 48x48
    model.add(Conv2DTranspose(64, kernel_size, activation='relu', strides=(2, 2), padding='same'))
    # model.add(LeakyReLU(alpha=0.2))
    # Output layer at 48x48x3
    model.add(Conv2D(3, (3,3), activation='tanh', padding='same'))
    return model

def define_gan(d_model, g_model):
    # Freeze the discriminator model's weights from the GAN model
    d_model.trainable = False
    # Connect the discriminator and generator
    model = Sequential()
    # Add the generator
    model.add(g_model)
    # Add the discriminator
    model.add(d_model)
    # Compile the model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    # Unfreeze the discriminator model's weights
    d_model.trainable = True
    return model

def load_real_samples(dataset_file='emojis-dataset.npz'):
    data = load(dataset_file)
    X = data['arr_0']
    # Convert from unsigned ints to floats
    X = X.astype('float32')
    # Scale from [0, 255] to [-1, 1] for the 'tanh' activation function
    X = (X - 127.5) / 127.5
    return X

def generate_real_samples(dataset, n_samples):
    # Choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # Retrieve selected images
    X = dataset[ix]
    # Generate 'real' class labels (1)
    y = ones((n_samples, 1))
    return X, y

def generate_latent_points(latent_dim, n_samples):
    # Generate points in latent space
    x_input = randn(latent_dim * n_samples)
    # Reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

def generate_fake_samples(g_model, latent_dim, n_samples):
    # Generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # Predict outputs
    X = g_model.predict(x_input)
    # Create 'fake' class labels (0)
    y = zeros((n_samples, 1))
    return X, y

def save_plot(filename, examples, n=10):
    # scale from [-1,1] to [0,1]
    examples = (examples + 1) / 2.0

    # plot images
    for i in range(n * n):
        pyplot.subplot(n, n, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(examples[i])

    # Save plot to file
    pyplot.savefig(filename)
    pyplot.close()

def summarize_performance(epoch, d_model, g_model, dataset, latent_dim, n_samples=100, directory='out_dir'):
    # Prepare real samples
    x_real, y_real = generate_real_samples(dataset, n_samples)
    # Evaluate discriminator on real examples
    _, acc_real = d_model.evaluate(x_real, y_real, verbose=0)
    # Prepare fake samples
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    # Evaluate generator on fake samples
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)

    # Summarize discriminator performance
    print('> Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))

    # TODO: Add historical graphs for the loss over time
    # TODO: Track total training time

    # Save the discriminator model to file
    filename = directory + '/e%03d_discriminator.h5' % (epoch+1)
    d_model.save(filename, include_optimizer=True)

    # Save the generator model to file
    filename = directory + '/e%03d_generator.h5' % (epoch+1)
    g_model.save(filename)

    # Save the emojis samples generated at this epoch
    filename = directory + '/emojis_training_e%03d.png' % (epoch+1)
    save_plot(filename, x_fake)

def train(d_model, g_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch_size=128):
    # Ensure that the output directory exists
    runs_dir = 'runs'
    if (not os.path.exists(runs_dir)):
        os.makedirs(runs_dir)

    # Count how many child directories are in the 'runs' directory
    run_count = sum(os.path.isdir(os.path.join(runs_dir, i)) for i in os.listdir(runs_dir))

    # HACK: Do/While Loop
    while True:
        # Build the output directory name
        out_dir = f'{runs_dir}/run-{run_count}'

        # Increment until we find an unused directory name
        if (os.path.exists(out_dir)):
            run_count += 1
        else:
            # Create the new output directory 
            os.makedirs(out_dir)
            break

    # Calculate batch related parameters
    half_batch = int(n_batch_size / 2)
    batch_per_epoch = int(dataset.shape[0] / n_batch_size)

    for i in range(n_epochs):
        for j in range(batch_per_epoch):
            # Get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # Update discriminator model weights
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)
            
            # Generate 'fake' samples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # Update discriminator model weghts
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)

            # Prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch_size)
            # Create inverted labels for the fake samples
            y_gan = ones((n_batch_size, 1))
            # Update the generator via the discriminator's loss
            g_loss = gan_model.train_on_batch(X_gan, y_gan)

            # Summarize loss on this batch
            print('> %d, %d/%d, d1=%.3f, d2=%.3f, g=%.3f' %
                (i+1, j+1, batch_per_epoch, d_loss1, d_loss2, g_loss))

        # Evaluate the model performance
        if ((i+1) % 5 == 0):
            summarize_performance(i, d_model, g_model, dataset, latent_dim, directory=out_dir)

# Size of the latent space
latent_dim = 100

# Create the discriminator
# d_model = define_discriminator()
d_model = load_model('e075_discriminator.h5')

# Create the generator
# g_model = define_generator(latent_dim)
g_model = load_model('e075_generator.h5')

# Create the GAN
gan_model = define_gan(d_model, g_model)

# Load the emojis
dataset = load_real_samples()

# Train!!!
train(d_model, g_model, gan_model, dataset, latent_dim, n_epochs=75)