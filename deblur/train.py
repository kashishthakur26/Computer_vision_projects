from keras.callbacks import TensorBoard
from keras.optimizers import Adam
import datetime
import os
from src.utils import load_images
from src.model import generator_model, discriminator_model, generator_containing_discriminator_multiple_outputs
import numpy as np
import tqdm
from src.losses import wasserstein_loss, perceptual_loss


BASE_DIR = 'weigths/'

import os
import datetime

BASE_DIR = 'weights/'

def save_all_weights(d, g, epoch_number, current_loss):
    now = datetime.datetime.now()
    save_dir = os.path.join(BASE_DIR, '{}_{}'.format(now.strftime("%Y%m%d_%H%M%S"), epoch_number, current_loss))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    g.save_weights(os.path.join(save_dir, 'generator_{}_{}.weights.h5'.format(epoch_number, current_loss)))
    d.save_weights(os.path.join(save_dir, 'discriminator_{}_{}.weights.h5'.format(epoch_number, current_loss)))

def train_multiple_outputs(n_images, batch_size, epoch_num, critic_updates=5):
  data = load_images('output/train', n_images)
  y_train, x_train = data['B'], data['A']

  g = generator_model()
  d = discriminator_model()

  d_on_g = generator_containing_discriminator_multiple_outputs(g,d)

  d_opt = Adam(learning_rate=1e-4, beta_1=0.9, beta_2= 0.999, epsilon=1e-08)
  d_on_g_opt = Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

  d.trainable = True
  d.compile(optimizer=d_opt, loss=wasserstein_loss)
  d.trainable = False
  loss = [perceptual_loss, wasserstein_loss]
  loss_weights = [100,1]
  d_on_g.compile(optimizer=d_on_g_opt, loss=loss, loss_weights=loss_weights)
  d.trainable = True
  output_true_batch, output_false_batch = np.ones((batch_size, 1)), -np.ones((batch_size, 1))
#   log_path = './logs'
#   tensorboard_callback = TensorBoard(log_path)

  for epoch in tqdm.tqdm(range(epoch_num)):
        permutated_indexes = np.random.permutation(x_train.shape[0])

        d_losses = []
        d_on_g_losses = []
        for index in range(int(x_train.shape[0] / batch_size)):
            batch_indexes = permutated_indexes[index*batch_size:(index+1)*batch_size]
            image_blur_batch = x_train[batch_indexes]
            image_full_batch = y_train[batch_indexes]

            generated_images = g.predict(x=image_blur_batch, batch_size=batch_size)

            for _ in range(critic_updates):
                d_loss_real = d.train_on_batch(image_full_batch, output_true_batch)
                d_loss_fake = d.train_on_batch(generated_images, output_false_batch)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
                d_losses.append(d_loss)

            d.trainable = False

            d_on_g_loss = d_on_g.train_on_batch(image_blur_batch, [image_full_batch, output_true_batch])
            d_on_g_losses.append(d_on_g_loss)

            d.trainable = True

        # write_log(tensorboard_callback, ['g_loss', 'd_on_g_loss'], [np.mean(d_losses), np.mean(d_on_g_losses)], epoch_num)
        # print(np.mean(d_losses), np.mean(d_on_g_losses))
        # with open('log.txt', 'a+') as f:
        #     f.write('{} - {} - {}\n'.format(epoch, np.mean(d_losses), np.mean(d_on_g_losses)))

        save_all_weights(d, g, epoch, int(np.mean(d_on_g_losses)))



train_multiple_outputs(100, 16, True, 4)
