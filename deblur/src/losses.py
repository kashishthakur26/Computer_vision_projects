import tensorflow.keras.backend as K
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf

image_shape = (256,256,3)
vgg = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)
feature_extractor = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)

def l1_loss(y_true, y_pred):
  return K.mean(K.abs(y_pred - y_true))

def perceptual_loss_100(y_true, y_pred):
  return 100*perceptual_loss(y_true, y_pred)

def perceptual_loss(y_true, y_pred):
    y_true_resized = tf.image.resize(y_true, (256, 256))
    y_pred_resized = tf.image.resize(y_pred, (256, 256))
    true_features = feature_extractor(y_true_resized)
    pred_features = feature_extractor(y_pred_resized)
    return tf.reduce_mean(tf.square(true_features - pred_features))



def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)


def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))

    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = K.square(1 - gradient_l2_norm)

    return K.mean(gradient_penalty)