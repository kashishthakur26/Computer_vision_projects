import numpy as np
from PIL import Image
import click
import os


from src.model import generator_model
from src.utils import load_image, deprocess_image, preprocess_image


def deblur(weight_path, input_dir, output_dir):
	g = generator_model()
	g.load_weights(weight_path)
	for image_name in os.listdir(input_dir):
		image = np.array([preprocess_image(load_image(os.path.join(input_dir, image_name)))])
		x_test = image
		generated_images = g.predict(x=x_test)
		generated = np.array([deprocess_image(img) for img in generated_images])
		x_test = deprocess_image(x_test)
		for i in range(generated_images.shape[0]):
			x = x_test[i, :, :, :]
			img = generated[i, :, :, :]
			output = np.concatenate((x, img), axis=1)
			im = Image.fromarray(output.astype(np.uint8))
			im.save(os.path.join(output_dir, image_name))

weight_path = r'C:\Users\91623\Desktop\Computer_vision_projects\deblur\weights\20240607_170910_0\generator_0_0.weights.h5'
output_dir = r'C:\Users\91623\Desktop\Computer_vision_projects\deblur\test_image'
input_dir = r'C:\Users\91623\Desktop\Computer_vision_projects\deblur\input_test'

deblur(weight_path, input_dir, output_dir)


