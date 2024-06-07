from flask import Flask, request , jsonify, render_template
from predicted import PredictionPipeline
import os


app = Flask(__name__)

# Ensure the output directory exists
if not os.path.exists('output'):
    os.makedirs('output')

# Define the weight path
WEIGHT_PATH = r'C:\Users\91623\Desktop\Computer_vision_projects\deblur\weights\20240607_170910_0\generator_0_0.weights.h5'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/deblur', methods=['POST'])
def deblur_image():
    if 'file' not in request.files:
        return "No file part in the request", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected for uploading", 400

    if file:
        filename = os.path.join('uploads', file.filename)
        file.save(filename)

        # Process the image through the deblurring pipeline
        pipeline = PredictionPipeline(filename, WEIGHT_PATH)
        output_path = pipeline.deblur()

        # Return the deblurred image
        return render_template('index.html', uploaded_image=filename, deblurred_image=output_path)

if __name__ == '__main__':
    app.run(debug=True)
