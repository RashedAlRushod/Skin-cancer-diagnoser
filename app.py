from flask import Flask, render_template, request, redirect, url_for
from keras.models import model_from_json
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load the model architecture from the JSON file
json_file_path = os.path.join(app.root_path, 'resnet50.json')
with open(json_file_path, 'r') as json_file:
    loaded_model_json = json_file.read()

# Create a model from the loaded architecture
model = model_from_json(loaded_model_json)

# Load the model weights from the .h5 file
weights_file_path = os.path.join(app.root_path, 'resnet50.h5')
model.load_weights(weights_file_path)
print("Model loaded successfully.")

# Function to preprocess the image and make a prediction
def predict_image(image_path):
    # Load the image
    img = Image.open(image_path).convert('RGB')
    # Resize the image to match the input size expected by the model
    img = img.resize((224, 224))  # Adjust the size if your model expects a different input size
    # Convert the image to a NumPy array
    img_array = np.array(img)
    # Normalize the image data if required by the model
    img_array = img_array / 255.0
    # Expand dimensions to match the model's input shape
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction using the loaded model
    prediction = model.predict(img_array)
    # Interpret the prediction
    predicted_class = np.argmax(prediction, axis=1)[0]
    if predicted_class == 0:
        result = 'Benign'
    else:
        result = 'Malignant'
    return result

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if an image was uploaded
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Save the uploaded image to the static folder
            image_name = file.filename  # Get the uploaded file name
            image_path = os.path.join(app.root_path, 'static', 'uploaded_image.jpg')
            file.save(image_path)
            # Make prediction
            result = predict_image(image_path)
            # Redirect to the result page, passing prediction and image name
            return redirect(url_for('result', prediction=result, image_name=image_name))
    return render_template('index.html')

@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    image_name = request.args.get('image_name')  # Retrieve the image name
    return render_template('result.html', prediction=prediction, image_name=image_name)

if __name__ == '__main__':
    app.run(debug=True)
