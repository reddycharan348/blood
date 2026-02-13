from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import onnxruntime as ort
from PIL import Image
import os
import io

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max upload

# Load ONNX model
model_path = "bloodgroup.onnx"
if not os.path.exists(model_path):
    print(f"Error: Model not found at {model_path}")
    sess = None
else:
    # Load model with CPU execution provider (safest for Vercel)
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

class_labels = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
blood_group_emojis = {
    'A+': '🅰️➕', 'A-': '🅰️➖',
    'B+': '🅱️➕', 'B-': '🅱️➖', 
    'AB+': '🆎➕', 'AB-': '🆎➖',
    'O+': '🅾️➕', 'O-': '🅾️➖'
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            try:
                # Read image
                img_bytes = file.read()
                img = Image.open(io.BytesIO(img_bytes))
                
                # Preprocess
                img = img.convert('L')
                img = img.resize((128, 128))
                x = np.array(img, dtype=np.float32) / 255.0
                x = np.expand_dims(x, axis=0)
                x = np.expand_dims(x, axis=-1) # Add channel dim if needed by model (1, 128, 128, 1)

                # Predict
                if sess:
                    pred = sess.run([output_name], {input_name: x})[0]
                    pred_class = np.argmax(pred)
                    confidence = float(np.max(pred)) * 100
                    
                    label = class_labels[pred_class]
                    emoji = blood_group_emojis.get(label, '')
                    
                    return render_template('index.html', 
                                         prediction=f"{emoji} {label}", 
                                         confidence=f"{confidence:.2f}%",
                                         image_data=file)
                else:
                    return "Model not loaded properly.", 500

            except Exception as e:
                return f"Error processing image: {str(e)}", 500
    
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
