from flask import Flask, render_template, request
from PIL import Image
import pytesseract
import os
import cv2
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Tesseract path


def preprocess_image(pil_image):
    # Convert PIL to OpenCV
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Denoise and reduce blur using bilateral filter
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 31, 2)

    return Image.fromarray(thresh)

@app.route('/', methods=['GET', 'POST'])
def index():
    extracted_text = ''
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])

            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            image_file.save(image_path)

            pil_image = Image.open(image_path)
            preprocessed_image = preprocess_image(pil_image)

            config = r"--psm 6 --oem 3"
            extracted_text = pytesseract.image_to_string(preprocessed_image, config=config)

    return render_template('index.html', extracted_text=extracted_text)

if __name__ == '__main__':
    app.run(debug=True)
