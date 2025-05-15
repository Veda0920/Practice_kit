from flask import Flask, render_template, request
from PIL import Image
import pytesseract
import os
import cv2
import numpy as np
import shutil

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Remove hardcoded Windows path for Tesseract
# pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

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
        image_file = request.files.get('image')
        if image_file and image_file.filename != '':
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])

            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            print("Saving image to:", image_path)
            image_file.save(image_path)

            try:
                pil_image = Image.open(image_path)
                print("Image opened successfully")
            except Exception as e:
                print("Error opening image:", e)
                return render_template('index.html', extracted_text="Error: Unable to open image.")

            try:
                preprocessed_image = preprocess_image(pil_image)
            except Exception as e:
                print("Error preprocessing image:", e)
                return render_template('index.html', extracted_text="Error: Unable to preprocess image.")

            print("Tesseract path:", shutil.which("tesseract"))

            try:
                config = r"--psm 6 --oem 3"
                extracted_text = pytesseract.image_to_string(preprocessed_image, config=config)
                print("Text extracted successfully")
            except Exception as e:
                print("Error extracting text:", e)
                return render_template('index.html', extracted_text="Error: OCR failed.")

        else:
            print("No file uploaded or filename empty")
            extracted_text = "Please upload an image file."

    return render_template('index.html', extracted_text=extracted_text)

@app.route('/healthz')
def health_check():
    return 'OK', 200

if __name__ == '__main__':
    app.run(debug=True)

