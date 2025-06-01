# Image to Text OCR Web Application

## Description
This web application allows users to upload an image file (PNG, JPG, JPEG) and extracts text from it using Optical Character Recognition (OCR). The extracted text is then displayed on the webpage, allowing users to easily copy it.

## Features
- Upload images (PNG, JPG, JPEG).
- Image preprocessing for improved OCR accuracy (grayscale, blur, denoise, thresholding).
- Text extraction using Tesseract OCR.
- Display extracted text in a user-friendly interface.
- "Copy Text" button for easy copying to clipboard.
- Health check endpoint (`/healthz`).

## Technologies Used
- **Backend:** Python, Flask
- **OCR Engine:** Tesseract OCR
- **Python OCR Wrapper:** Pytesseract
- **Image Processing:** Pillow (PIL), OpenCV, NumPy
- **Frontend:** HTML, Tailwind CSS (via CDN), JavaScript
- **Deployment (Example):** Gunicorn, Render

## Setup and Installation

### Prerequisites
- Python 3.7+
- Tesseract OCR Engine

### Installing Tesseract OCR
**Linux (Debian/Ubuntu):**
```bash
sudo apt update
sudo apt install tesseract-ocr
```
**macOS:**
```bash
brew install tesseract
```
**Windows:**
- Download the installer from the [official Tesseract at UB Mannheim page](https://github.com/UB-Mannheim/tesseract/wiki).
- During installation, make sure to add Tesseract to your system PATH or note the installation directory to set `TESSERACT_PATH` manually.

*(For deployment on platforms like Render, Tesseract is installed via build commands, as seen in `render.yaml`.)*

### Project Setup
1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    # .\venv\Scripts\activate
    # On macOS/Linux
    # source venv/bin/activate
    ```
3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

### Locally
1.  **Set Tesseract Path (if needed):**
    - If Tesseract is not in your system's PATH, you might need to tell `pytesseract` where to find it. The `app.py` script attempts to read the `TESSERACT_PATH` environment variable.
    - Alternatively, on Windows, it checks the default installation path (`C:/Program Files/Tesseract-OCR/tesseract.exe`).
    - You can set the `TESSERACT_PATH` environment variable before running the app:
      ```bash
      # For Linux/macOS
      # export TESSERACT_PATH="/usr/local/bin/tesseract" # Or your actual path
      # For Windows (Command Prompt)
      # set TESSERACT_PATH="C:\Program Files\Tesseract-OCR\tesseract.exe"
      ```
2.  **Run the Flask development server:**
    ```bash
    python app.py
    ```
    The application will typically be available at `http://127.0.0.1:5000/`.

### Deployment
- The project includes a `render.yaml` file as an example for deploying to the Render platform.
- It uses Gunicorn as the WSGI server for production.
- The `render.yaml` handles the installation of Tesseract OCR and Python dependencies during the build process.
- The `TESSERACT_PATH` environment variable is set in `render.yaml` to `/usr/bin/tesseract` for the deployed environment.

## Project Structure
```
├── app.py            # Main Flask application logic
├── requirements.txt  # Python dependencies
├── render.yaml       # Deployment configuration for Render
├── static/           # Static assets (CSS, JS, images)
│   ├── background.jpg
│   └── style.css
├── templates/        # HTML templates
│   └── index.html
├── uploads/          # Directory for storing uploaded images (created automatically)
└── README.md         # This file
```

## Health Check
The application provides a health check endpoint at `/healthz`. A `GET` request to this endpoint will return an `OK` response with a 200 status code if the application is running.
