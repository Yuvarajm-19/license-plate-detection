from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from PIL import Image
import pytesseract
from ultralytics import YOLO
import uuid

app = Flask(__name__)
CORS(app)

# Load YOLO model (Ensure the model file `best.pt` is in the correct path)
model = YOLO("best.pt")

# Folder to store uploaded and processed images
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def extract_text_from_image(image):
    """Extract text from an image using pytesseract."""
    return pytesseract.image_to_string(image, lang="eng").strip()


def preprocess_image_for_ocr(image):
    """Preprocess the cropped plate image for better OCR results."""
    image = image.convert('L')  # Convert to grayscale
    image = image.point(lambda p: p > 200 and 255)  # Increase contrast (binary image)
    return image


@app.route('/')
def index():
    """Render the main HTML page."""
    return render_template("index.html")


@app.route('/detect', methods=['POST'])
def detect_license_plate():
    # Check if an image file is in the request
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    # Save the uploaded image
    file = request.files['image']
    unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
    image_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(image_path)

    # Run YOLO model for detection
    results = model.predict(image_path)
    plate_image = None
    cropped_plate_path = None

    # Log the detection results to check if the plate is detected
    print(f"Detection results: {results}")

    # Crop the license plate from the image based on detected bounding box
    for result in results:
        for bbox in result.boxes.data:
            x1, y1, x2, y2, _ = bbox[:5].int().tolist()
            print(f"Detected bbox: {x1}, {y1}, {x2}, {y2}")
            img = Image.open(image_path)
            plate_image = img.crop((x1, y1, x2, y2))

    if plate_image:
        if plate_image.mode != 'RGB':
            plate_image = plate_image.convert('RGB')
        cropped_plate_path = os.path.join(
            UPLOAD_FOLDER, f"cropped_{uuid.uuid4().hex}.jpg"
        )
        plate_image.save(cropped_plate_path)

    return jsonify({
        "message": "Processed successfully",
        "uploaded_image": image_path,
        "plate_image": cropped_plate_path or image_path,
    }), 200


@app.route('/ocr', methods=['POST'])
def ocr_license_plate():
    # Check if plate image path is provided in the request
    data = request.json
    if "plate_image_path" not in data:
        return jsonify({"error": "No plate image path provided"}), 400

    plate_image_path = data["plate_image_path"]

    # Check if the image path exists
    if not os.path.exists(plate_image_path):
        return jsonify({"error": "Plate image file not found."}), 400

    plate_image = Image.open(plate_image_path)

    # Preprocess the image for OCR
    plate_image = preprocess_image_for_ocr(plate_image)

    # Extract text from the processed image
    text = extract_text_from_image(plate_image)

    return jsonify({
        "text": text,
        "plate_image": plate_image_path
    }), 200


if __name__ == '__main__':
    app.run(debug=True)
