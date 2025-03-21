from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS

import os

UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
CORS(app)  # Allow all origins

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Get the name from the form data
    name = request.form.get('name')
    if not name:
        return jsonify({"error": "No name provided"}), 400

    # Save the file to server (local path for testing)
    file.save(f"./uploads/{name}.jpg")
    return jsonify({"message": "File uploaded successfully!"})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
