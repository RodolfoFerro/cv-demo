"""API module."""

import base64

from flask import Flask
from flask import request
from flask import jsonify
import numpy as np

from src.config import load_config
from src.fsl import build_siamese_network
from src.dataset import prepare_image
from src.pipeline import load_query_images
from src.pipeline import inference_pipeline

# Create Flask app
app = Flask(__name__)

# Load config
config = load_config("config.ini")

# Load trained model
model = build_siamese_network()
model.load_weights(config["model"]["path"])

# Load query images
query_images, query_labels = load_query_images(
    query_folder=config["query"]["path"])


@app.route("/", methods=["GET"])
async def status():
    """Base route for API status."""

    message = {"status": 200, "message": ["This API is up and running!"]}
    response = jsonify(message)
    response.status_code = 200

    return response


@app.route("/api/inference", methods=["POST"])
async def inference():
    """"""

    # Get data as JSON from POST
    data = request.get_json()
    image_path = data["image"]

    # Prepare image (OpenCV pipeline)
    image = prepare_image(image_path,
                          target_size=eval(config["model"]["img_size"]))
    image = np.expand_dims(image, axis=0)
    scores = inference_pipeline(model, image, query_images)

    # Serialize response
    scores = [float(score) for score in scores]
    index = int(np.argmax(scores))

    message = {
        "status":
        200,
        "message": [{
            "task": "Classification",
            "image": image_path,
            "class_probability": scores,
            "class_id": index,
            "class_name": query_labels[index]
        }]
    }
    response = jsonify(message)
    response.status_code = 200

    return response


@app.route("/api/encode", methods=["POST"])
async def encode_image():
    """"""

    # Get data as JSON from POST
    data = request.get_json()
    image_path = data["image"]

    with open(image_path, "rb") as image_file:
        binary_image_data = image_file.read()
    img_b64 = base64.b64encode(binary_image_data)

    message = {
        "status":
        200,
        "message": [{
            "task": "Encoding",
            "image": image_path,
            "base64": str(img_b64)
        }]
    }
    response = jsonify(message)
    response.status_code = 200

    return response


@app.errorhandler(404)
async def not_found(error=None):
    """GET method for not found routes."""

    message = {"status": 404, "message": ["[ERROR] URL not found."]}
    response = jsonify(message)
    response.status_code = 404

    return response


if __name__ == "__main__":
    app.run(port=5000, debug=True)
