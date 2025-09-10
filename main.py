"""FastAPI web service."""

import os

from fastapi import FastAPI
from fastapi import Response
from PIL import Image
import numpy as np
import pytesseract

from src.config import load_config
from src.fsl import build_siamese_network
from src.dataset import prepare_image
from src.pipeline import load_query_images
from src.pipeline import inference_pipeline
from src.schemas import ImageData

app = FastAPI()

# Load config
config = load_config("config.ini")

# Load trained model
model = build_siamese_network()
model.load_weights(config["model"]["path"])

# Load query images
query_images, query_labels = load_query_images(
    query_folder=config["query"]["path"])


@app.get("/")
async def root():
    """GET method to base endpoint."""

    message = {"status": 200, "message": ["This API is up and running!"]}
    return message


@app.get("/health")
async def health():
    """GET method to status endpoint."""

    message = {"status": 200, "message": ["This API is up and running!"]}
    return message


@app.post("/api/inference")
async def inference(data: ImageData, response: Response):
    """POST method to inference endpoint."""

    # Get data as JSON from POST
    image_path = data.model_dump()["image"]

    if not os.path.exists(image_path):
        message = {"status": 404, "message": ["Image not found!"]}
        response.status_code = 404
        return message

    # Prepare image (OpenCV pipeline)
    image = prepare_image(image_path,
                          target_size=eval(config["model"]["img_size"]))
    image = np.expand_dims(image, axis=0)
    scores = inference_pipeline(model, image, query_images)

    # Serialize response
    scores = [float(score) for score in scores]
    index = int(np.argmax(scores))

    response = {
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

    return response


@app.post("/api/inference-ocr")
async def inference_and_ocr(data: ImageData, response: Response):
    """POST method to inference + OCR endpoint."""

    # Get data as JSON from POST
    image_path = data.model_dump()["image"]

    if not os.path.exists(image_path):
        message = {"status": 404, "message": ["Image not found!"]}
        response.status_code = 404
        return message

    # Prepare image (OpenCV pipeline)
    image = prepare_image(image_path,
                          target_size=eval(config["model"]["img_size"]))
    image = np.expand_dims(image, axis=0)
    scores = inference_pipeline(model, image, query_images)

    # Serialize response
    scores = [float(score) for score in scores]
    index = int(np.argmax(scores))
    text = None

    # Process with OCR if ticket found
    if index == 0:
        ocr_img = Image.open(image_path)
        text = pytesseract.image_to_string(ocr_img)

    response = {
        "status":
        200,
        "message": [{
            "task": "Classification",
            "image": image_path,
            "class_probability": scores,
            "class_id": index,
            "class_name": query_labels[index],
            "text": text
        }]
    }

    return response
