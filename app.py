from PIL import ImageCms
from preprocessing import preprocess
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles

import tensorflow as tf
import numpy as np
from PIL import Image
import io

import base64

import matplotlib.pyplot as plt


app = FastAPI()

model = tf.keras.models.load_model("mnist_model.keras")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    image = Image.open(io.BytesIO(contents))

    processed_array = preprocess(image)

    # Convert processed array (28x28 float32 in [0, 1]) to Base64 PNG
    processed_uint8 = (processed_array * 255).astype(np.uint8)
    processed_pil = Image.fromarray(processed_uint8)
    buf = io.BytesIO()
    processed_pil.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    processed_image_url = f"data:image/png;base64,{b64_str}"

    model_input = np.expand_dims(processed_array, axis=0)

    # Make prediction
    logits = model.predict(model_input)

    probabilities = tf.nn.softmax(logits[0]).numpy()

    digit = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))

    return {
        "digit": digit,
        "confidence": confidence,
        "processed_image": processed_image_url
    }


app.mount("/", StaticFiles(directory="static", html=True), name="static")