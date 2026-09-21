from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles

import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

model = tf.keras.models.load_model("mnist_model.keras")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    image = Image.open(io.BytesIO(contents))

    # Convert to grayscale
    image = image.convert("L")

    # MNIST images are 28x28
    image = image.resize((28, 28))

    # Convert image to numpy array
    image = np.array(image).astype("float32")

    # Normalize exactly like training data
    image = image / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    # Make prediction
    logits = model.predict(image)

    probabilities = tf.nn.softmax(logits[0]).numpy()

    digit = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))

    return {
        "digit": digit,
        "confidence": confidence
    }


app.mount("/", StaticFiles(directory="static", html=True), name="static")