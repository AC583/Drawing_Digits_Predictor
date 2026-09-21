from preprocessing import preprocess_multiple

from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles

import tensorflow as tf
import numpy as np

from PIL import Image

import io
import base64


app = FastAPI()

model = tf.keras.models.load_model("mnist_model.keras")


def create_processed_image_url(processed_digits):
    """
    Creates ONE PNG showing exactly what was sent to the model.

    Example for 3 digits:

        28x28   28x28   28x28
        digit   digit   digit

    These are combined horizontally for display.
    """

    if len(processed_digits) == 0:
        return None

    # Add a small gap between digits
    gap = 4

    width = (
        len(processed_digits) * 28
        + (len(processed_digits) - 1) * gap
    )

    combined = Image.new(
        "L",
        (width, 28),
        0
    )

    for i, processed_array in enumerate(processed_digits):

        processed_uint8 = (
            processed_array * 255
        ).astype(np.uint8)

        processed_pil = Image.fromarray(processed_uint8)

        x = i * (28 + gap)

        combined.paste(
            processed_pil,
            (x, 0)
        )

    # Convert to Base64 PNG
    buf = io.BytesIO()

    combined.save(
        buf,
        format="PNG"
    )

    b64_str = base64.b64encode(
        buf.getvalue()
    ).decode("utf-8")

    return f"data:image/png;base64,{b64_str}"

def create_processed_image_urls(processed_digits):
    """
    Convert every 28x28 processed digit into its own
    Base64 PNG data URL.

    The order exactly matches model_input.
    """

    processed_image_urls = []

    for processed_array in processed_digits:

        # Convert normalized [0, 1] image back to [0, 255]
        processed_uint8 = (
            processed_array * 255
        ).astype(np.uint8)

        processed_pil = Image.fromarray(processed_uint8)

        # Encode this individual digit as PNG
        buf = io.BytesIO()
        processed_pil.save(buf, format="PNG")

        b64_str = base64.b64encode(
            buf.getvalue()
        ).decode("utf-8")

        processed_image_urls.append(
            f"data:image/png;base64,{b64_str}"
        )

    return processed_image_urls

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read uploaded canvas image
    contents = await file.read()

    image = Image.open(
        io.BytesIO(contents)
    )

    # Find and preprocess every digit

    processed_rows = preprocess_multiple(image)

    if len(processed_rows) == 0:
        return {
            "digit": "",
            "number": "",
            "confidence": 0.0,
            "digits": [],
            "numbers": [],
            "processed_images": []
        }


    # Flatten everything temporarily for TensorFlow
    processed_digits = []

    for row in processed_rows:
        for number in row:
            for digit in number:
                processed_digits.append(digit)


    model_input = np.stack(processed_digits)


    # Predict every digit
    logits = model.predict(model_input)

    results = []
    prediction_index = 0
    predicted_rows = []
    confidences = []

    for row in processed_rows:

        predicted_row = []

        for number in row:

            predicted_number = ""

            digit_results = []

            for digit_image in number:

                logit = logits[prediction_index]

                probabilities = tf.nn.softmax(
                    logit
                ).numpy()

                digit = int(
                    np.argmax(probabilities)
                )

                confidence = float(
                    np.max(probabilities)
                )

                predicted_number += str(digit)

                confidences.append(
                    confidence
                )

                digit_results.append({
                    "digit": digit,
                    "confidence": confidence
                })

                results.append({
                    "digit": digit,
                    "confidence": confidence
                })

                prediction_index += 1


            predicted_row.append({
                "number": predicted_number,
                "digits": digit_results
            })


        predicted_rows.append(
            predicted_row
        )


    # Create processed-image preview
    processed_image_urls = create_processed_image_urls(
        processed_digits
    )

    return {
        "rows": predicted_rows,
        "confidence": float(np.mean(confidences)),
        "digits": results,
        "processed_images": processed_image_urls
    }


app.mount(
    "/",
    StaticFiles(
        directory="static",
        html=True
    ),
    name="static"
)