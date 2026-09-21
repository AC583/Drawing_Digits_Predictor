import numpy as np

from PIL import Image


def preprocess(image: Image):
    # Grayscale
    image = image.convert("L")

    img = np.array(image)

    # Find where the drawing exists
    coords = np.argwhere(img > 20)

    if len(coords) == 0:
        return np.zeros((28, 28), dtype=np.float32)

    # Bounding box
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop digit
    img = img[y_min:y_max + 1, x_min:x_max + 1]

    # Convert back to PIL
    digit = Image.fromarray(img)

    # Preserve aspect ratio
    width, height = digit.size

    scale = 20 / max(width, height)

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    digit = digit.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    # Empty MNIST-sized canvas
    final = Image.new("L", (28, 28), 0)

    # Center digit
    x = (28 - new_width) // 2
    y = (28 - new_height) // 2

    final.paste(digit, (x, y))

    # Convert to numpy + normalize
    final = np.array(final).astype("float32") / 255.0

    return final