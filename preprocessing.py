import numpy as np
from PIL import Image
import cv2

def find_digit_components(img):
    """
    Finds separate drawn components and returns their bounding boxes.
    """

    binary = (img > 20).astype(np.uint8) * 255

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    components = []

    # label 0 is the background
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        # Ignore tiny dots/noise
        if area < 20:
            continue

        components.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "center_x": x + w / 2,
            "center_y": y + h / 2,
            "image": img[y:y + h, x:x + w]
        })

    return components


def group_into_rows(components, row_threshold=None):
    """
    Groups digits based on their vertical position.
    """

    if not components:
        return []

    if row_threshold is None:
        average_height = np.mean([
            component["h"]
            for component in components
        ])

        row_threshold = average_height * 0.6

    components = sorted(
        components,
        key=lambda component: component["center_y"]
    )

    rows = []

    for component in components:
        placed = False

        for row in rows:
            average_y = np.mean([
                item["center_y"]
                for item in row
            ])

            if abs(component["center_y"] - average_y) <= row_threshold:
                row.append(component)
                placed = True
                break

        if not placed:
            rows.append([component])

    # Sort each row left -> right
    for row in rows:
        row.sort(
            key=lambda component: component["x"]
        )

    return rows


def split_row_into_numbers(row, gap_threshold=None):
    """
    Splits one row into separate numbers using horizontal distance.
    """

    if not row:
        return []

    if gap_threshold is None:
        average_width = np.mean([
            component["w"]
            for component in row
        ])

        gap_threshold = average_width * 1

    numbers = [[row[0]]]

    for previous, current in zip(row, row[1:]):

        previous_right = (
            previous["x"]
            + previous["w"]
        )

        horizontal_gap = (
            current["x"]
            - previous_right
        )

        if horizontal_gap > gap_threshold:
            # Large gap = new number
            numbers.append([current])

        else:
            # Small gap = same number
            numbers[-1].append(current)

    return numbers




def preprocess_digit(img):
    """
    Takes an already-cropped single digit and converts it
    to the same 28x28 MNIST format used by the original code.
    """

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

    # Normalize
    final = np.array(final).astype("float32") / 255.0

    return final


def preprocess_multiple(image: Image):
    """
    Detect rows, split each row into numbers,
    and preprocess each digit.
    """

    # Convert to grayscale
    image = image.convert("L")
    img = np.array(image)

    # 1. Detect individual digit components
    components = find_digit_components(img)

    if not components:
        return []

    # 2. Group components vertically into rows
    rows = group_into_rows(components)

    processed_rows = []

    # 3. Process every row
    for row in rows:

        # Split row into separate numbers
        numbers = split_row_into_numbers(row)

        processed_numbers = []

        for number in numbers:

            processed_digits = []

            for component in number:

                crop = component["image"]

                processed_digit = preprocess_digit(
                    crop
                )

                processed_digits.append(
                    processed_digit
                )

            processed_numbers.append(
                processed_digits
            )

        processed_rows.append(
            processed_numbers
        )

    return processed_rows


# def preprocess_multiple(image: Image):


#     # Grayscale
#     image = image.convert("L")
#     img = np.array(image)

#     # Drawing mask
#     mask = img > 20

#     # Find columns containing drawing
#     occupied_columns = np.any(mask, axis=0)

#     digits = []
#     start = None

#     for x, occupied in enumerate(occupied_columns):

#         if occupied and start is None:
#             # Beginning of a digit
#             start = x

#         elif not occupied and start is not None:
#             # End of a digit
#             end = x

#             crop = img[:, start:end]

#             # Remove empty space above/below
#             rows = np.any(crop > 20, axis=1)

#             if np.any(rows):
#                 y_positions = np.where(rows)[0]

#                 y_min = y_positions[0]
#                 y_max = y_positions[-1]

#                 crop = crop[y_min:y_max + 1, :]

#                 digits.append(preprocess_digit(crop))

#             start = None

#     # Handle digit touching right edge of canvas
#     if start is not None:

#         crop = img[:, start:]

#         rows = np.any(crop > 20, axis=1)

#         if np.any(rows):
#             y_positions = np.where(rows)[0]

#             crop = crop[
#                 y_positions[0]:y_positions[-1] + 1,
#                 :
#             ]

#             digits.append(preprocess_digit(crop))

#     return digits



def preprocess(image: Image):
    """
    Original single-digit preprocessing behavior.
    """

    image = image.convert("L")
    img = np.array(image)

    coords = np.argwhere(img > 20)

    if len(coords) == 0:
        return np.zeros((28, 28), dtype=np.float32)

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    img = img[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    return preprocess_digit(img)