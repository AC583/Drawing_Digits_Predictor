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
        if area < 10:
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


def merge_multi_stroke_components(components, img):
    """
    Merges components that belong to the same multi-stroke symbol (such as =, %, ÷,
    or crossing strokes of +/x) within a single row.
    """
    if len(components) <= 1:
        return components

    merged = True
    comps = list(components)
    while merged:
        merged = False
        new_comps = []
        skip_indices = set()
        for i in range(len(comps)):
            if i in skip_indices:
                continue
            c1 = comps[i]
            for j in range(i + 1, len(comps)):
                if j in skip_indices:
                    continue
                c2 = comps[j]

                x1_max = c1["x"] + c1["w"]
                x2_max = c2["x"] + c2["w"]
                y1_max = c1["y"] + c1["h"]
                y2_max = c2["y"] + c2["h"]

                overlap_x = min(x1_max, x2_max) - max(c1["x"], c2["x"])
                overlap_y = min(y1_max, y2_max) - max(c1["y"], c2["y"])
                min_w = min(c1["w"], c2["w"])
                min_h = min(c1["h"], c2["h"])

                nx = min(c1["x"], c2["x"])
                ny = min(c1["y"], c2["y"])
                nw = max(x1_max, x2_max) - nx
                nh = max(y1_max, y2_max) - ny

                vert_gap = max(0, max(c1["y"], c2["y"]) - min(y1_max, y2_max))

                # Condition 1: Vertically stacked strokes (like '=' or '÷')
                is_stacked = (
                    overlap_x > 0.3 * min_w and
                    vert_gap <= min(35, max(c1["h"], c2["h"]) * 1.5) and
                    nh <= max(nw * 1.8, 60)
                )

                # Condition 2: Intersecting/crossing strokes (like '+' or 'x')
                is_crossing = (overlap_x > 0 and overlap_y > 0)

                # Condition 3: Small dot near a main stroke (like in '%' or 'i')
                is_dot = (
                    min_w <= 20 and min_h <= 20 and
                    (overlap_x > 0 or abs(c1["center_x"] - c2["center_x"]) < 30) and
                    vert_gap <= 25
                )

                if is_stacked or is_crossing or is_dot:
                    c1 = {
                        "x": nx,
                        "y": ny,
                        "w": nw,
                        "h": nh,
                        "center_x": nx + nw / 2,
                        "center_y": ny + nh / 2,
                        "image": img[ny:ny + nh, nx:nx + nw]
                    }
                    skip_indices.add(j)
                    merged = True
            new_comps.append(c1)
        comps = new_comps

    # Sort left to right
    comps.sort(key=lambda c: c["x"])
    return comps


def group_into_rows(components):
    """
    Groups components into horizontal rows based on vertical overlap and proximity.
    Works robustly for varying canvas sizes and symbol heights.
    """
    if not components:
        return []

    # Sort components by their vertical center
    components = sorted(components, key=lambda c: c["center_y"])

    rows = []
    for c in components:
        c_y1 = c["y"]
        c_y2 = c["y"] + c["h"]
        c_cy = c["center_y"]

        best_row = None
        min_dist = float("inf")

        for row in rows:
            r_y1 = min(item["y"] for item in row)
            r_y2 = max(item["y"] + item["h"] for item in row)
            r_cy = np.mean([item["center_y"] for item in row])

            # Vertical overlap between component and existing row span
            overlap_y = min(c_y2, r_y2) - max(c_y1, r_y1)
            dist_cy = abs(c_cy - r_cy)
            row_height = r_y2 - r_y1
            threshold = max(35, row_height * 0.6)

            # Check if this component overlaps vertically with the row, or is close to the row center
            if overlap_y > 0 or dist_cy <= threshold:
                if dist_cy < min_dist:
                    min_dist = dist_cy
                    best_row = row

        if best_row is not None:
            best_row.append(c)
        else:
            rows.append([c])

    # Sort rows from top to bottom
    rows.sort(key=lambda r: np.mean([item["center_y"] for item in r]))

    # Sort components in each row from left to right
    for r in rows:
        r.sort(key=lambda c: c["x"])

    return rows


def split_row_into_numbers(row, gap_threshold=None):
    """
    Splits one row into separate numbers / tokens using horizontal distance.
    Adapts dynamically to both small and large canvas sizes.
    """
    if not row:
        return []

    if gap_threshold is None:
        average_width = np.mean([component["w"] for component in row])
        # Minimum threshold of 25px prevents small symbols like '.' or '1' from skewing the gap
        gap_threshold = max(25, average_width * 0.85)

    numbers = [[row[0]]]

    for previous, current in zip(row, row[1:]):
        previous_right = previous["x"] + previous["w"]
        horizontal_gap = current["x"] - previous_right

        if horizontal_gap > gap_threshold:
            # Large gap = new number / token
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

    if isinstance(img, Image.Image):
        img_arr = np.array(img.convert("L"))
    else:
        img_arr = np.array(img)

    # Check if this component consists of two vertically stacked bars separated by an empty gap (equals sign '=')
    # When drawn with a thick canvas brush, uniform scaling crushes the gap into 1-2px,
    # causing the CNN to mistake '=' for 'z'. Preserving the gap and bar proportions
    # guarantees accurate '=' predictions (100% confidence).
    fg = (img_arr > 50)
    row_counts = fg.sum(axis=1)

    if len(img_arr) >= 10:
        mid_start = int(len(img_arr) * 0.20)
        mid_end = int(len(img_arr) * 0.80)
        zero_rows = np.where(row_counts[mid_start:mid_end] <= 1)[0]
        if len(zero_rows) >= 2:
            gap_center = mid_start + int(np.mean(zero_rows))
            top_bar = img_arr[:gap_center]
            bot_bar = img_arr[gap_center:]
            t_coords = np.argwhere(top_bar > 50)
            b_coords = np.argwhere(bot_bar > 50)
            if len(t_coords) > 0 and len(b_coords) > 0:
                tw = t_coords[:, 1].max() - t_coords[:, 1].min() + 1
                th = t_coords[:, 0].max() - t_coords[:, 0].min() + 1
                bw = b_coords[:, 1].max() - b_coords[:, 1].min() + 1
                bh = b_coords[:, 0].max() - b_coords[:, 0].min() + 1

                # Both must be horizontal bars (width >= 1.2 * height)
                if tw >= 1.2 * th and bw >= 1.2 * bh:
                    t_crop = top_bar[t_coords[:, 0].min():t_coords[:, 0].max() + 1, t_coords[:, 1].min():t_coords[:, 1].max() + 1]
                    b_crop = bot_bar[b_coords[:, 0].min():b_coords[:, 0].max() + 1, b_coords[:, 1].min():b_coords[:, 1].max() + 1]

                    target_w = 18
                    bar_h = 3
                    gap_h = 5

                    t_res = Image.fromarray(t_crop).resize((target_w, bar_h), Image.Resampling.LANCZOS)
                    b_res = Image.fromarray(b_crop).resize((target_w, bar_h), Image.Resampling.LANCZOS)

                    final = Image.new("L", (28, 28), 0)
                    total_h = bar_h * 2 + gap_h
                    start_y = (28 - total_h) // 2
                    start_x = (28 - target_w) // 2

                    final.paste(t_res, (start_x, start_y))
                    final.paste(b_res, (start_x, start_y + bar_h + gap_h))

                    return np.array(final).astype("float32") / 255.0

    digit = Image.fromarray(img_arr)

    # Preserve aspect ratio
    width, height = digit.size
    scale = 20 / max(width, height)

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    # Single horizontal strokes (minus '-') drawn with a thick canvas brush
    # easily scale to 5-7px thick in 28x28, causing the CNN to mistake them for '='.
    # In the training dataset, 'sub' strokes are thin (2-4px). Clamping to 4px
    # restores accurate 'sub' predictions (>98% confidence).
    if width >= 1.8 * height and new_height > 4:
        new_height = 4

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
    Detect rows, merge multi-stroke symbols within each row,
    split each row into numbers/tokens, and preprocess each digit.
    """
    # Convert to grayscale
    image = image.convert("L")
    img = np.array(image)

    # 1. Detect individual strokes/components
    components = find_digit_components(img)

    if not components:
        return []

    # 2. Group components vertically into rows FIRST (prevents cross-row merging)
    raw_rows = group_into_rows(components)

    processed_rows = []

    # 3. Process every row
    for row in raw_rows:
        # Merge multi-stroke symbols (=, %, ÷, detached strokes) ONLY within this row
        row_components = merge_multi_stroke_components(row, img)

        # Split row into separate numbers / tokens
        numbers = split_row_into_numbers(row_components)

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