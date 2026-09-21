const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let drawing = false;

// MNIST uses black background
ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);

ctx.strokeStyle = "white";
ctx.lineWidth = 25;
ctx.lineCap = "round";

canvas.addEventListener("mousedown", () => {
    drawing = true;
    ctx.beginPath();
});

canvas.addEventListener("mouseup", () => {
    drawing = false;
    predict();
});

canvas.addEventListener("mouseleave", () => {
    drawing = false;
});

canvas.addEventListener("mousemove", (event) => {
    if (!drawing) return;

    const rect = canvas.getBoundingClientRect();

    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    ctx.lineTo(x, y);
    ctx.stroke();
});


document.getElementById("clear").addEventListener("click", () => {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    document.getElementById("prediction").textContent =
        "Prediction: -";

    document.getElementById("confidence").textContent = "";

    const preview = document.getElementById("processed-preview");
    const container = document.getElementById("processed-container");
    if (preview) preview.src = "";
    if (container) container.style.display = "none";
});

async function predict() {

    canvas.toBlob(async (blob) => {

        const formData = new FormData();

        formData.append(
            "file",
            blob,
            "drawing.png"
        );

        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const result = await response.json();

        // Main combined prediction
        const mainPrediction = result.rows
            .map(row => row.map(number => number.number).join(" "))
            .join("\n");

        document.getElementById("prediction").textContent =
            `Prediction:\n${mainPrediction}`;

        document.getElementById("confidence").textContent =
            `Confidence: ${(result.confidence * 100).toFixed(1)}%`;


        // Individual digit predictions
        const processedImages =
            document.getElementById("processedImages");

        const container =
            document.getElementById("processed-container");

        processedImages.innerHTML = "";

        result.processed_images.forEach((imageUrl, index) => {

            const prediction = result.digits[index];

            const item = document.createElement("div");

            const img = document.createElement("img");
            img.src = imageUrl;
            img.width = 112;
            img.height = 112;
            img.style.imageRendering = "pixelated";

            const label = document.createElement("p");

            label.textContent =
                `Prediction: ${prediction.digit} | Confidence: ${(prediction.confidence * 100).toFixed(1)}%`;

            item.appendChild(img);
            item.appendChild(label);

            processedImages.appendChild(item);
        });

        // Show results
        if (result.processed_images.length > 0) {
            container.style.display = "block";
        }

    }, "image/png");
}
