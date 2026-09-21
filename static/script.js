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

        document.getElementById("prediction").textContent =
            `Prediction: ${result.digit}`;

        document.getElementById("confidence").textContent =
            `Confidence: ${(result.confidence * 100).toFixed(1)}%`;

    }, "image/png");
}