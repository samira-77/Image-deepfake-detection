document.getElementById("imageInput").addEventListener("change", function () {
    const file = this.files[0];
    const preview = document.getElementById("previewImage");
    const placeholder = document.getElementById("placeholderText");

    if (file) {
        preview.src = URL.createObjectURL(file);
        preview.style.display = "block";
        placeholder.style.display = "none";
    }
});

function detectDeepfake() {
    const input = document.getElementById("imageInput");
    const loader = document.getElementById("loader");
    const resultBox = document.getElementById("resultBox");
    const resultText = document.getElementById("resultText");
    const confidenceText = document.getElementById("confidenceText");

    if (!input.files.length) {
        alert("Please select an image first!");
        return;
    }

    const formData = new FormData();
    formData.append("image", input.files[0]);

    loader.style.display = "block";
    resultBox.style.display = "none";

    fetch("https://image-deepfake-detection-ssgi.onrender.com/predict", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        loader.style.display = "none";
        resultBox.style.display = "block";

        resultText.textContent = "Prediction: " + data.result;
        confidenceText.textContent = "Confidence: " + data.confidence + "%";

        resultText.style.color = data.result === "FAKE" ? "red" : "lime";
    })
    .catch(err => {
        loader.style.display = "none";
        alert("Error analyzing image!");
    });
}
function resetDetection() {
    const input = document.getElementById("imageInput");
    const preview = document.getElementById("previewImage");
    const placeholder = document.getElementById("placeholderText");
    const resultBox = document.getElementById("resultBox");
    const loader = document.getElementById("loader");

    // Reset input
    input.value = "";

    // Hide preview image
    preview.style.display = "none";
    preview.src = "";

    // Show placeholder
    placeholder.style.display = "block";

    // Hide result box
    resultBox.style.display = "none";

    // Hide loader
    loader.style.display = "none";
}
