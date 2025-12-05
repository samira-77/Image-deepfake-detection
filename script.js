// Preview Image
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
    const resultCard = document.getElementById("resultCard");
    const resultText = document.getElementById("resultText");

    if (!input.files.length) {
        alert("Please select an image first!");
        return;
    }

    const formData = new FormData();
    formData.append("image", input.files[0]);

    loader.style.display = "block";
    resultCard.style.display = "none";

    fetch("http://localhost:5000/predict", {
        method: "POST",
        body: formData
    })
        .then(res => res.json())
        .then(data => {
            loader.style.display = "none";
            resultCard.style.display = "block";

            resultText.textContent = "Prediction: " + data.result;
            resultText.style.color = data.result === "FAKE" ? "red" : "lime";
        })
        .catch(err => {
            loader.style.display = "none";
            alert("Error analyzing image!");
        });
}
