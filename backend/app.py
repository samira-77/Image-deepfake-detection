# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
import io

from model_definitions.EfficientNetLSTM import EfficientNetLSTM

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = EfficientNetLSTM(num_classes=2)
model.load_state_dict(torch.load("model/model.pth", map_location=device))
model.to(device)
model.eval()

# Same transforms as training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_bytes = request.files["image"].read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = float(probs[pred_idx].item() * 83.0)

    # ImageFolder usually maps: {'fake': 0, 'real': 1}
    # So:
    # 0 → FAKE
    # 1 → REAL
    if pred_idx == 0:
        label = "FAKE"
    else:
        label = "REAL"

    return jsonify({
        "result": label,
        "confidence": round(confidence, 2)
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
