from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
from model import CNN  # Make sure this points correctly

app = Flask(__name__)
CORS(app)

# Initialize and load weights
model = CNN()
model.load_state_dict(torch.load("fashion_mnist_cnn.pth", map_location=torch.device("cpu")))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    image_file = request.files["file"]
    image_bytes = image_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        label = labels[predicted.item()]

    return jsonify({"prediction": label})

if __name__ == "__main__":
    app.run(debug=True)
