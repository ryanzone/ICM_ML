import torch
import gradio as gr
import traceback
from PIL import Image
from model_arch import PneumoniaModel, get_inference_transform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pth"

# 1. Load Model and Transforms
model = PneumoniaModel(2).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
test_transform = get_inference_transform()

def classify_xray(img):
    if img is None:
        return "No image provided"
    try:
        # Preprocessing
        img = img.convert("RGB")
        x = test_transform(img).unsqueeze(0)
        
        # Inference
        x = x.to(DEVICE)
        
        with torch.no_grad():
            out = model(x)
            prob = torch.softmax(out, 1)[0]
            idx = prob.argmax().item()
            
        classes = ['Normal', 'Pneumonia']
        
        return f"{classes[idx]} ({prob[idx]:.4f})"
    except Exception:
        return traceback.format_exc()

# 2. Gradio Interface Setup
# inference.py

# ... (Previous code) ...

# 2. Gradio Interface Setup
# Update the Gradio Interface Setup in inference.py
if __name__ == '__main__':
    gr.Interface(
        fn=classify_xray,
        inputs=gr.Image(type="pil"),
        outputs=gr.Textbox(),
        title="Pneumonia X-ray Classifier",
        description="Upload a chest X-ray image for classification. Powered by PyTorch and ResNet18.",
        # Add sample images here
        examples=["samples/normal_sample.jpeg", "samples/pneumonia_sample.jpeg"]
    ).launch()
    