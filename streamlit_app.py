import streamlit as st
import torch
import traceback
from PIL import Image
from torchvision import transforms
from model_arch import PneumoniaModel, get_test_transform

# Set page config
st.set_page_config(
    page_title="Chest X-ray Classifier",
    layout="centered"
)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = PneumoniaModel(2)
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Get transform
test_transform = get_test_transform()

# Title and description
st.title("🫁 Chest X-ray Classifier")
st.markdown("""
This application classifies chest X-ray images as **Normal** or **Pneumonia**.
Upload a chest X-ray image to get started.
""")

# Display device info in sidebar
st.sidebar.header("System Information")
st.sidebar.info(f"**Device:** {device}")
if torch.cuda.is_available():
    st.sidebar.info(f"**GPU:** {torch.cuda.get_device_name(0)}")

# Load the model
model = load_model()

if model is None:
    st.error("Failed to load model. Please ensure 'best_model.pth' exists in the same directory.")
else:
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image...", 
        type=["jpg", "jpeg", "png"],
        help="Upload a chest X-ray image in JPG, JPEG, or PNG format"
    )
    
    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file).convert("RGB")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="Uploaded X-ray", use_container_width=True)
            
            with col2:
                # Make prediction
                with st.spinner("Analyzing..."):
                    # Transform image
                    x = test_transform(image).unsqueeze(0)
                    x = x.to(device)
                    
                    # Get prediction
                    with torch.no_grad():
                        out = model(x)
                        prob = torch.softmax(out, 1)[0]
                        idx = prob.argmax().item()
                    
                    classes = ['Normal', 'Pneumonia']
                    prediction = classes[idx]
                    confidence = prob[idx].item()
                    
                    # Display results
                    st.markdown("### Results")
                    
                    if prediction == "Pneumonia":
                        st.error(f"**Prediction:** {prediction}")
                    else:
                        st.success(f"**Prediction:** {prediction}")
                    
                    st.metric("Confidence", f"{confidence*100:.2f}%")
                    
                    # Show probabilities for both classes
                    st.markdown("#### Class Probabilities")
                    for i, class_name in enumerate(classes):
                        st.progress(prob[i].item(), text=f"{class_name}: {prob[i].item()*100:.2f}%")
                    
                    # Disclaimer
                    st.warning("⚠️ **Disclaimer:** This is a demonstration model and should not be used for actual medical diagnosis. Please consult healthcare professionals for medical advice.")
        
        except Exception as e:
            st.error("Error processing image")
            with st.expander("Show error details"):
                st.code(traceback.format_exc())
    
    else:
        st.info("👆 Please upload a chest X-ray image to begin classification.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit 🎈 | Model: ResNet18</p>
</div>
""", unsafe_allow_html=True)