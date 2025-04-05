import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from models import CustomResNet18
# App configuration
st.set_page_config(page_title="Shoe-Sandal-Boot-Classifier", layout="wide")

# Load class labels
@st.cache_data
def load_labels():
    with open('class_labels.txt') as f:
        return [line.strip() for line in f.readlines()]

# Load your trained model
@st.cache_resource
def load_model():
    # Initialize your model (adjust based on your architecture)
    model = CustomResNet18(num_classes=len(load_labels()))  
    
    # Load trained weights
    model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# Main app
def main():
    st.title("Shoe-Sandal-Boot-Classifier")
    st.write("Upload an image to classify it using your trained model")
    
    # Sidebar
    st.sidebar.header("Settings")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0, 100, 50) / 100
    
    # File upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            # Load model and labels
            model = load_model()
            labels = load_labels()
            
            # Preprocess and predict
            input_tensor = preprocess_image(image)
            
            with torch.inference_mode():
                output = model(input_tensor)
            
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            # Display results
            st.subheader("Classification Results")
            
            results = []
            for i, prob in enumerate(probabilities):
                if prob > confidence_threshold:
                    results.append((labels[i], prob.item()))
            
            if not results:
                st.warning(f"No predictions above {confidence_threshold*100:.0f}% confidence")
            else:
                results.sort(key=lambda x: x[1], reverse=True)
                for label, prob in results[:5]:  # Show top 5
                    st.progress(prob)
                    st.write(f"**{label}**: {prob*100:.2f}% confidence")

if __name__ == "__main__":
    main()