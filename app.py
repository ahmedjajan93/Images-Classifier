import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import json
import urllib.request
from torchvision import  models

# App configuration
st.set_page_config(page_title="My Images Classifier", layout="wide")

# Build Model
@st.cache_resource
def load_efficientnet():
    model = models.efficientnet_b7(pretrained=True)
    model.eval()
    return model

model = load_efficientnet()
 

# Image preprocessing
@st.cache_data(ttl=3600)
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # Standard ImageNet values
            std=[0.229, 0.224, 0.225]
        )
        ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

# Get class labels from ImageNet
LABELS_URL = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
with urllib.request.urlopen(LABELS_URL) as url:
    class_idx = json.load(url)
    idx2label = {int(key): value[1] for key, value in class_idx.items()}

# Predict
def predict(image_tensor):
    with torch.inference_mode():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

 # Supported image extensions (add more if needed)
SUPPORTED_EXTENSIONS = ["jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp", "jfif"]

# Main app
def main():

    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://github.com/ahmedjajan93/Images-Classifier/main/background.PNG");
             background-size:cover;
             background-position:center;
             background-repeat: no-repeat;
             background-attachment: fiexed;
         }}
         </style>
         """,
           unsafe_allow_html=True
     )

    st.title("My Images Classifier")
    st.write("Upload an image to classify it using your trained model")
    
    # Sidebar
    st.sidebar.header("Settings")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 20, 100, 50) / 100
    
    # File upload
   
    uploaded_file = st.file_uploader( "Choose an image...",
                                      type=None,  # Allow all files
                                      accept_multiple_files=False, 
                                      help=f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}" )
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
           
             # Display image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:

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
                    results.append((idx2label[i], prob.item()))
            
            if not results:
                st.warning(f"No predictions above {confidence_threshold*100:.0f}% confidence")
            else:
                results.sort(key=lambda x: x[1], reverse=True)
                for label, prob in results[:5]:  # Show top 5
                    st.progress(prob)
                    st.write(f"**{label}**: {prob*100:.2f}% confidence")
   

if __name__ == "__main__":
    main()