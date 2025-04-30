import streamlit as st
import os
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

# Load ONNX model
session = ort.InferenceSession("best.onnx")

# Function to preprocess image for ONNX model
def preprocess_image(image):
    img = cv2.resize(image, (224, 224))  # Adjust if your model expects a different input size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Predict and annotate
def predict_and_display(image_path):
    original = cv2.imread(image_path)
    input_tensor = preprocess_image(original)

    # ONNX inference
    outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
    prediction = np.argmax(outputs[0])

    # Annotate image
    annotated = original.copy()
    label = f"Predicted: {prediction}"
    cv2.putText(annotated, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)

    # Resize to 256x256
    resized_output = cv2.resize(annotated, (256, 256))
    return resized_output

# Streamlit app
st.title("ONNX Image Classifier")
image_folder = "images"

image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
selected_image = st.selectbox("Choose an image", image_files)

if selected_image:
    path = os.path.join(image_folder, selected_image)
    output_img = predict_and_display(path)
    
    # Convert to RGB PIL for Streamlit
    output_pil = Image.fromarray(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    st.image(output_pil, use_column_width=False)
