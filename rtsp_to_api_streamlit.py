import streamlit as st
import requests
import numpy as np
import cv2

st.title("Webcam Face Recognition Demo")

st.markdown("Use your webcam to capture an image and send it to the Face Recognition API.")

# Capture a frame from webcam
img_file = st.camera_input("Take a photo")

if img_file is not None:
    st.image(img_file, caption="📸 Captured Frame", use_container_width=True)

    with st.spinner("Processing image..."):
        # Read the image as bytes
        img_bytes = img_file.getvalue()

        # Send to FastAPI backend
        try:
            response = requests.post(
                "http://zahangir.pythonanywhere.com/face-reg",
                files={"file": ("frame.jpg", img_bytes, "image/jpeg")}
            )
            if response.ok:
                st.success("✅ Face Recognition Result:")
                st.json(response.json())
            else:
                st.error(f"API Error: {response.text}")
        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")
