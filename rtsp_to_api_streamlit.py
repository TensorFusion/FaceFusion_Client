import streamlit as st
import cv2
import time
import requests
import numpy as np

st.title("RTSP to Face Recognition API")

rtsp_url = st.text_input("Enter RTSP URL:", value="rtsp://your_camera_ip/stream1")
send_interval = st.slider("Send every N-th frame", 1, 30, 10)
show_preview = st.checkbox("Show Video Preview", value=True)
start_button = st.button("Start Streaming")

def send_frame_to_api(frame):
    _, img_encoded = cv2.imencode('.jpg', frame)
    response = requests.post(
        "http://localhost:8000/recognize-frame",
        files={"file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}
    )
    return response.json()

if start_button:
    st.write("Connecting to RTSP stream...")
    cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        st.error("Failed to open RTSP stream.")
    else:
        frame_counter = 0
        frame_placeholder = st.empty()
        result_box = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("Stream ended or failed to read frame.")
                break

            frame_counter += 1

            if frame_counter % send_interval == 0:
                with st.spinner("Sending frame to API..."):
                    result = send_frame_to_api(frame)
                    result_box.json(result)

            if show_preview:
                # Resize for display
                preview = cv2.resize(frame, (640, 360))
                frame_placeholder.image(preview, channels="BGR")

            # Add small sleep to prevent locking the browser
            time.sleep(0.03)
