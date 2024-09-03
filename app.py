import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile

# Load YOLOv9 custom model
# model = YOLO("yolov9c.pt")
model = YOLO("yolov8n.pt") 

# Streamlit UI
st.set_page_config(page_title="Object Detection", page_icon="🖼")
st.title("YOLOv8 Custom Object Detection")

# Option to upload image, video, or open camera
upload_type = st.sidebar.selectbox("Choose Input Type", ["Image", "Video", "Camera"])

if upload_type == "Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        #st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert image to array
        image = np.array(image)

        # Run YOLOv9 on the image
        results = model(image)

        # Draw bounding boxes on the image
        annotated_image = results[0].plot()  # This includes the bounding boxes

        st.image(annotated_image, caption="Processed Image with Detections", use_column_width=True)

elif upload_type == "Video":
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_file is not None:
        # Save the uploaded file temporarily to disk
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLOv8 on the frame
            results = model(frame)

            # Draw bounding boxes on the frame
            annotated_frame = results[0].plot()

            # Display the frame with detections
            stframe.image(annotated_frame, channels="BGR")

        cap.release()
        tfile.close()

elif upload_type == "Camera":
    st.write("Opening webcam...")
    cap = cv2.VideoCapture(0)  # Open the default camera

    stframe = st.empty()
    stop_button = st.button('Stop', key='stop_button')  # Assign a unique key

    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to access webcam.")
            break

        # Run YOLOv9 on the frame
        results = model(frame)

        # Draw bounding boxes on the frame
        annotated_frame = results[0].plot()

        # Display the frame with detections
        stframe.image(annotated_frame, channels="BGR")

        if stop_button:
            break

    cap.release()
    st.write("Webcam stopped.")
