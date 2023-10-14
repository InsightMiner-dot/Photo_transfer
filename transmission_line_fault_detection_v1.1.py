import streamlit as st
from ultralytics import YOLO
import cv2
import os
import glob
import numpy as np
from PIL import Image as PILImage
import base64
import pandas as pd
import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import streamlit as st
from IPython.display import Image, display

# Specify your file paths
image_path = r"D:\Machine_Learning\pythonProject\Dataset\20230925_215716.jpg"
folder_path = r"D:\Machine_Learning\pythonProject\test"
video_path = r"D:\Machine_Learning\pythonProject\test.mp4"
model_weights_path = r"D:\Machine_Learning\pythonProject\best.pt"

# Create a Streamlit app
st.set_page_config(layout="wide")
st.title("YOLO Object Detection App")

# Load the YOLO model
model = YOLO(model_weights_path)

# Create navigation sidebar
page = st.sidebar.selectbox("Choose Detection Type", ["Single Image", "Folder", "Video"])

# Define class names
class_names = ["Bad Insulator", "Good Insulator", "Good Structure", "Rusted Structure"]
# Add a sidebar section to choose the confidence level
# st.sidebar.title("Settings")
# confidence_level = st.sidebar.slider("Confidence Level", 0.0, 1.0, 0.4, 0.05)

if page == "Single Image":
    st.header("Image Detection")
    st.sidebar.title("Settings")
    confidence_level = st.sidebar.slider("Confidence Level", 0.0, 1.0, 0.4, 0.05)
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_image is not None:
        # Get the filename of the uploaded image
        image_name = uploaded_image.name

        # Convert the uploaded image to a format that OpenCV can work with
        image = PILImage.open(uploaded_image)
        image = np.array(image)

        # Calculate the new dimensions to maintain the aspect ratio
        desired_height = 300  # Set your desired height
        aspect_ratio = image.shape[1] / image.shape[0]
        new_width = int(desired_height * aspect_ratio)

        # Resize the image proportionally using PIL
        image = PILImage.fromarray(image)
        image = image.resize((new_width, desired_height), PILImage.ANTIALIAS)
        image = np.array(image)

        st.image(image, caption=f"Uploaded Image: {image_name} (Resized)", use_column_width=True)
        if st.button("Detect Objects"):
            st.write("Detecting objects in the uploaded image...")
            results = model.predict(image, save=True, save_txt=True, conf=confidence_level)
            detected_image_path = results[0].save_dir + "/image0.jpg"

            # Open and display the resized detected image using PIL
            detected_image = PILImage.open(detected_image_path)
            detected_image = detected_image.resize((new_width, desired_height), PILImage.ANTIALIAS)
            st.image(np.array(detected_image), caption="Detected Image (Resized)", use_column_width=True)

            # Now process the detection results and display in the sidebar
            image_data = {}

            # Loop through the results
            for result in results:
                boxes = result.boxes
                probs = result.probs
                cls = boxes.cls.tolist()
                xyxy = boxes.xyxy
                conf = boxes.conf

                for i in range(len(cls)):
                    class_index = cls[i]
                    class_name = class_names[int(class_index)]
                    confidence = round(conf[i].item(), 2)  # Round confidence to 2 decimal places

                    # Increment the count in the image data dictionary
                    if class_name not in image_data:
                        image_data[class_name] = {'count': 0, 'confidence_sum': 0}
                    image_data[class_name]['count'] += 1
                    image_data[class_name]['confidence_sum'] += confidence

            # Create a DataFrame from the detection results
            detection_results = []
            for class_name, data in image_data.items():
                count = data['count']
                confidence_sum = data['confidence_sum']
                detection_results.append([image_name, class_name, count, confidence_sum])

            df = pd.DataFrame(detection_results, columns=["Image Name", "Class", "Count", "Confidence"])

            # Now display the detection results in the sidebar
            st.sidebar.header("Detection Results")
            st.sidebar.dataframe(df)

            # Add a "Download" button for the DataFrame
            csv_filename = f"{image_name}_detection_results.csv"
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            st.sidebar.markdown(f"**Download the DataFrame:**")
            href = f'<a href="data:file/csv;base64,{b64}" download="{csv_filename}">Click here to download</a>'
            st.sidebar.write(href, unsafe_allow_html=True)

### For Folder
elif page == "Folder":

    st.title("Object Detection and Image Preview for Batch")
    # Sidebar for folder selection
    st.sidebar.header("Select a folder")
    st.sidebar.title("Settings")
    # confidence_level = st.sidebar.slider("Confidence Level", 0.0, 1.0, 0.4, 0.05)

    # Folder browse input widget
    folder_path = st.sidebar.text_input("Enter folder path:")

    # Create a list to store the object detection results
    results_list = []
    class_names = ["Bad Insulator", "Good Insulator", "Good Structure", "Rusted Structure"]

    # Initialize an empty list to store detection results as dictionaries
    detection_results = []

    # Process the selected folder and its contents
    if folder_path and os.path.exists(folder_path):
        image_files = glob.glob(os.path.join(folder_path, "*.jpg"))

        if not image_files:
            st.warning("No image files found in the selected folder.")
        else:
            st.title("Image Preview")
            num_columns = 4  # Number of columns in the grid
            num_rows = (len(image_files) + num_columns - 1) // num_columns
            fig, axes = plt.subplots(num_rows, num_columns, figsize=(12, 8))

            for i, image_path in enumerate(image_files):
                row = i // num_columns
                col = i % num_columns
                ax = axes[row, col]
                img = mpimg.imread(image_path)
                ax.imshow(img)
                ax.set_title(os.path.basename(image_path))
                ax.axis("off")

            for i in range(len(image_files), num_rows * num_columns):
                row = i // num_columns
                col = i % num_columns
                fig.delaxes(axes[row, col])

            st.pyplot(fig)

        # "Detect Objects" button to trigger object detection
        if st.button("Detect Objects"):
            st.title("Object Detection Results")
            model_weights_path = r"best.pt"  # Replace with the path to your YOLO weights

            if not os.path.exists(model_weights_path):
                st.warning("YOLO model weights not found. Please provide a valid model_weights_path.")
            else:
                # Create the YOLO model
                model = YOLO(model_weights_path)

                detected_images = []  # Store the detected images
                image_names = []  # Store image names

                for image_path in image_files:
                    results = model(image_path, save=True)
                    image_name = os.path.basename(image_path)
                    class_counts = {class_name: 0 for class_name in class_names}

                    for result in results:
                        boxes = result.boxes
                        cls = boxes.cls.tolist()
                        probs = result.probs
                        xyxy = boxes.xyxy
                        conf = boxes.conf

                        detected_image_path = result.path  # Get the path of the detected image
                        detected_image = PILImage.open(detected_image_path)
                        detected_image = np.array(detected_image)

                        for i in range(len(cls)):  # Iterate over the boxes in each result
                            class_index = cls[i]
                            class_name = class_names[int(class_index)]
                            class_counts[class_name] += 1
                            confidence = round(conf[i].item(), 2)  # Access confidence using 'i'

                            # Get bounding box coordinates
                            x1, y1, x2, y2 = map(int, xyxy[i].tolist())

                            # Increase bounding box thickness and font size
                            thickness = 5  # You can adjust the thickness
                            font_scale = 8  # You can adjust the font scale

                            # Draw bounding box and class label
                            cv2.rectangle(detected_image, (x1, y1), (x2, y2), (0, 0, 255), thickness)
                            cv2.putText(detected_image, f"{class_name}: {confidence}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

                        detected_images.append(detected_image)
                        image_names.append(image_name)

                        # Add detection results to the list as dictionaries
                        for class_name, count in class_counts.items():
                            detection_results.append({
                                "Image Name": image_name,
                                "Class": class_name,
                                "Count": count,
                                "Confidence": confidence
                            })

                num_columns = 3  # Number of columns in the grid
                num_rows = (len(detected_images) + num_columns - 1) // num_columns
                fig, axes = plt.subplots(num_rows, num_columns, figsize=(16, 12))

                for i, (image, image_name) in enumerate(zip(detected_images, image_names)):
                    row = i // num_columns
                    col = i % num_columns
                    ax = axes[row, col]
                    ax.imshow(image)
                    ax.set_title(image_name)
                    ax.axis("off")

                for i in range(len(detected_images), num_rows * num_columns):
                    row = i // num_columns
                    col = i % num_columns
                    fig.delaxes(axes[row, col])

                st.pyplot(fig)

    # Now display the detection results in the sidebar
    st.sidebar.header("Detection Results")

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(detection_results)

    st.sidebar.dataframe(df)

    # Add a "Download" button for the DataFrame
    if not df.empty:
        csv_filename = f"detection_results.csv"
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        st.sidebar.markdown("**Download the DataFrame:**")
        href = f'<a href="data:file/csv;base64,{b64}" download="{csv_filename}">Click here to download</a>'
        st.sidebar.write(href, unsafe_allow_html=True)
### For video file
elif page == "Video":
    # Create a Streamlit app
    st.title("YOLO Object Detection for Video")

    # Upload a video file
    uploaded_video = st.file_uploader("Upload a video", type=["mp4"])

    # Check if a video is uploaded
    if uploaded_video:
        st.write("Video uploaded successfully!")
        st.write("You can now proceed with object detection.")

        # Store the uploaded video
        video_path = "uploaded_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())

        # Button to run object detection
        if st.button("Detect Objects"):
            # Create a YOLO object with the model path
            model = YOLO(r"D:\Machine_Learning\pythonProject\best.pt")

            # Perform object detection
            st.write("Running object detection...")
            detection_result = model.predict(video_path, save=True, save_txt=True)
            st.write("Object detection complete!")