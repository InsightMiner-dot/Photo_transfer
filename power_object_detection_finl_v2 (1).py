# -*- coding: utf-8 -*-
"""Power_object_detection_finl_v2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18m2T8Gp9jcvXt63fwH4TAD-pFjfMFmft
"""

!pip install ultralytics

!pip install ipywidgets

!jupyter nbextension enable --py widgetsnbextension

!pip install opencv-python

# Import necessary library
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import glob
from IPython.display import Image, display
import glob
import os
import csv
import pandas as pd
import os
import csv
import glob
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Mention the sourc file
image_path = r"/content/drive/MyDrive/CustomObject.v1i.yolov8/valid/images/20231009_180141_jpg.rf.fc1070c8e617b729cac4808ff53f6291.jpg"
folder_path = r"/content/drive/MyDrive/CustomObject.v1i.yolov8/valid/images"
video_path = r"/content/drive/MyDrive/CustomObject.v1i.yolov8/test.mp4"

model_weights_path=r"/content/drive/MyDrive/CustomObject.v1i.yolov8/Weights/best.pt"

# Import pretrained custom model
model= YOLO(r"/content/drive/MyDrive/CustomObject.v1i.yolov8/Weights/best.pt")

# result = model(image_path, save = True)
# class_names = ["Bad Insulator "]

infer = YOLO(r"/content/drive/MyDrive/CustomObject.v1i.yolov8/Weights/best.pt")

# Load the image using cv2
image = cv2.imread(image_path)

# Convert from BGR to RGB (cv2 loads images in BGR format)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image using matplotlib
plt.figure(figsize=(3, 8))
plt.imshow(image_rgb)
plt.axis('off')
plt.show()

# from google.colab.patches import cv2_imshow
# import cv2

# # Load the image using cv2
# image = cv2.imread(image_path)

# # Display the image using cv2_imshow
# cv2_imshow(image)

### Detection for single image
infer_single_img =infer.predict(image_path,save = True, save_txt=True)

for image_path in glob.glob(f'/content/runs/detect/predict/*.jpg')[:3]:
  display(Image(filename=image_path, height=300))
  print("\n")

results = model(image_path, save= True)
class_names = ["Bad Insulator", "Good Insulator", "Good Structure", "Rusted Structure"]

for result in results:
  boxes = result.boxes
  probs = result.probs
  cls = boxes.cls.tolist()
  xyxy = boxes.xyxy
  xywh= boxes.xywh
  conf = boxes.conf
  print(cls)
  for class_index in cls:
    class_name = class_names[int(class_index)]
    print("Class", class_name)

# # Create a dictionary to store the count of objects in each class
# class_count = {class_name: 0 for class_name in class_names}

# # Loop through the results
# for result in results:
#     boxes = result.boxes
#     cls = boxes.cls.tolist()

#     # Increment the count for each detected class
#     for class_index in cls:
#         class_name = class_names[int(class_index)]
#         class_count[class_name] += 1

# # Print the counts for each class
# for class_name, count in class_count.items():
#     print(f"Class: {class_name}, Count: {count}")

import pandas as pd

# Create an empty dictionary to store the count of objects in each class
class_count = {class_name: 0 for class_name in class_names}

# Create a list to store detection results
detection_results = []

# Initialize a dictionary to track class counts and confidences for each image
image_data = {}

# Loop through the results
for result in results:
    boxes = result.boxes
    probs = result.probs
    cls = boxes.cls.tolist()
    xyxy = boxes.xyxy
    conf = boxes.conf

    # Create a unique identifier for the current image (you can customize this)
    image_identifier = f"img{len(detection_results) + 1}"

    if image_identifier not in image_data:
        image_data[image_identifier] = {class_name: {'count': 0, 'confidence_sum': 0} for class_name in class_names}

    for i in range(len(cls)):
        class_index = cls[i]
        class_name = class_names[int(class_index)]
        confidence = conf[i]

        # Increment the count and add the confidence to the image data
        image_data[image_identifier][class_name]['count'] += 1
        image_data[image_identifier][class_name]['confidence_sum'] = confidence

# Process the image data to generate the final detection results
for image_identifier, class_data in image_data.items():
    for class_name, data in class_data.items():
        count = data['count']
        confidence_sum = data['confidence_sum']
        detection_results.append([image_identifier, class_name, count, confidence_sum])

# Create a DataFrame from the detection results
df = pd.DataFrame(detection_results, columns=["Image", "Class", "Count", "Confidence"])

# Display the DataFrame
print(df)

### Detection for folder
infer_batch = infer.predict(folder_path,save = True, save_txt= True)

# class_names = ["Bad Insulator", "Good Insulator", "Good Structure", "Rusted Structure"]
# folder_path =r"/content/drive/MyDrive/CustomObject.v1i.yolov8/valid/images"  # Replace with the path to your image folder

# # Create a list to store the results
# results_list = []

# # Use glob to get a list of image files in the folder
# image_files = glob.glob(os.path.join(folder_path, "*.jpg"))

# # Create the YOLO model
# model = YOLO(r"/content/drive/MyDrive/CustomObject.v1i.yolov8/Weights/best.pt")  # Replace with the path to your YOLO weights

# for image_path in image_files:
#     results = model(image_path, save=True)
#     for result in results:
#         boxes = result.boxes
#         cls = boxes.cls.tolist()
#         image_name = os.path.basename(image_path)
#         for class_index in cls:
#             class_name = class_names[int(class_index)]
#             results_list.append([image_name, class_name])

# # Save the results to a CSV file
# output_file = "detection_results.csv"
# with open(output_file, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(["Image Name", "Class"])
#     writer.writerows(results_list)

# print(f"Results saved to {output_file}")

# import glob
# import os
# import csv

# class_names = ["Bad Insulator", "Good Insulator", "Good Structure", "Rusted Structure"]
# folder_path =r"/content/drive/MyDrive/CustomObject.v1i.yolov8/valid/images"  # Replace with the path to your image folder

# # Create a list to store the results
# results_list = []

# # Use glob to get a list of image files in the folder
# image_files = glob.glob(os.path.join(folder_path, "*.jpg"))

# # Create the YOLO model
# model = YOLO(r"/content/drive/MyDrive/CustomObject.v1i.yolov8/Weights/best.pt")  # Replace with the path to your YOLO weights

# for image_path in image_files:
#     results = model(image_path, save=True)
#     image_name = os.path.basename(image_path)
#     class_counts = {class_name: 0 for class_name in class_names}

#     for result in results:
#         boxes = result.boxes
#         cls = boxes.cls.tolist()
#         for class_index in cls:
#             class_name = class_names[int(class_index)]
#             class_counts[class_name] += 1

#     for class_name, count in class_counts.items():
#         results_list.append([image_name, class_name, count])

# # Save the results to a CSV file
# output_file = "detection_results.csv"
# with open(output_file, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(["Image Name", "Class", "Count"])
#     writer.writerows(results_list)

# print(f"Results saved to {output_file}")

class_names = ["Bad Insulator", "Good Insulator", "Good Structure", "Rusted Structure"]
# folder_path = r"/content/drive/MyDrive/CustomObject.v1i.yolov8/valid/images" # Replace with the path to your image folder

# Create a list to store the results
results_list = []

# Use glob to get a list of image files in the folder
image_files = glob.glob(os.path.join(folder_path, "*.jpg"))

# Create the YOLO model
model = YOLO(model_weights_path)  # Replace with the path to your YOLO weights

for image_path in image_files:
    results = model(image_path, save=True)
    image_name = os.path.basename(image_path)
    class_counts = {class_name: 0 for class_name in class_names}

    for result in results:
        boxes = result.boxes
        cls = boxes.cls.tolist()
        for class_index in cls:
            class_name = class_names[int(class_index)]
            class_counts[class_name] += 1

    for class_name, count in class_counts.items():
        results_list.append([image_name, class_name, count])

# Create a DataFrame from the results_list
df = pd.DataFrame(results_list, columns=["Image Name", "Class", "Count"])

# Display the DataFrame
print(df)

# Save the results to a CSV file
output_file = "detection_results.csv"
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Image Name", "Class", "Count"])
    writer.writerows(results_list)

print(f"Results saved to {output_file}")

# Display the images in a grid
num_columns = 4
num_rows = (len(image_files) + num_columns - 1) // num_columns

fig, axes = plt.subplots(num_rows, num_columns, figsize=(12, 8))

for i, image_path in enumerate(image_files):
    row = i // num_columns
    col = i % num_columns
    ax = axes[row, col]
    img = mpimg.imread(image_path)
    ax.imshow(img)
    ax.axis("off")

for i in range(len(image_files), num_rows * num_columns):
    row = i // num_columns
    col = i % num_columns
    fig.delaxes(axes[row, col])

plt.tight_layout()
plt.show()

# import pandas as pd

# # ... (previous code for object detection)

# # Create a DataFrame from the results_list
# df = pd.DataFrame(results_list, columns=["Image Name", "Class", "Count"])

# # Create a pivot table
# pivot_table = df.pivot_table(index='Image Name', columns='Class', values='Count', fill_value=0)

# # Display the pivot table
# print(pivot_table)

import pandas as pd

# ... (previous code for object detection)

# Create a DataFrame from the results_list
df = pd.DataFrame(results_list, columns=["Image Name", "Class", "Count"])

# Specify the class name you want to filter
class_to_filter = "Bad Insulator"  # Replace with the class name you want to filter

# Filter the DataFrame for the specified class
filtered_df = df[df['Class'] == class_to_filter]

# Display the filtered DataFrame
print(filtered_df)



import pandas as pd
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt

# ... (previous code for object detection, filtering, and dropdown)

# Define a function to create and display the bar chart
def create_bar_chart(filtered_df):
    filtered_df['Count'] = filtered_df['Count'].astype(int)  # Convert Count column to integers
    plt.figure(figsize=(10, 6))
    plt.bar(filtered_df['Image Name'], filtered_df['Count'])
    plt.xlabel('Image Name')
    plt.ylabel('Count')
    plt.title('Count of Each Image for Selected Class')
    plt.xticks(rotation=90)  # Rotate x-axis labels for readability
    plt.show()

import pandas as pd
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt

# ... (previous code for object detection, filtering, and dropdown)

# Define a function to create and display the bar chart
def create_bar_chart(filtered_df):
    filtered_df['Count'] = filtered_df['Count'].astype(int)  # Convert Count column to integers
    plt.figure(figsize=(10, 6))
    plt.bar(filtered_df['Image Name'], filtered_df['Count'])
    plt.xlabel('Image Name')
    plt.ylabel('Count')
    plt.title('Count of Each Image for Selected Class')
    plt.xticks(rotation=90)  # Rotate x-axis labels for readability
    plt.show()

# Define an event handler for the dropdown widget
def on_dropdown_change(change):
    with output:
        output.clear_output()
        selected_class = change.new
        if selected_class == 'All':
            filtered = df
        else:
            filtered = df[df['Class'] == selected_class]
        display(filtered)
        create_bar_chart(filtered)

class_dropdown.observe(on_dropdown_change, names='value')

# Display the dropdown and the output
display(class_dropdown)
display(output)

import cv2
from google.colab.patches import cv2_imshow

# Define the path to your video file
video_path = '/content/drive/MyDrive/CustomObject.v1i.yolov8/test.mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
else:
    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # Break the loop if the video has ended
        if not ret:
            break

        # Display the frame
        cv2_imshow(frame)

        # Wait for a short duration (e.g., 30 milliseconds) and listen for a key press
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

# Release the video file and close the OpenCV window
cap.release()
cv2.destroyAllWindows()



20 # Detection for video
infer.predict(video_path,save = True, save_txt = True)

res_vdi= model(video_path,save = True, save_txt = True)
class_names = ["Bad Insulator", "Good Insulator", "Good Structure", "Rusted Structure"]
for result in results:
  boxes = result.boxes
  probs = result.probs
  cls = boxes.cls.tolist()
  xyxy = boxes.xyxy
  xywh= boxes.xywh
  conf = boxes.conf
  print(cls)
  for class_index in cls:
    class_name = class_names[int(class_index)]
    print("Class", class_name)



# import streamlit as st
# import os
# import csv
# import glob
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# from ultralytics import YOLO

# # Function to run YOLO object detection on a video
# def run_yolo_detection_on_video(video_path, weights_path, class_names):
#     model = YOLO(weights_path)
#     results = model(video_path, save=True)

#     # Create an empty dictionary to store the count of objects in each class
#     class_count = {class_name: 0 for class_name in class_names}

#     for result in results:
#         boxes = result.boxes
#         cls = boxes.cls.tolist()
#         for class_index in cls:
#             class_name = class_names[int(class_index)]
#             class_count[class_name] += 1

#     return class_count

# # Streamlit app
# st.title("YOLO Object Detection Dashboard")

# # Sidebar for file upload and weights selection
# st.sidebar.header("Navigation")
# page = st.sidebar.radio("Go to:", ("Run Detection", "Plot Graphs", "Detect Video"))

# if page == "Run Detection":
#     st.sidebar.header("Upload Image")
#     uploaded_image = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

#     st.sidebar.header("Select Weights")
#     weights_path = st.sidebar.selectbox("Choose YOLO weights", ["path/to/your/weights.pt"])

#     if st.sidebar.button("Run Detection"):
#         if uploaded_image is not None:
#             st.subheader("Uploaded Image")
#             st.image(uploaded_image, use_column_width=True)

#             # Get class names (modify as needed)
#             class_names = ["Bad Insulator", "Good Insulator", "Good Structure", "Rusted Structure"]

#             # Run YOLO detection
#             detection_results, yolo_results = run_yolo_detection(uploaded_image, weights_path, class_names)

#             st.subheader("Object Detection Results")
#             st.write("Class Name, Count")
#             for class_name, count in detection_results.items():
#                 st.write(f"{class_name}, {count}")

# elif page == "Plot Graphs":
#     st.header("Graphs")

#     # Add code here to plot graphs, such as the count of images and the count of objects by class.
#     # Save the graphs to the output folder.

#     # Example: Save a sample graph
#     fig, ax = plt.subplots()
#     ax.plot([1, 2, 3], [4, 5, 6])
#     ax.set_xlabel("X-axis")
#     ax.set_ylabel("Y-axis")
#     ax.set_title("Sample Graph")
#     fig.savefig(f"{output_folder}/sample_graph.png")

# elif page == "Detect Video":
#     st.sidebar.header("Select Video")
#     video_file = st.sidebar.file_uploader("Choose a video", type=["mp4"])

#     if video_file is not None:
#         st.subheader("Uploaded Video")
#         st.video(video_file)

#         # Get class names (modify as needed)
#         class_names = ["Bad Insulator", "Good Insulator", "Good Structure", "Rusted Structure"]

#         # Run YOLO detection on the video
#         detection_results = run_yolo_detection_on_video(video_file, weights_path, class_names)

#         st.subheader("Video Object Detection Results")
#         st.write("Class Name, Count")
#         for class_name, count in detection_results.items():
#             st.write(f"{class_name}, {count}")

# # Display the images in a grid
# st.sidebar.header("Images in Batch")
# image_files = glob.glob("path/to/your/images_folder/*.jpg")

# num_columns = 4
# num_rows = (len(image_files) + num_columns - 1) // num_columns

# fig, axes = plt.subplots(num_rows, num_columns, figsize=(12, 8))

# for i, image_path in enumerate(image_files):
#     row = i // num_columns
#     col = i % num_columns
#     ax = axes[row, col]
#     img = mpimg.imread(image_path)
#     ax.imshow(img)
#     ax.axis("off")

# for i in range(len(image_files), num_rows * num_columns):
#     row = i // num_columns
#     col = i % num_columns
#     fig.delaxes(axes[row, col])

# plt.tight_layout()
# st.pyplot(fig)

# # Save all results to the output folder
# st.sidebar.header("Save All Results")
# if st.sidebar.button("Save All"):
#     # Move detection result to the output folder
#     os.rename("detection_results/detection_result.jpg", f"{output_folder}/detection_result.jpg")

#     st.sidebar.text("All results saved to the output folder.")



