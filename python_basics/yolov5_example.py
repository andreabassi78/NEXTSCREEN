import yolov5
import urllib.request
import cv2
import numpy as np

# Load an image of cells from a URL
url = 'https://raw.githubusercontent.com/andreabassi78/NEXTSCREEN/refs/heads/future/images/blood_cells.jpg'
response = urllib.request.urlopen(url)
img_data = np.asarray(bytearray(response.read()), dtype="uint8")

# Convert the raw image bytes into a proper image (RGB format)
img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)  # OpenCV loads the image in BGR by default
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for YOLOv5

# Load YOLOv5 model
model = yolov5.load('keremberke/yolov5m-blood-cell')

# Set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

# Perform inference
results = model(img, size=620)

# Parse results
predictions = results.pred[0]
boxes = predictions[:, :4]  # x1, y1, x2, y2
scores = predictions[:, 4]
categories = predictions[:, 5]

# Show detection bounding boxes on the image
results.show(labels=True)
results.print()