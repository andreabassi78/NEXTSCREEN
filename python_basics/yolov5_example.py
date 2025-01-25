import yolov5
import os
import urllib.request

# load model
model = yolov5.load('keremberke/yolov5m-blood-cell')
  
# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

# Load an image from a URL
url = 'https://raw.githubusercontent.com/andreabassi78/NEXTSCREEN/refs/heads/future/python_basics/blood_cells.jpg'
img = urllib.request.urlopen(url)

# perform inference
results = model(img, size=620)

# inference with test time augmentation
results = model(img, augment=True)

# parse results
predictions = results.pred[0]
boxes = predictions[:, :4] # x1, y1, x2, y2
scores = predictions[:, 4]
categories = predictions[:, 5]

# show detection bounding boxes on image
results.show(labels=True)

# save results into "results/" folder
#results.save(save_dir=os.path.join(folder,'saved','results'))

# Results

results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
#results.xyxy[0]  # im predictions (tensor)
#print(results.pandas().xyxy[0])  # im predictions (pandas)
#print(results.pandas().xyxy[0].value_counts('name'))  # class counts (pandas)
