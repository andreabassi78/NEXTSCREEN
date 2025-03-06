"""
Face recognition within an image    

For description of cascade classifiers read https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request

# Load an image from a URL
url = 'https://raw.githubusercontent.com/andreabassi78/NEXTSCREEN/refs/heads/future/images/meme.jpg'
response = urllib.request.urlopen(url)
image_data = np.asarray(bytearray(response.read()), dtype=np.uint8)
image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Convert the image to grayscale (Haar cascades require grayscale images)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform face detection
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
# scalefactor 1.1 : resizes the image in each iteration to find faces of different sizes
# minNeighbors = 5 minimum number of detected neighbors rectangles to confirm that a face was detected
# minSize = (30,30) minimum size of the object to be detected  


# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Display the result
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Detected Character: S')
plt.axis('off')
plt.show()