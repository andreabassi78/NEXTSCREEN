import cv2
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlopen

# Load an image with text from an online URL
url = 'https://raw.githubusercontent.com/andreabassi78/NEXTSCREEN/refs/heads/main/images/logo.svg'  # Replace with a valid image URL
response = urlopen(url)
image = cv2.imdecode(np.frombuffer(response.read(), np.uint8), cv2.IMREAD_COLOR)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to preprocess the image
_, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Detect contours to locate characters
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through contours to detect an 'S'-shaped character
for contour in contours:
    # Get the bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour)
    roi = binary_image[y:y+h, x:x+w]

    # Resize the ROI to a fixed size (e.g., 28x28) for comparison
    resized_roi = cv2.resize(roi, (28, 28))

    # Flatten the ROI and compare with predefined 'S' features (basic heuristic)
    # Here, you could use a machine learning model or shape descriptors instead of heuristics
    aspect_ratio = w / h
    if 0.8 < aspect_ratio < 1.2:  # Example heuristic for 'S'
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the result
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Detected Character: S')
plt.axis('off')
plt.show()