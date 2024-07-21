import cv2
import numpy as np

# Load the image
image = cv2.imread('invoice.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Use Canny edge detection
edged = cv2.Canny(blurred, 75, 200)

# Find contours
contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw bounding boxes
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 1000:  # Adjust this threshold based on your image size
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 0)

# Show the output image
cv2.imshow("Bounding Boxes", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
