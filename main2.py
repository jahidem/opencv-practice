import cv2
import numpy as np

directory = "color_image_216"
img = cv2.imread(directory + "/14_back_whole.jpg")
img2 = cv2.imread(directory + "/14_front_whole.jpg")

images = [img, img2]

cv2.imshow("image", img)

hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define lower and upper bounds for white color
white_lower = np.array([0, 0, 200])
white_upper = np.array([180, 30, 255])

# Threshold the image to get binary mask for white color
white_mask = cv2.inRange(hsv_image, white_lower, white_upper)

contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

min_blob_area = 200

# Create a mask to store the masked out contours
mask = np.zeros_like(white_mask)

cropped_images = []

for i, contour in enumerate(contours):
    # Calculate the area of the contour
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if area <= 0 or perimeter <= 0:
        continue

    # Approximate the contour
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Get the bounding box
    x, y, w, h = cv2.boundingRect(approx)

    # Filter blobs based on area
    if area > min_blob_area:
        # Draw filled contour on the mask
        cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
        
        # Crop the region of interest (ROI) from the original image
        cropped_image = img[y:y+h, x:x+w]
        cropped_images.append(cropped_image)

# Display cropped images
for i, cropped_image in enumerate(cropped_images):
    cv2.imshow(f"Cropped Image {i+1}", cropped_image)

cv2.waitKey(0)
