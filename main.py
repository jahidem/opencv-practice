import cv2
import numpy as np

directory = "color_image_216"
img2 = cv2.imread(directory + "/14_back_whole.jpg")
img = cv2.imread(directory + "/14_front_whole.jpg")

images = [img, img2]

cv2.imshow("image", img)

img = cv2.medianBlur(img, 5, 3)


hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


# # Define color ranges for different fruits
# purple_lower = np.array([120, 50, 50])
# purple_upper = np.array([150, 255, 255])

# red_lower = np.array([0, 50, 50])
# red_upper = np.array([20, 255, 255])

# yellow_lower = np.array([20, 50, 50])
# yellow_upper = np.array([30, 255, 255])

# # Threshold the image to get binary masks for each color
# purple_mask = cv2.inRange(hsv_image, purple_lower, purple_upper)
# red_mask = cv2.inRange(hsv_image, red_lower, red_upper)
# yellow_mask = cv2.inRange(hsv_image, yellow_lower, yellow_upper)

# # Combine masks to get the final mask
# final_mask = red_mask + yellow_mask + purple_mask


# cv2.imshow("dilated", final_mask)


# Define lower and upper bounds for white color
white_lower = np.array([0, 0, 200])
white_upper = np.array([180, 30, 255])

# Threshold the image to get binary mask for white color
white_mask = cv2.inRange(hsv_image, white_lower, white_upper)

contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

min_blob_area = 400

# Create a mask to store the masked out contours
mask = np.zeros_like(white_mask)

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
    if area < min_blob_area:
        # Draw filled contour on the mask
        cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

# Apply the mask to the original image
masked_image = cv2.bitwise_and(white_mask, white_mask, mask=~mask)

cv2.imshow("Masked Image", masked_image)

cv2.waitKey(0)
