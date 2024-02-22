import cv2
import numpy as np


def count_fruit(image):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges for different fruits
    purple_lower = np.array([120, 50, 50])
    purple_upper = np.array([150, 255, 255])

    red_lower = np.array([0, 50, 50])
    red_upper = np.array([20, 255, 255])

    yellow_lower = np.array([20, 50, 50])
    yellow_upper = np.array([30, 255, 255])

    # Threshold the image to get binary masks for each color
    purple_mask = cv2.inRange(hsv_image, purple_lower, purple_upper)
    red_mask = cv2.inRange(hsv_image, red_lower, red_upper)
    yellow_mask = cv2.inRange(hsv_image, yellow_lower, yellow_upper)

    # Combine masks to get the final mask
    final_mask = red_mask + yellow_mask + purple_mask

    # Find contours in the final mask
    contours, _ = cv2.findContours(
        final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter and count red, yellow, and purple blobs based on size
    min_blob_area = 200  # Adjust this value based on your requirement
    max_blob_area = 800  # Adjust this value based on your requirement
    red_fruits_count = 0
    yellow_fruits_count = 0
    purple_fruits_count = 0

    count = 0
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

        # Calculate circularity
        circularity = (4 * np.pi * area) / (perimeter**2)

        # Filter blobs based on area
        if area > min_blob_area and area < max_blob_area:
            count += 1

            # Calculate the average color within the contour
            average_color = np.mean(
                hsv_image[contour[:, 0, 1], contour[:, 0, 0]], axis=0
            )

            # Determine the dominant color based on the average color
            if red_lower[0] <= average_color[0] <= red_upper[0]:
                red_fruits_count += 1

            elif (
                yellow_lower[0] <= average_color[0] <= yellow_upper[0]
                and circularity >= 0.30
            ):
                yellow_fruits_count += 1
                if area >= 500:
                    yellow_fruits_count += 1

            elif purple_lower[0] <= average_color[0] <= purple_upper[0]:
                purple_fruits_count += 1

    return [yellow_fruits_count, red_fruits_count, purple_fruits_count]


directory = "color_image_216"
number = 4
img_back = cv2.imread(directory + f"/{number}_back_whole.jpg")
img_back = cv2.flip(img_back, 1)
img_front = cv2.imread(directory + f"/{number}_front_whole.jpg")


front_back_list = [[], []]

for index, img_raw in enumerate([img_front, img_back]):
    cv2.imshow("image", img_raw)
    img = cv2.medianBlur(img_raw, 5, 3)

    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for white color
    white_lower = np.array([0, 0, 200])
    white_upper = np.array([180, 30, 255])

    # Define lower and upper bounds for green color
    green_lower = np.array([40, 40, 40])
    green_upper = np.array([70, 255, 255])

    # Threshold the image to get binary masks for white and green colors
    white_mask = cv2.inRange(hsv_image, white_lower, white_upper)
    green_mask = cv2.inRange(hsv_image, green_lower, green_upper)

    # Combine the masks
    combined_mask = cv2.bitwise_or(white_mask, green_mask)

    contours, _ = cv2.findContours(
        combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # make it more than 1700, less than 16800
    min_blob_area = 15000

    # Create a mask to store the masked out contours
    mask = np.zeros_like(combined_mask)

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
            cropped_image = img_raw[y : y + h, x : x + w]
            cropped_images.append((x, cropped_image))

    cv2.imshow("whole", mask)

    # Sort cropped images based on x-coordinate values
    cropped_images.sort(key=lambda x: x[0])

    # Display sorted cropped images
    for i, (_, cropped_image) in enumerate(cropped_images):
        cv2.imshow(f"Cropped Image {i+1}", cropped_image)

    front_back_list[index] = [cropped for (_, cropped) in cropped_images]

    cv2.waitKey(0)

# Determine minimum dimensions among the images in front_back_list
min_width = min([img.shape[1] for sublist in front_back_list for img in sublist])
min_height = min([img.shape[0] for sublist in front_back_list for img in sublist])

# Resize all images in front_back_list to the minimum dimensions
for sublist in front_back_list:
    resized_sublist = []
    for img in sublist:
        resized_img = cv2.resize(img, (min_width, min_height))
        resized_sublist.append(resized_img)
        print(
            "Resized image shape:", resized_img.shape
        )  # Debug print to check dimensions
    sublist[:] = resized_sublist

# Ensure the lists are of the same length
min_length = min(len(front_back_list[0]), len(front_back_list[1]))

for i in range(min_length):
   
    print(count_fruit(front_back_list[0][i]))
    print(count_fruit(front_back_list[1][i]))
    # print(count_duplicate_fruit(front_back_list[0][i], front_back_list[1][i]))
    cv2.waitKey(0)
