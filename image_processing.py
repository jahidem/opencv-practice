import math
from typing import List, Sequence
import cv2
import numpy as np
from cv2.typing import MatLike


def find_topmost_point(contour, rightmost=True):
    # Initialize variables to store the topmost point and its x-coordinate
    topmost_point = None

    # Iterate over all points in the contour
    for point in contour:
        # Get the x and y coordinates of the current point
        x, y = point[0]
        print("Point:", x, y)

        # Update the topmost point if it's None or if the current point has a smaller y-coordinate
        if topmost_point is None:
            topmost_point = (x, y)
        elif (not rightmost and (x > topmost_point[1])) or (
            rightmost and (x < topmost_point[1])
        ):
            topmost_point = (x, y)

    # Ensure that topmost_point is not None
    if topmost_point is not None:
        # Convert the coordinates to integers
        topmost_point = (round(topmost_point[0]), round(topmost_point[1]))

    return topmost_point


def find_top_left_contour(image):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for white color in HSV
    white_lower = np.array([0, 0, 200])
    white_upper = np.array([180, 30, 255])

    # Threshold the image to get a binary mask for white color
    white_mask = cv2.inRange(hsv_image, white_lower, white_upper)

    # Find contours in the white mask
    contours, _ = cv2.findContours(
        white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Sort contours based on their x-coordinate (left to right)
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])

    # Get the leftmost contour
    leftmost_contour = contours[0] if contours else None

    return leftmost_contour


def crop_image(image):
    contour = find_top_left_contour(image)
    # top lef
    x, y = find_topmost_point(contour, rightmost=False)
    if x is not None and y is not None:
        image = image[y:, x:]

    # top right
    # x, y = find_topmost_point(contour, rightmost=True)
    # if x is not None and y is not None:
    #     image = image[:, :x]
    return image


class IMAGE_PROCESSING:
    def __init__(self) -> None:
        # Define color ranges for different fruits
        self.purple_lower = np.array([120, 50, 50])
        self.purple_upper = np.array([150, 255, 255])
        self.average_eggplant = 243

        self.red_lower = np.array([0, 50, 50])
        self.red_upper = np.array([20, 255, 255])
        self.average_tomato = 427

        self.yellow_lower = np.array([20, 50, 50])
        self.yellow_upper = np.array([30, 255, 255])
        self.average_pepper = 346

        # Define lower and upper bounds for white color
        self.white_lower = np.array([0, 0, 200])
        self.white_upper = np.array([180, 30, 255])

        # Define lower and upper bounds for green color
        self.green_lower = np.array([40, 40, 40])
        self.green_upper = np.array([70, 255, 255])

    def calculate_iou(self, mask1, mask2):
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        iou = np.sum(intersection) / np.sum(union)

        return iou

    def get_contours(self, image) -> Sequence[MatLike]:
        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Threshold the image to get binary masks for each color
        purple_mask = cv2.inRange(hsv_image, self.purple_lower, self.purple_upper)
        red_mask = cv2.inRange(hsv_image, self.red_lower, self.red_upper)
        yellow_mask = cv2.inRange(hsv_image, self.yellow_lower, self.yellow_upper)

        # Combine masks to get the final mask
        final_mask = red_mask + yellow_mask + purple_mask

        # Find contours in the final mask
        contours, _ = cv2.findContours(
            final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return contours

    def count_fruits(self, image):
        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Find contours in the final mask
        contours = self.get_contours(image)

        # Filter and count red, yellow, and purple blobs based on size
        min_blob_area = 200  # Adjust this value based on your requirement
        red_fruits_count = 0
        yellow_fruits_count = 0
        purple_fruits_count = 0

        for i, contour in enumerate(contours):
            # Calculate the area of the contour
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if area <= 0 or perimeter <= 0:
                continue

            # Approximate the contour
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Calculate circularity
            circularity = (4 * np.pi * area) / (perimeter**2)
            # Filter blobs based on area
            if area > min_blob_area:
                # Calculate the average color within the contour
                average_color = np.mean(
                    hsv_image[contour[:, 0, 1], contour[:, 0, 0]], axis=0
                )

                # Determine the dominant color based on the average color
                if self.red_lower[0] <= average_color[0] <= self.red_upper[0]:
                    red_fruits_count += math.ceil(area / self.average_tomato)
                elif (
                    self.yellow_lower[0] <= average_color[0] <= self.yellow_upper[0]
                    and circularity >= 0.30
                ):
                    yellow_fruits_count += math.ceil(area / self.average_pepper)
                elif self.purple_lower[0] <= average_color[0] <= self.purple_upper[0]:
                    purple_fruits_count += math.ceil(area / self.average_eggplant)

        return [yellow_fruits_count, red_fruits_count, purple_fruits_count]

    def get_list_plants(self, img_raw) -> List[MatLike]:

        hsv_image = cv2.cvtColor(img_raw, cv2.COLOR_BGR2HSV)

        # Threshold the image to get binary masks for white and green colors
        white_mask = cv2.inRange(hsv_image, self.white_lower, self.white_upper)
        green_mask = cv2.inRange(hsv_image, self.green_lower, self.green_upper)

        # Combine the masksd
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
            # TODO: for now setting to 0.16
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Get the bounding box
            x, y, w, h = cv2.boundingRect(approx)

            # Filter blobs based on area
            if area > min_blob_area:
                # Draw filled contour on the mask
                cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

                # Crop the region of interest (ROI) from the original image

                # top lef
                x_l, _ = find_topmost_point(contour, rightmost=False)
                x_r, _ = find_topmost_point(contour, rightmost=True)

                cropped_image = img_raw[
                    y : y+h,
                    x if x_l is None else x_l : x+w if x_r is None else x_r,
                    # x:x+w
                ]

                cropped_images.append((x if x_l is None else x_l , cropped_image))

        cv2.imshow("whole", mask)

        # Sort cropped images based on x-coordinate values
        cropped_images.sort(key=lambda x: x[0])

        return [cropped for (_, cropped) in cropped_images]


image_processing = IMAGE_PROCESSING()

for plant in range(1, 28):
    directory = "color_image_216"
    number = plant
    img_back = cv2.imread(directory + f"/{number}_back_whole.jpg")
    img_back = cv2.flip(img_back, 1)
    img_front = cv2.imread(directory + f"/{number}_front_whole.jpg")

    front_back_list = [[], []]

    for index, img_raw in enumerate([img_front, img_back]):

        front_back_list[index] = image_processing.get_list_plants(img_raw)

        cv2.waitKey(0)

    if not len(front_back_list[0]):
        continue

    # Ensure the lists are of the same length
    min_length = min(len(front_back_list[0]), len(front_back_list[1]))

    sum_all = [0, 0, 0]
    for i in range(min_length):
        front_count = image_processing.count_fruits(front_back_list[0][i])
        back_count = image_processing.count_fruits(front_back_list[1][i])

        print(front_count)
        print(back_count)
        print()
        cv2.imshow("front", front_back_list[0][i])
        cv2.imshow("back", front_back_list[1][i])
        cv2.waitKey(0)

        sum_all = [i + j + k for i, j, k in zip(sum_all, front_count, back_count)]
    print(sum_all)
