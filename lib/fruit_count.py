import math
from typing import Sequence
import cv2
import numpy as np
from cv2.typing import MatLike


class FRUIT_COUNT:
    def __init__(self) -> None:
        self.average_eggplant = 243
        self.average_tomato = 427
        self.average_pepper = 346
        

        self.red_lower1 = np.array([0, 100, 100])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([160, 100, 100])
        self.red_upper2 = np.array([180, 255, 255])

        self.red_lower = np.array([0, 50, 50])
        self.red_upper = np.array([20, 255, 255])

        self.yellow_lower = np.array([20, 100, 100])
        self.yellow_upper = np.array([30, 255, 255])

        self.purple_lower = np.array([120, 50, 50])
        self.purple_upper = np.array([150, 255, 255])

    def get_contours_fruit(self, image) -> Sequence[MatLike]:
        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Threshold the image to get binary masks for each color
        purple_mask = cv2.inRange(hsv_image, self.purple_lower, self.purple_upper)
        red_mask1 = cv2.inRange(hsv_image, self.red_lower1, self.red_upper1)
        red_mask2 = cv2.inRange(hsv_image, self.red_lower2, self.red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
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
        contours = self.get_contours_fruit(image)

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
                    count = math.ceil(area / self.average_tomato)
                    red_fruits_count += count if count < 3 else 0
                elif (
                    self.yellow_lower[0] <= average_color[0] <= self.yellow_upper[0]
                    and circularity >= 0.30
                ):
                    count = math.ceil(area / self.average_pepper)
                    yellow_fruits_count += count if count < 3 else 0
                elif self.purple_lower[0] <= average_color[0] <= self.purple_upper[0]:
                    count = math.ceil(area / self.average_eggplant)
                    purple_fruits_count += count if count < 3 else 0

        return [yellow_fruits_count, red_fruits_count, purple_fruits_count]
