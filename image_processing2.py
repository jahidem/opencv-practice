import math
from typing import List, Sequence
import cv2
import numpy as np


class FRUIT_COUNT:
    def __init__(self) -> None:
        # Define color ranges for different fruits

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

    def isSimilar(self, mask1, mask2):
        # Ensure the masks have the same dimensions
        if mask1.shape != mask2.shape:
            # raise ValueError("The mask images must have the same dimensions.") TODO: maybe not a good idea to leave it here
            return False  # TODO: for now returning false as the shape mismatch

        # Calculate the Intersection over Union (IoU)
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        iou = np.sum(intersection) / np.sum(union)

        # You can set a threshold to determine if the masks are similar
        iou_threshold = 0.5  # Adjust as needed TODO : too less?
        if iou > iou_threshold:
            return True
        else:
            return False

    def fruit_color_mask(self, img):
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Threshold the image to get binary masks for each color
        final_mask = cv2.inRange(hsv_image, self.yellow_lower, self.yellow_upper)

        red_mask1 = cv2.inRange(hsv_image, self.red_lower1, self.red_upper1)
        red_mask2 = cv2.inRange(hsv_image, self.red_lower2, self.red_upper2)
        final_mask += cv2.bitwise_or(red_mask1, red_mask2)

        final_mask += cv2.inRange(hsv_image, self.purple_lower, self.purple_upper)

        return final_mask

    def getContours(self, image):
        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        final_mask = self.fruit_color_mask(image)

        # Find contours in the final mask
        contours, _ = cv2.findContours(
            final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter and count red, yellow, and purple blobs based on size
        min_blob_area = 200  # Adjust this value based on your requirement

        red_contours = []
        purple_contours = []
        yellow_contours = []

        count = 0
        for _, contour in enumerate(contours):
            # Calculate the area of the contour
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if area <= 0 or perimeter <= 0:
                continue

            # Filter blobs based on area
            if area > min_blob_area:
                count += 1
                # Calculate the average color within the contour
                average_color = np.mean(
                    hsv_image[contour[:, 0, 1], contour[:, 0, 0]], axis=0
                )

                # Determine the dominant color based on the average color
                if self.red_lower[0] <= average_color[0] <= self.red_upper[0]:
                    red_contours.append(contour)

                elif self.yellow_lower[0] <= average_color[0] <= self.yellow_upper[0]:
                    yellow_contours.append(contour)

                elif self.purple_lower[0] <= average_color[0] <= self.purple_upper[0]:
                    purple_contours.append(contour)

        return [yellow_contours, red_contours, purple_contours]

    def getContourImage(self, contour, shape):
        height, width = shape

        # Create an empty mask
        mask = np.zeros((height, width), dtype=np.uint8)

        # Draw the contour on the mask
        cv2.drawContours(mask, [contour], 0, (255), thickness=cv2.FILLED)
        return mask

    def count_duplicate_fruits(self, front_image, back_image):

        yellow_contours_front, red_contours_front, purple_contours_front = (
            self.getContours(front_image)
        )
        yellow_contours_back, red_contours_back, purple_contours_back = (
            self.getContours(back_image)
        )

        count = 0
        # Check if we are counting yellow
        if True:
            for i, contour_front in enumerate(yellow_contours_front):
                contour_image_front = self.getContourImage(contour_front, [110, 110])

                for i, contour_back in enumerate(yellow_contours_back):
                    contour_image_back = self.getContourImage(contour_back, [110, 110])

                    if self.isSimilar(contour_image_front, contour_image_back):
                        count += round(
                            cv2.contourArea(contour_front) / self.average_pepper
                        )

        # Check if we are counting red
        if True:
            for i, contour_front in enumerate(red_contours_front):
                contour_image_front = self.getContourImage(contour_front, [110, 110])

                for i, contour_back in enumerate(red_contours_back):
                    contour_image_back = self.getContourImage(contour_back, [110, 110])

                    if self.isSimilar(contour_image_front, contour_image_back):
                        count += round(
                            cv2.contourArea(contour_front) / self.average_tomato
                        )

        # Check if we are counting purple
        if True:
            for i, contour_front in enumerate(purple_contours_front):
                contour_image_front = self.getContourImage(contour_front, [110, 110])

                for i, contour_back in enumerate(purple_contours_back):
                    contour_image_back = self.getContourImage(contour_back, [110, 110])

                    if self.isSimilar(contour_image_front, contour_image_back):
                        count += round(
                            cv2.contourArea(contour_front) / self.average_eggplant
                        )

        return count

    def get_contours_fruit(self, image):
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
                    red_fruits_count += round(area / self.average_tomato)
                elif (
                    self.yellow_lower[0] <= average_color[0] <= self.yellow_upper[0]
                    and circularity >= 0.30
                ):
                    yellow_fruits_count += round(area / self.average_pepper)
                elif self.purple_lower[0] <= average_color[0] <= self.purple_upper[0]:
                    purple_fruits_count += round(area / self.average_eggplant)

        return [yellow_fruits_count, red_fruits_count, purple_fruits_count]


class IMAGE_PROCESSING:
    def __init__(self) -> None:
        # Define lower and upper bounds for white color
        self.white_lower = np.array([0, 0, 200])
        self.white_upper = np.array([180, 30, 255])

        # Define the color range for green in HSV
        self.green_lower = np.array([40, 40, 40])
        self.green_upper = np.array([80, 255, 255])

    def calculate_iou(self, mask1, mask2):
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        iou = np.sum(intersection) / np.sum(union)

        return iou

    def get_white_parts(self, image):
        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Threshold the image to get binary masks for each color
        white_mask = cv2.inRange(hsv_image, self.white_lower, self.white_upper)

        # Find contours in each mask
        contours_white, _ = cv2.findContours(
            white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        image_with_contours = image.copy()

        square_images = []

        count = 0
        for i, contour in enumerate(contours_white):
            # Calculate the area of the contour
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if area <= 0 or perimeter <= 0:
                continue

            x, y = contour[0][0]
            # print(i, area, perimeter, contour[0][0], x, y)
            # cv2.drawContours(image_with_contours, [contour[0]], -1, (0, 0, 0), 2)
            cv2.drawContours(image_with_contours, [contour], -1, (0, 0, 0), 2)
            # break
            count += 1
            if area >= 10000:
                square_image = self.get_square_image(contour, image)
                square_images.append((x, square_image))
        square_images.sort(key=lambda x: x[0])
        return [square_image for _, square_image in square_images]

    def is_white_point(self, img, point):
        hsv_image = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)
        return np.all(
            np.logical_and(
                self.white_lower <= hsv_image[point[1]][point[0]],
                hsv_image[point[1]][point[0]] <= self.white_upper,
            )
        )

    def is_white_around(self, img, point):
        is_white = True
        for x in range(-3, -3):
            for y in range(-3, -3):
                x_now = x + point[0]
                y_now = y + point[1]

                if x_now < img.shape[1] and y_now < img.shape[0]:
                    is_white = is_white and self.is_white_point(img, [x_now, y_now])
        return is_white

    def get_square_image(self, contour, image):
        # square_points = self.largest_contained_square(contour)
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        x, y, w, h = cv2.boundingRect(approx)

        hsv_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)

        left_most = [x, y]
        while left_most[0] < x + w:
            if np.all(
                np.logical_and(
                    self.white_lower <= hsv_image[left_most[1]][left_most[0]],
                    hsv_image[left_most[1]][left_most[0]] <= self.white_upper,
                )
            ):
                break
            left_most[0] += 1

        right_most = [x + w, y]
        while right_most[0] > left_most[0]:
            if np.logical_and(
                (self.white_lower <= hsv_image[right_most[1]][right_most[0]]).all(),
                (hsv_image[right_most[1]][right_most[0]] <= self.white_upper).all(),
            ):
                break
            right_most[0] -= 1
        # print(left_most)
        # print(right_most)
        w = right_most[0] - left_most[0] + 1
        expected_w = 135

        if w < expected_w:
            if self.is_white_around(image, [left_most[0] + expected_w, left_most[1]]):
                right_most[0] = left_most[0] + expected_w
            elif self.is_white_around(
                image, [right_most[0] - expected_w, right_most[1]]
            ):
                left_most[0] = right_most[0] - expected_w

        # Crop the region from the original image
        cropped_image = image[y : y + h, left_most[0] : right_most[0]]

        return cropped_image

    def largest_contained_square(self, contour):
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        hull = cv2.convexHull(approx)

        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        box = box.astype(int)

        return box

    def replace_green_with_white(self, image):
        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Threshold the image to get a binary mask for green
        green_mask = cv2.inRange(hsv_image, self.green_lower, self.green_upper)

        no_green_image = image.copy()
        # Change the green part to white in the original image
        no_green_image[np.where(green_mask)] = [255, 255, 255]

        return no_green_image

    def get_plant_list(self, front_image, back_image):
        mirrored_back_image = cv2.flip(
            back_image, 1
        )  # 1 for horizontal flip, 0 for vertical flip, -1 for both

        front_image_without_green = self.replace_green_with_white(front_image)
        mirrored_back_image_without_green = self.replace_green_with_white(
            mirrored_back_image
        )

        front_square_parts = self.get_white_parts(front_image_without_green)
        back_square_parts = self.get_white_parts(mirrored_back_image_without_green)

        return [front_square_parts, back_square_parts]

    def resized_plants_list(self, front_image, back_image):
        front_back_plants = self.get_plant_list(front_image, back_image)

        if not len(front_back_plants[0]):
            return [[], []]

        # Determine minimum dimensions among the images in front_back_list
        min_width = min(
            [img.shape[1] for sublist in front_back_plants for img in sublist]
        )
        min_height = min(
            [img.shape[0] for sublist in front_back_plants for img in sublist]
        )

        # Resize all images in front_back_list to the minimum dimensions
        for sublist in front_back_plants:
            resized_sublist = []
            for img in sublist:
                resized_img = cv2.resize(img, (min_width, min_height))
                resized_sublist.append(resized_img)
            sublist[:] = resized_sublist

        return front_back_plants


image_processing = IMAGE_PROCESSING()
fruit_count = FRUIT_COUNT()

for plant in range(1, 28):
    directory = "color_image"
    number = plant
    img_back = cv2.imread(directory + f"/-{number}.jpg")
    img_front = cv2.imread(directory + f"/{number}.jpg")
    cv2.imshow(f"front {number}", img_front)
    cv2.imshow(f"back {number}", img_back)

    front_back_list = image_processing.get_plant_list(img_front, img_back)
    min_length = min(len(front_back_list[0]), len(front_back_list[1]))

    sum_all = [0, 0, 0]
    for i in range(min_length):
        front_count = fruit_count.count_fruits(front_back_list[0][i])
        back_count = fruit_count.count_fruits(front_back_list[1][i])
        duplicate = fruit_count.count_duplicate_fruits(
            front_back_list[0][i], front_back_list[1][i]
        )
        cv2.imshow(f"front_{number}_{i}", (front_back_list[0][i]))
        cv2.imshow(f"back_{number}_{i}", (front_back_list[1][i]))
        print("front", front_count)
        print("back", back_count)
        print("duplicate", duplicate)
        print()

        sum_all = [i + j + k for i, j, k in zip(sum_all, front_count, back_count)]

    cv2.waitKey(0)
    cv2.destroyAllWindows()
