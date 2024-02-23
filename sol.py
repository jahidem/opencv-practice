import cv2
import numpy as np

# Define color ranges for different balls
white_lower = np.array([0, 0, 200])
white_upper = np.array([180, 30, 255])

green_lower = np.array([40, 40, 40])
green_upper = np.array([80, 255, 255])

purple_lower = np.array([120, 50, 50])
purple_upper = np.array([150, 255, 255])

red_lower = np.array([0, 50, 50])
red_upper = np.array([20, 255, 255])

yellow_lower = np.array([20, 50, 50])
yellow_upper = np.array([30, 255, 255])



def getContours(image):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


    # Threshold the image to get binary masks for each color
    white_mask = cv2.inRange(hsv_image, white_lower, white_upper)
    green_mask = cv2.inRange(hsv_image, green_lower, green_upper)
    purple_mask = cv2.inRange(hsv_image, purple_lower, purple_upper)
    red_mask = cv2.inRange(hsv_image, red_lower, red_upper)
    yellow_mask = cv2.inRange(hsv_image, yellow_lower, yellow_upper)

    # Combine masks to get the final mask
    final_mask = red_mask + yellow_mask + purple_mask + white_mask + green_mask


    # Find contours in the final mask
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and count red, yellow, and purple blobs based on size
    min_blob_area = 200  # Adjust this value based on your requirement
    max_blob_area = 800  # Adjust this value based on your requirement

    red_contours = []
    purple_contours = []
    yellow_contours = []
    white_contours = []
    green_contours = []

    count = 0
    for i, contour in enumerate(contours):
        # Calculate the area of the contour
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if (area <= 0 or perimeter <= 0):
            continue
        
        print(i, area)

        # Approximate the contour
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Get the bounding box
        x, y, w, h = cv2.boundingRect(approx)

        # Calculate aspect ratio
        aspect_ratio = float(w) / h

        # Calculate circularity
        circularity = (4 * np.pi * area) / (perimeter ** 2)

        # Calculate convexity
        convexity = cv2.isContourConvex(contour)

        # number of vertices
        num_vertices = len(approx)

        # Filter blobs based on area
        if area > min_blob_area and area < max_blob_area:
            count += 1

            # Calculate the average color within the contour
            average_color = np.mean(hsv_image[contour[:, 0, 1], contour[:, 0, 0]], axis=0)

            # Determine the dominant color based on the average color
            if red_lower[0] <= average_color[0] <= red_upper[0]:
                red_contours.append(contour)
                cv2.drawContours(image, [contour], -1, (0, 0, 0), 2)

            elif yellow_lower[0] <= average_color[0] <= yellow_upper[0] and circularity >= 0.30:
                yellow_contours.append(contour)

            elif purple_lower[0] <= average_color[0] <= purple_upper[0]:
                purple_contours.append(contour)
            
            elif white_lower[0] <= average_color[0] <= white_upper[0]:
                white_contours.append(contour)
                cv2.drawContours(image, [contour], -1, (0, 0, 0), 2)
            
            elif green_lower[0] <= average_color[0] <= green_upper[0]:
                green_contours.append(contour)
                cv2.drawContours(image, [contour], -1, (0, 0, 0), 2)


    cv2.imshow('Original Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return [yellow_contours, red_contours, purple_contours]



def test(image):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    red_lower1 = np.array([0, 100, 100])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 100, 100])
    red_upper2 = np.array([180, 255, 255])

    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])

    purple_lower = np.array([120, 50, 50])
    purple_upper = np.array([150, 255, 255])

    # Threshold the image to get binary masks for each color
    white_mask = cv2.inRange(hsv_image, white_lower, white_upper)
    green_mask = cv2.inRange(hsv_image, green_lower, green_upper)
    red_mask1 = cv2.inRange(hsv_image, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv_image, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    yellow_mask = cv2.inRange(hsv_image, yellow_lower, yellow_upper)
    purple_mask = cv2.inRange(hsv_image, purple_lower, purple_upper)

    # Combine masks to get the final masks
    white_result = cv2.bitwise_and(image, image, mask=white_mask)
    green_result = cv2.bitwise_and(image, image, mask=green_mask)
    red_result = cv2.bitwise_and(image, image, mask=red_mask)
    yellow_result = cv2.bitwise_and(image, image, mask=yellow_mask)
    purple_result = cv2.bitwise_and(image, image, mask=purple_mask)

    # Find contours in each mask
    contours_white, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_green, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_yellow, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_purple, _ = cv2.findContours(purple_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # print(contours_white)

    # Draw contours on the original image
    image_with_contours = image.copy()
    # cv2.drawContours(image_with_contours, contours_white, -1, (0, 0, 0), 2)
    # cv2.drawContours(image_with_contours, contours_green, -1, (0, 255, 0), 2)
    # cv2.drawContours(image_with_contours, contours_red, -1, (0, 255, 0), 2)
    # cv2.drawContours(image_with_contours, contours_yellow, -1, (0, 255, 0), 2)
    # cv2.drawContours(image_with_contours, contours_purple, -1, (0, 255, 0), 2)



    count = 0
    for i, contour in enumerate(contours_white):
        # Calculate the area of the contour
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if (area <= 0 or perimeter <= 0):
            continue
        
        x, y = contour[0][0]
        print(i, area, perimeter, contour[0][0], x, y)
        # cv2.drawContours(image_with_contours, [contour[0]], -1, (0, 0, 0), 2)
        cv2.drawContours(image_with_contours, [contour], -1, (0, 0, 0), 2)
        # break
        count += 1
        if area >= 10000:
            helloSquare(contour, image)

    print("count:", count)
    # Display the results
    cv2.imshow('Original Image', image)
    cv2.imshow('White Segmentation', white_result)
    cv2.imshow('Green Segmentation', green_result)
    cv2.imshow('Red Segmentation', red_result)
    cv2.imshow('Yellow Segmentation', yellow_result)
    cv2.imshow('Purple Segmentation', purple_result)
    cv2.imshow('Image with Contours', image_with_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def largest_contained_square(contour):
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    hull = cv2.convexHull(approx)

    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    box = box.astype(int)

    return box

def get_square_image(contour, image):
    square_points = largest_contained_square(contour)

    # Find bounding box coordinates of the contour
    x, y, w, h = cv2.boundingRect(square_points)

    # Crop the region from the original image
    cropped_image = image[y:y+h, x:x+w]

    # Draw the square on a copy of the original image
    result_image = image.copy()
    cv2.drawContours(result_image, [square_points], 0, (0, 0, 0), 2)

    # Display the result
    cv2.imshow('Largest Square Inside Contour', result_image)
    cv2.imshow('Cropped Image', cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return cropped_image


def get_white_parts(image):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Threshold the image to get binary masks for each color
    white_mask = cv2.inRange(hsv_image, white_lower, white_upper)

    # Combine masks to get the final masks
    white_result = cv2.bitwise_and(image, image, mask=white_mask)

    # Find contours in each mask
    contours_white, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_with_contours = image.copy()

    square_images = []

    count = 0
    for i, contour in enumerate(contours_white):
        # Calculate the area of the contour
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if (area <= 0 or perimeter <= 0):
            continue
        
        x, y = contour[0][0]
        print(i, area, perimeter, contour[0][0], x, y)
        # cv2.drawContours(image_with_contours, [contour[0]], -1, (0, 0, 0), 2)
        cv2.drawContours(image_with_contours, [contour], -1, (0, 0, 0), 2)
        # break
        count += 1
        if area >= 10000:
            square_image = get_square_image(contour, image)
            square_images.append([x, square_image])

    print("count:", count)
    # Display the results
    cv2.imshow('Original Image', image)
    cv2.imshow('White Segmentation', white_result)
    cv2.imshow('Image with Contours', image_with_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return sorted(square_images)

def replace_green_with_white(image):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the color range for green in HSV
    green_lower = np.array([40, 40, 40])
    green_upper = np.array([80, 255, 255])

    # Threshold the image to get a binary mask for green
    green_mask = cv2.inRange(hsv_image, green_lower, green_upper)

    result_image = image.copy()
    # Change the green part to white in the original image
    result_image[np.where(green_mask)] = [255, 255, 255]

    # Display or save the result_image
    cv2.imshow("Image", image)
    cv2.imshow("Green Mask", green_mask)
    cv2.imshow("result_image Image", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return result_image


def solve_problem(front_image, back_image):
    mirrored_back_image = cv2.flip(back_image, 1) # 1 for horizontal flip, 0 for vertical flip, -1 for both

    front_image_without_green = replace_green_with_white(front_image)
    mirrored_back_image_without_green = replace_green_with_white(mirrored_back_image)



    cv2.imshow('front_image', front_image)
    cv2.imshow('back_image', back_image)
    cv2.imshow('mirrored_back_image', mirrored_back_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('front_image_without_green', front_image_without_green)
    cv2.imshow('mirrored_back_image_without_green', mirrored_back_image_without_green)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    front_square_parts = get_white_parts(front_image_without_green)
    back_square_parts = get_white_parts(mirrored_back_image_without_green)


    for i, front_part in enumerate(front_square_parts):
        print(front_part[0], "front")
        cv2.imshow('front_part_' + str(i), front_image_without_green)

    for back_part in back_square_parts:
        print(back_part[0], "back")

    # print("front", front_square_parts)
    # print("back", back_square_parts)



# Read the image
front_image = cv2.imread('color_image_216/5_front_whole.jpg')
back_image = cv2.imread('color_image_216/5_back_whole.jpg')

solve_problem(front_image, back_image)
