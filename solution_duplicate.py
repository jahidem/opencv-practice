import cv2
import numpy as np
import csv


# Define color ranges for different balls
purple_lower = np.array([120, 50, 50])
purple_upper = np.array([150, 255, 255])

red_lower = np.array([0, 50, 50])
red_upper = np.array([20, 255, 255])

yellow_lower = np.array([20, 50, 50])
yellow_upper = np.array([30, 255, 255])

# Define rectangle points (x, y) of the top-left and bottom-right corners
first_top_left = (60, 190)  # Example coordinates
first_bottom_right = (170, 300)  # Example coordinates

# Define rectangle points (x, y) of the top-left and bottom-right corners
second_top_left = (275, 190)  # Example coordinates
second_bottom_right = (385, 300)  # Example coordinates

# Define rectangle points (x, y) of the top-left and bottom-right corners
third_top_left = (480, 190)  # Example coordinates
third_bottom_right = (590, 300)  # Example coordinates

# Function to generate a random color
def random_color():
    return tuple(np.random.randint(0, 255, 3).tolist())


def getShape(contour):
    # Approximate the contour
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Get the bounding box
    x, y, w, h = cv2.boundingRect(approx)

    # Calculate aspect ratio
    aspect_ratio = float(w) / h

    # Calculate area and perimeter
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # Calculate circularity
    circularity = (4 * np.pi * area) / (perimeter ** 2)

    # Determine the shape
    if circularity > 0.85:
        return "Circle"
    elif 0.85 >= circularity >= 0.6:
        return "Ellipse"
    elif len(approx) == 3:
        return "Triangle"
    elif len(approx) == 4 and aspect_ratio >= 0.95 and aspect_ratio <= 1.05:
        return "Square"
    elif len(approx) == 4:
        return "Rectangle"
    else:
        return "Irregular"

def solve(image):
      
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


    # Threshold the image to get binary masks for each color
    purple_mask = cv2.inRange(hsv_image, purple_lower, purple_upper)
    red_mask = cv2.inRange(hsv_image, red_lower, red_upper)
    yellow_mask = cv2.inRange(hsv_image, yellow_lower, yellow_upper)

    # Combine masks to get the final mask
    final_mask = red_mask + yellow_mask + purple_mask


    # Find contours in the final mask
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and count red, yellow, and purple blobs based on size
    min_blob_area = 200  # Adjust this value based on your requirement
    max_blob_area = 800  # Adjust this value based on your requirement
    red_balls_count = 0
    yellow_balls_count = 0
    purple_balls_count = 0


    # Create a CSV file
    csv_file_path = 'csv_results/' + image_file_name + '_results.csv'
    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = ['count', 'color', 'area', 'aspect_ratio', 'num_vertices', 'circularity', 'convexity', 'shape']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header row
        writer.writeheader()

        count = 0
        for i, contour in enumerate(contours):
            # Calculate the area of the contour
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if (area <= 0 or perimeter <= 0):
                continue
            
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
                # Draw the contour on the original image
                color = random_color()
                # cv2.drawContours(image, [contour], -1, color, 2)

                
                # Calculate the average color within the contour
                average_color = np.mean(hsv_image[contour[:, 0, 1], contour[:, 0, 0]], axis=0)
                


                # Determine the dominant color based on the average color
                if red_lower[0] <= average_color[0] <= red_upper[0]:
                    red_balls_count += 1
                    print(count, "red", area, aspect_ratio, num_vertices, circularity, convexity, getShape(contour))
                    writer.writerow({
                        'count': count,
                        'color': 'red',
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'num_vertices': num_vertices,
                        'circularity': circularity,
                        'convexity': convexity,
                        'shape': getShape(contour)
                    })

                    cv2.drawContours(image, [contour], -1, (0, 0, 0), 2)

                    # Find the centroid of the contour to place the text
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])

                        # Write the contour index inside the contour
                        cv2.putText(red_mask, str(count), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                elif yellow_lower[0] <= average_color[0] <= yellow_upper[0] and circularity >= 0.30:
                    yellow_balls_count += 1
                    if area >= 500:
                        yellow_balls_count += 1
                    print(count, "yellow", area, aspect_ratio, num_vertices, circularity, convexity, getShape(contour))
                    writer.writerow({
                        'count': count,
                        'color': 'yellow',
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'num_vertices': num_vertices,
                        'circularity': circularity,
                        'convexity': convexity,
                        'shape': getShape(contour)
                    })


                    cv2.drawContours(image, [contour], -1, (0, 0, 0), 2)

                    # Find the centroid of the contour to place the text
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])

                        # Write the contour index inside the contour
                        cv2.putText(yellow_mask, str(count), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                elif purple_lower[0] <= average_color[0] <= purple_upper[0]:
                    purple_balls_count += 1
                    print(count, "purple", area, aspect_ratio, num_vertices, circularity, convexity, getShape(contour))
                    writer.writerow({
                        'count': count,
                        'color': 'purple',
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'num_vertices': num_vertices,
                        'circularity': circularity,
                        'convexity': convexity,
                        'shape': getShape(contour)
                    })

                    cv2.drawContours(image, [contour], -1, (0, 0, 0), 2)

                    # Find the centroid of the contour to place the text
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])

                        # Write the contour index inside the contour
                        cv2.putText(purple_mask, str(count), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)



    # Print the counts of red, yellow, and purple balls
    print("Number of Red Balls:", red_balls_count)
    print("Number of Yellow Balls:", yellow_balls_count)
    print("Number of Purple Balls:", purple_balls_count)

    # Display the results
    cv2.imshow('Original Image', image)
    cv2.imshow('HSV Image', hsv_image)
    cv2.imshow('Purple Balls ' + str(purple_balls_count), purple_mask)
    cv2.imshow('Red Balls ' + str(red_balls_count), red_mask)
    cv2.imshow('Yellow Balls ' + str(yellow_balls_count), yellow_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return [yellow_balls_count, red_balls_count, purple_balls_count]



def getContours(image):
        # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


    # Threshold the image to get binary masks for each color
    purple_mask = cv2.inRange(hsv_image, purple_lower, purple_upper)
    red_mask = cv2.inRange(hsv_image, red_lower, red_upper)
    yellow_mask = cv2.inRange(hsv_image, yellow_lower, yellow_upper)

    # Combine masks to get the final mask
    final_mask = red_mask + yellow_mask + purple_mask


    # Find contours in the final mask
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and count red, yellow, and purple blobs based on size
    min_blob_area = 200  # Adjust this value based on your requirement
    max_blob_area = 800  # Adjust this value based on your requirement

    red_contours = []
    purple_contours = []
    yellow_contours = []

    count = 0
    for i, contour in enumerate(contours):
        # Calculate the area of the contour
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if (area <= 0 or perimeter <= 0):
            continue
        
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
            # Draw the contour on the original image
            color = random_color()
            # cv2.drawContours(image, [contour], -1, color, 2)

            
            # Calculate the average color within the contour
            average_color = np.mean(hsv_image[contour[:, 0, 1], contour[:, 0, 0]], axis=0)
            


            # Determine the dominant color based on the average color
            if red_lower[0] <= average_color[0] <= red_upper[0]:
                red_contours.append(contour)

            elif yellow_lower[0] <= average_color[0] <= yellow_upper[0] and circularity >= 0.30:
                yellow_contours.append(contour)

            elif purple_lower[0] <= average_color[0] <= purple_upper[0]:
                purple_contours.append(contour)

    return [yellow_contours, red_contours, purple_contours]


def getSolution(image_file_path):
    yellow_balls_count = 0
    red_balls_count = 0
    purple_balls_count = 0

    image = cv2.imread(image_file_path)

    # Crop the image using the specified rectangle points
    first_cropped_image = image[first_top_left[1]:first_bottom_right[1], first_top_left[0]:first_bottom_right[0]]

    # Crop the image using the specified rectangle points
    second_cropped_image = image[second_top_left[1]:second_bottom_right[1], second_top_left[0]:second_bottom_right[0]]

    # Crop the image using the specified rectangle points
    third_cropped_image = image[third_top_left[1]:third_bottom_right[1], third_top_left[0]:third_bottom_right[0]]


    # Display the original and cropped images
    cv2.imshow(image_file_path, image)
    cv2.imshow('First Cropped Image', first_cropped_image)
    cv2.imshow('Second Cropped Image', second_cropped_image)
    cv2.imshow('Third Cropped Image', third_cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return [first_cropped_image, second_cropped_image, third_cropped_image]



def getContourImage(contour, shape):
    height, width = shape

    # Create an empty mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Draw the contour on the mask
    cv2.drawContours(mask, [contour], 0, (255), thickness=cv2.FILLED)

    # Display the result
    # cv2.imshow('Result Image', result_image)
    cv2.imshow('Mask Image', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return mask


def isSimilar(mask1, mask2):
    # Ensure the masks have the same dimensions
    if mask1.shape != mask2.shape:
        raise ValueError("The mask images must have the same dimensions.")

    # Calculate the Intersection over Union (IoU)
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)

    # Print the IoU value
    print(f"Intersection over Union (IoU): {iou}")


    cv2.imshow('mask1', mask1)
    cv2.imshow('mask2', mask2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # You can set a threshold to determine if the masks are similar
    iou_threshold = 0.2  # Adjust as needed
    if iou > iou_threshold:
        print("The masks are similar.")
        return True
    else:
        print("The masks are not similar.")
        return False

def checkForDuplicate(front_image, back_image):
    # Create a mirrored version of the image
    mirrored_back_image = cv2.flip(back_image, 1) # 1 for horizontal flip, 0 for vertical flip, -1 for both

    # Display the original and mirrored images
    cv2.imshow('front_image', front_image)
    cv2.imshow('back_image', back_image)
    cv2.imshow('mirrored_back_image', mirrored_back_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    yellow_contours_front, red_contours_front, purple_contours_front = getContours(front_image)
    yellow_contours_back, red_contours_back, purple_contours_back = getContours(mirrored_back_image)

    count = 0
    for i, contour_front in enumerate(red_contours_front):
        contour_image_front = getContourImage(contour_front, [110, 110])

        for i, contour_back in enumerate(red_contours_back):
            contour_image_back = getContourImage(contour_back, [110, 110])

            if isSimilar(contour_image_front, contour_image_back):
                count += 1

    for i, contour_front in enumerate(yellow_contours_front):
        contour_image_front = getContourImage(contour_front, [110, 110])

        for i, contour_back in enumerate(yellow_contours_back):
            contour_image_back = getContourImage(contour_back, [110, 110])

            if isSimilar(contour_image_front, contour_image_back):
                count += 1

    for i, contour_front in enumerate(purple_contours_front):
        contour_image_front = getContourImage(contour_front, [110, 110])

        for i, contour_back in enumerate(purple_contours_back):
            contour_image_back = getContourImage(contour_back, [110, 110])

            if isSimilar(contour_image_front, contour_image_back):
                count += 1

    return count

# Create a CSV file
csv_file_path = 'results.csv'
with open(csv_file_path, 'w', newline='') as csvfile:
    fieldnames = ['pic', 'yellow', 'red', 'purple']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write the header row
    writer.writeheader()

    for i in range(17, 18):
        image_file_name = str(i) + '_front.jpg'
        image_file_path = '27_plant_bed_cap/color_image/' + image_file_name

        print(image_file_name, "start...")
        front_pics = getSolution(image_file_path)

        # print(res, image_file_name)

        # writer.writerow({
        #     'pic': image_file_name,
        #     'yellow': res[0],
        #     'red': res[1],
        #     'purple': res[2]
        # })
        print(image_file_name, "done!")

        image_file_name = str(i) + '_back.jpg'
        image_file_path = '27_plant_bed_cap/color_image/' + image_file_name


        print(image_file_name, "start...")
        back_pics = getSolution(image_file_path)

        # print(res, image_file_name)

        # writer.writerow({
        #     'pic': image_file_name,
        #     'yellow': res[0],
        #     'red': res[1],
        #     'purple': res[2]
        # })
        print(image_file_name, "done!")

        count = 0
        count += checkForDuplicate(front_pics[0], back_pics[2])
        count += checkForDuplicate(front_pics[1], back_pics[1])
        count += checkForDuplicate(front_pics[2], back_pics[0])

        print("count:", count)
