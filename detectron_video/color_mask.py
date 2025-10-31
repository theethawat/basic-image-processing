import cv2
import numpy as np

def binary_color_mask(image, color="blue"):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])

    lower_red = np.array([10, 120, 70])
    upper_red = np.array([190, 255, 255])

    lower_green = np.array([40, 150, 0])
    upper_green = np.array([80, 255, 255])

    if color == "blue":
        mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    elif color == "red":
        mask = cv2.inRange(hsv_image, lower_red, upper_red)
    elif color == "green":
        mask = cv2.inRange(hsv_image, lower_green, upper_green)
    else:
        raise ValueError("Color must be 'blue', 'red', or 'green'")

    result = cv2.bitwise_and(image, image, mask=mask)
    return mask, result
