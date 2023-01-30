import pyautogui
import numpy as np
from PIL import Image
import cv2

def sobel_filter(image):
    
    image = image.astype("float32") / 255

    blur_filter = np.ones((3,3)) / 9
    image = cv2.filter2D(image, -1, np.float32(blur_filter))

    vertical_filter = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
    horizontal_filter = np.array([[-1,-2,-1], [0,0,0], [1,2,1]]) 

    vertical = cv2.filter2D(image, -1, np.float32(vertical_filter))
    horizontal = cv2.filter2D(image, -1, np.float32(horizontal_filter))

    edge_magnitude = np.sqrt(np.square(vertical) + np.square(horizontal))
    edge_magnitude = (edge_magnitude / edge_magnitude.max()) * 255

    return np.uint8(edge_magnitude)

def seperating_dice(image):

    height, width = image.shape[0], image.shape[1]
    
    a = image.max()
    threshold = 18
    image[image<a-threshold] = 0
    image[image>a-threshold] = 255

    die1 = image[0:int(height/2), 0:int(width/3)]
    die2 = image[0:int(height/2), int(width/3):int(2*width/3)]
    die3 = image[0:int(height/2), int(2*width/3):width]

    return die1,die2,die3

def circle(image, i):
    
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 30, param1=300, param2=8, minRadius=30, maxRadius=35)

    if circles is not None:
        return circles.shape[1]
    else:
        return 0
    
while(1):
    myScreenshot = pyautogui.screenshot()
    myScreenshot.save('game.png')

    image = np.array(Image.open('game.png'))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = sobel_filter(gray)

    die1, die2, die3 = seperating_dice(np.array(edges))

    d1, d2, d3 = circle(die1, 1), circle(die2, 2), circle(die3, 3)

    if (d1<7 and d1>0) and (d2<7 and d2>0) and (d3<7 and d3>0):

        if d1>d2 and d1>d3:
            pyautogui.press('a')
        elif d2>d1 and d2>d3:
            pyautogui.press('s')
        else:
            pyautogui.press('d')