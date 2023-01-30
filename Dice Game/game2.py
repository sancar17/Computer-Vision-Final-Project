import pyautogui
import numpy as np
from PIL import Image
import cv2
import time

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
    threshold = 100
    image[image<a-threshold] = 0

    die1 = image[0:int(height/2), 0:int(width/3)]
    die2 = image[0:int(height/2), int(width/3):int(2*width/3)]
    die3 = image[0:int(height/2), int(2*width/3):width]

    return die1,die2,die3

def order_points(square):

    pts = np.zeros((square.shape[0], 2))
    pts[:,0] = square[:,0,0]
    pts[:,1] = square[:,0,1]
    
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def find_square(img_gray):
    cnts = cv2.findContours(img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    square = max(cnts, key=cv2.contourArea)

    x, y = int(np.sqrt(cv2.contourArea(square))), int(np.sqrt(cv2.contourArea(square)))
        
    corners = np.float32(order_points(square))
    trans_points = np.float32(([0,0], [0,y], [x,y], [x,0]))

    t_matrix = cv2.getPerspectiveTransform(corners, trans_points)

    transformed_image = cv2.warpPerspective(img_gray, t_matrix, (x,y))

    return np.uint8(transformed_image)

def circle(image, i):
    
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 30, param1=300, param2=10, minRadius=30, maxRadius=35)

    if circles is not None:
        return circles.shape[1]
    else:
        return 0

while(1):

    time.sleep(1)
    myScreenshot = pyautogui.screenshot()
    myScreenshot.save('game.png')

    image = np.array(Image.open('game.png'))

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    edges = sobel_filter(gray)

    die1gray, die2gray, die3gray = seperating_dice(edges)

    trans1 = find_square(die1gray)
    trans2 = find_square(die2gray)
    trans3 = find_square(die3gray)

    d1, d2, d3 = circle(trans1, 1), circle(trans2, 2), circle(trans3, 3)

    if (d1<7 and d1>0) and (d2<7 and d2>0) and (d3<7 and d3>0):

        if d1>d2 and d1>d3:
            pyautogui.press('a')
        elif d2>d1 and d2>d3:
            pyautogui.press('s')
        else:
            pyautogui.press('d')