import moviepy.video.io.VideoFileClip as mpy
import cv2
import numpy as np
import moviepy.editor as mp
import matplotlib.pyplot as plt
import sys
import numpy
import matplotlib

numpy.set_printoptions(threshold=sys.maxsize)

def find_balls(frame_gray):

    red_mask = np.logical_and(frame_gray>59, frame_gray<90)
    green_mask = np.logical_and(frame_gray>145, frame_gray<155)
    blue_mask = np.logical_and(frame_gray>120, frame_gray<135)
    pink_mask = np.logical_and(frame_gray>210, frame_gray<230)

    red = np.zeros(frame_gray.shape)
    red[red_mask] = frame_gray[red_mask]

    green = np.zeros(frame_gray.shape)
    green[green_mask] = frame_gray[green_mask]
    
    blue = np.zeros(frame_gray.shape)
    blue[blue_mask] = frame_gray[blue_mask]

    pink = np.zeros(frame_gray.shape)
    pink[pink_mask] = frame_gray[pink_mask]
    pink[:,646:647]=0
    pink[:,73:75]=0

    return red, green, blue, pink

def std(ball):

    coords = np.where(ball>0)
    
    std = np.std(coords)

    coordinates = np.vstack((coords[0], coords[1])).T
    middle_point = np.int16(np.mean(coordinates, axis=0))

    for i ,j in coordinates:
        if (abs(i-middle_point[0])>std):
            ball[i,j] = 0

        if (abs(j-middle_point[1])>std):
            ball[i,j] = 0

    return ball


def find_V(ball, next_frame, w):

    ball = std(ball)
    coords = np.where(ball>0)

    next_ball = np.zeros(next_frame.shape)
    next_ball[coords] = next_frame[coords]

    coordinates = np.vstack((coords[0], coords[1])).T
    start_point = np.int16(np.mean(coordinates, axis=0))
    
    if (w != 0):

        x = np.zeros(ball.shape)
        y = np.zeros(ball.shape)
        t = np.zeros(ball.shape)

        t[coords] = (next_ball[coords] - ball[coords])
        x[coords] = (ball[(coords[0]+1, coords[1])] - ball[coords])
        y[coords] = (ball[(coords[0], coords[1]+1)] - ball[coords])

        i = start_point[0]
        j = start_point[1]

        Grad_x = x[i-w:i+w+1, j-w:j+w+1].flatten()
        Grad_y = y[i-w:i+w+1, j-w:j+w+1].flatten()
        Grad_t = t[i-w:i+w+1, j-w:j+w+1].flatten()

    else:
        Grad_t = (next_ball[coords] - ball[coords])
        Grad_x = (ball[(coords[0]+1, coords[1])] - ball[coords])
        Grad_y = (ball[(coords[0], coords[1]+1)] - ball[coords])        

    A = np.vstack((Grad_x, Grad_y)).T
    V = np.matmul(np.linalg.pinv(np.matmul(A.T, A)), -np.matmul(A.T, Grad_t))

    return V, start_point

def compare(balls):

    x = sorted(balls)

    for i in range(0,4):
        if balls[0] == x[i]:
            print("Red", end =" ")
        elif balls[1] == x[i]: 
            print("Green", end =" ")
        elif balls[2] == x[i]:
            print("Blue", end =" ")
        else:
            print("Pink", end =" ")
        if i != 3:
            print(" < ", end =" ")

video = mpy.VideoFileClip("movie_001.avi")
frame_count = video.reader.nframes
video_fps = video.fps

x_red = 0
x_green = 0
x_blue = 0
x_pink = 0

for i in range(frame_count):
    
    frame = video.get_frame(i*1.0/video_fps)
    next_frame = video.get_frame((i+1)*1.0/video_fps)

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    next_frame_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    red, green, blue, pink = find_balls(frame_gray)
    
    Vred, sp_red = find_V(red, next_frame_gray, 0)
    Vgreen, sp_green = find_V(green, next_frame_gray, 0)
    Vblue, sp_blue = find_V(blue, next_frame_gray, 0)
    Vpink, sp_pink = find_V(pink, next_frame_gray, 0)

    x_red += np.sqrt((Vred[0]**2)+(Vred[1]**2))
    x_green += np.sqrt((Vgreen[0]**2)+(Vgreen[1]**2))
    x_blue += np.sqrt((Vblue[0]**2)+(Vblue[1]**2))
    x_pink += np.sqrt((Vpink[0]**2)+(Vpink[1]**2))

time = frame_count/ video_fps

'''print("Red:", x_red/time)
print("Green:", x_green/time)
print("Blue:", x_blue/time)
print("Pink:", x_pink/time)'''

balls = [x_red, x_green, x_blue, x_pink]

compare(balls)