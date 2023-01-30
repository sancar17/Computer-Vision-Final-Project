import numpy as np
import nibabel as nib
import cv2
import matplotlib.pyplot as plt

def calc_dice_score(X_seg, X_gt): #calculates dice scores

    return 2*np.count_nonzero(np.logical_and(X_seg, X_gt)) / (np.count_nonzero(X_seg) + np.count_nonzero(X_gt))

def find_8_neighbors(points, x, y): #finds 8 neighbor points of the corresponding coordinate

    result = np.array([[0,0]])
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0: 
                continue  #(0,0) distance is not neighbor
            side = np.array([i, j])
            neighbors_new_side = points + side
            result = np.vstack((result, neighbors_new_side))

    check_row = np.logical_and((result[:,0] >= 0), (result[:,0] < x))
    check_column = np.logical_and((result[:,1] >= 0), (result[:,1] < y))
    inside_the_area =  np.logical_and(check_column, check_row)

    neighbors = result[inside_the_area]

    return neighbors[1:]

def find_4_neighbors(points, x, y): #finds 4 neighbor points of the corresponding coordinate

    result = np.array([[0,0]])
    for i in range(-1, 2):
        for j in range(-1, 2):
            if abs(i)+abs(j) == 1:   #only calculating 4 sides
                side = np.array([i, j])
                neighbors_new_side = points + side
                result = np.vstack((result, neighbors_new_side))

    check_row = np.logical_and((result[:,0] >= 0), (result[:,0] < x))
    check_column = np.logical_and((result[:,1] >= 0), (result[:,1] < y))
    inside_the_area =  np.logical_and(check_column, check_row)

    neighbors = result[inside_the_area]

    return neighbors[1:]

def find_26_neighbors(points, x, y, z): #finds 26 neighbor points of the corresponding coordinate

    result = np.array([[0, 0, 0]])
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                if (i == 0 and j == 0) and k == 0: #(0,0,0) distance is not neighbor, it is the point
                    continue
                side = np.array([i, j, k ])
                neighbors_new_side = points + side
                result = np.vstack((result, neighbors_new_side))

    check_row = np.logical_and((result[:,0] >= 0), (result[:,0] < x))
    check_column = np.logical_and((result[:,1] >= 0), (result[:,1] < y))
    check_depth = np.logical_and((result[:,2] >= 0), (result[:,2] < z))
    inside_the_volume =  np.logical_and(np.logical_and(check_column, check_depth), check_row)     

    neighbors = result[inside_the_volume]

    return neighbors[1:]

def find_6_neighbors(points, x, y, z): #finds 6 neighbor points of the corresponding coordinate

    result = np.array([[0, 0, 0]])
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                if (abs(i) + abs(j) + abs(k)) != 1: #only calculating sides
                    continue
                side = np.array([i, j, k ])
                neighbors_new_side = points + side
                result = np.vstack((result, neighbors_new_side))

    check_row = np.logical_and((result[:,0] >= 0), (result[:,0] < x))
    check_column = np.logical_and((result[:,1] >= 0), (result[:,1] < y))
    check_depth = np.logical_and((result[:,2] >= 0), (result[:,2] < z))
    inside_the_volume =  np.logical_and(np.logical_and(check_column, check_depth), check_row)     

    neighbors = result[inside_the_volume]

    return neighbors[1:]

def region_growing_2D(image, seeds, neighborhood, T):
    
    x,y,z = image.shape
    done = np.zeros(image.shape)

    for seed in seeds:
        points = seed[:2].reshape(1, 2)

        while len(points)>0:
            if neighborhood == 8:
                neighbors = find_8_neighbors(points, x, y)
            elif neighborhood == 4:
                neighbors = find_4_neighbors(points, x, y)

            before = np.copy(done[:, :, seed[2]])
            
            for i in range(len(neighbors)):
                if image[neighbors[i,0], neighbors[i,1], seed[2]] > T:
                    done[neighbors[i,0], neighbors[i,1], seed[2]] = 1

                else:
                    done[neighbors[i,0], neighbors[i,1], seed[2]] = 20

            after = done[:, :, seed[2]] - before
            new_points = np.where(after == 1)
            points = np.vstack((new_points[0], new_points[1])).T

    done[(done == 20)] = 0
    return done


def region_growing_3D(image, seeds, neighborhood, T):

    x,y,z = image.shape
    done = np.zeros(image.shape)

    for seed in seeds:
        points = seed.reshape(1, 3)

        while len(points) > 0:
            if neighborhood == 26:
                neighbors = find_26_neighbors(points, x, y, z)
            elif neighborhood == 6:
                neighbors = find_6_neighbors(points, x, y, z)

            before = np.copy(done)  

            for i in range(len(neighbors)):
                if image[neighbors[i,0], neighbors[i,1], neighbors[i,2]] > T:
                    done[neighbors[i,0], neighbors[i,1], neighbors[i,2]] = 1
                
                else:
                    done[neighbors[i,0], neighbors[i,1], neighbors[i,2]] = 20

            after = done - before 
            new_points = np.where(after == 1)
            points = np.vstack((new_points[0], new_points[1], new_points[2])).T

    done[(done == 20)] = 0
    return done

#reading the image and ground truth segmentation
data = nib.load("V.nii")
image = data.get_fdata()

gt_data = nib.load("V_seg.nii")
gt_image = gt_data.get_fdata()

#defining seeds
seeds2D = np.array([[81,39,43], [80,35,44], [79,31,45], [77,23,46], [103,47,47], [103,21,48], [103,25,49], [81,24,50], [80,24,51], [78,25,52], [72,22,53], [80,38,54], [81,39,55]])
seeds3D = np.array([[79,31,45], [103,21,48], [80,24,51], [80,38,54]])

segmentation_8 = region_growing_2D(image, seeds2D, 8, 0.72)
segmentation_4 = region_growing_2D(image, seeds2D, 4, 0.72)

segmentation_26 = region_growing_3D(image, seeds3D, 26, 0.72)
segmentation_6 = region_growing_3D(image, seeds3D, 6, 0.72)

print("score of 8-neighbors: ", calc_dice_score(segmentation_8, gt_image))
print("score of 4-neighbors: ", calc_dice_score(segmentation_4, gt_image))

print("score of 26-neighbors: ", calc_dice_score(segmentation_26, gt_image))
print("score of 6-neighbors: ", calc_dice_score(segmentation_6, gt_image))