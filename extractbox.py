import cv2 
import numpy as np
from glob import glob
import os
import re

count = 1

def detectMarkers(src):
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    parameters =  cv2.aruco.DetectorParameters_create()
    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(src, dictionary, parameters=parameters) 
    detectedMarkers = src.copy()
    cv2.aruco.drawDetectedMarkers(detectedMarkers, markerCorners, markerIds)
    # cv2.imshow('Markers', detectedMarkers)
    # cv2.waitKey()
    return markerCorners, markerIds

def compute_source_points(markerCorners, markerIds):
    d = dict(enumerate(markerIds.flatten(), 1))
    key_list = list(d.keys())
    val_list = list(d.values()) 

    #30,31,32,33 are id's of markers in the form
    source_points = np.array([markerCorners[key_list[val_list.index(30)]-1][0][0],
                        markerCorners[key_list[val_list.index(31)]-1][0][1],
                        markerCorners[key_list[val_list.index(33)]-1][0][2],
                        markerCorners[key_list[val_list.index(32)]-1][0][3]], dtype=np.float32)
    
    return source_points 

def compute_dest_points():
    height = 1344
    width = 896
    dest_points = np.array([(0,0),
                           (width,0),
                           (width,height),
                           (0,height)], dtype=np.float32)
    return dest_points, height, width

def detectForm(src, source_points, dest_points, height, width):
    H = cv2.getPerspectiveTransform(source_points, dest_points)
    detected_form = cv2.warpPerspective(src,H,  (width, height))
    # cv2.imshow("detected form", detected_form)
    # cv2.waitKey()
    return detected_form

def writecells(pic, detected_form, width, height):
    global count
    form1 = re.match(r'/home/alireza/phase2/dataset/[0-9]+_1[a-d]', pic)
    form2 = re.match(r'/home/alireza/phase2/dataset/[0-9]+_2[a-d]', pic)
    w = int(width/14) 
    h = int(height/21) 
    dest_points = np.array([(0,0),(w,0),(w,h),(0,h)], dtype=np.float32)
    #x,y is the coordinate of topleft corner of each cell
    x = 0 
    y = 0
    form1_labels = ['0', '1'] + [str(i) for i in range(17)] + ['2', '3']
    form2_labels = ['4', '5'] + [str(i) for i in range(17,32)] + ['6', '7', '8', '9']
    for j in range(21):
        for i in range(14):
            if not((i == 0 or i == 1 or i == 12 or i == 13) and (j == 0 or j == 1 or j == 19 or j == 20)):
                sourc_points = np.array([(x+i*w,y+j*h), (x+(i+1)*w,y+j*h), (x+(i+1)*w,y+(j+1)*h), (x+i*w,y+(j+1)*h)], dtype=np.float32)
                H = cv2.getPerspectiveTransform(sourc_points, dest_points)
                cell = cv2.warpPerspective(detected_form,H,  (h,w))
                width = 64
                height = 64
                dsize = (width, height)
                temp = cell.copy()
                temp = temp[6:58,6:58]
                cell = cv2.resize(temp, dsize)
                if form1 and (j in [0,1,19,20]):
                    os.chdir('/home/alireza/phase2/dataset/digit_folders/' + form1_labels[j] + '/')
                    cv2.imwrite(str(count) + '.jpg' , cell)            
                    count += 1
                elif form1:
                    os.chdir('/home/alireza/phase2/dataset/letter_folders/' + form1_labels[j] + '/')
                    cv2.imwrite(str(count) + '.jpg' , cell)            
                    count += 1
                elif form2 and (j in [0,1,17,18,19,20]):
                    os.chdir('/home/alireza/phase2/dataset/digit_folders/' + form2_labels[j] + '/')
                    cv2.imwrite(str(count) + '.jpg' , cell)            
                    count += 1
                elif form2:
                    os.chdir('/home/alireza/phase2/dataset/letter_folders/' + form2_labels[j] + '/')
                    cv2.imwrite(str(count) + '.jpg' , cell)            
                    count += 1

def main():
    for pic in glob('/home/alireza/phase2/dataset/*.*'):
        I = cv2.imread(pic)
        markerCorners, markerIds = detectMarkers(I)
        source_points = compute_source_points(markerCorners, markerIds)
        dest_points, height, width = compute_dest_points()
        detected_form = detectForm(I, source_points, dest_points, height, width)
        writecells(pic, detected_form, width, height)

   
    
    
if __name__ == '__main__':
    main()
    