from __future__ import division
import cv2
import time
import numpy as np
import math

def find_shoulder_size(point1,point2):
    a = point2[0]-point1[0]
    b = point2[1]-point1[1]
    a2 =a*a
    b2= b*b
    c=a2+b2
    width = math.sqrt(c)
    return width

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return int(x),int(y)
    else:
        return False

def inlination(p1 , p2):
    if  p1 and p2!= None:
        slope = (p2[1]-p1[1])/(p2[0]-p1[0])
        angle = math.degrees(math.atan(slope))
        return angle
    else:
        return 0
    
    
protoFile = "models/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "models/pose_iter_160000.caffemodel"
nPoints = 15
POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]

unresized = cv2.imread("TestShirt.png",-1)
unresized = cv2.cvtColor(unresized,cv2.COLOR_BGR2BGRA)

inWidth = 368
inHeight = 368
threshold = 0.1

cap = cv2.VideoCapture(0)
hasFrame, frame = cap.read()

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


while cv2.waitKey(1) < 0:
    #t = time.time()
    hasFrame, frame = cap.read()
    framecopy2=cv2.cvtColor(frame,cv2.COLOR_BGR2BGRA)
    frameCopy = np.copy(frame)
    if not hasFrame:
        cv2.waitKey()
        break

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                              (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()

    H = output.shape[2]
    W = output.shape[3]
    # Empty list to store the detected keypoints
    points = []

    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        
        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > threshold : 
            # Add the point to the list if the probability is greater than the threshold
            #points.append((int(x), int(y)))
            points.append((int(x),int(y)))
        else :
            points.append(None)

    if points[8] and   points[2] and points[5] and points[11]!= None:
        cv2.line(frame, points[8], points[2], (0, 255, 255), 2)
        cv2.line(frame, points[2], points[5], (0, 255, 255), 2)
        cv2.line(frame, points[5], points[11], (0, 255, 255), 2)
        cv2.line(frame, points[11], points[8], (0, 255, 255), 2)
        ax,ay=intersection(line(points[2],points[11]),line(points[5],points[8]))
        m_width = find_shoulder_size(points[2],points[5])
        um_width = abs(m_width)
        resized =image_resize(unresized,width = int(um_width))
        
        frame_h,frame_w,frame_c=frame.shape
        overlay = np.zeros((frame_h,frame_w,4),dtype = 'uint8')
        
        resized_h ,resized_w,resized_c=resized.shape
        angle = -inlination(points[2],points[5])
        center = (resized_w/2,resized_h/2)
        size = (resized_w,resized_h)
        scale = 0.5
        rotationMatrix  = cv2.getRotationMatrix2D(center, angle, scale)
        imageRot = cv2.warpAffine(resized,rotationMatrix,size,flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_TRANSPARENT )
        imageRot_h,imageRot_w,imageRot_c=imageRot.shape
        cv2.imshow('rotimg', imageRot)
        for i in range(0,imageRot_h):
            for j in range(0,imageRot_w):                
                overlay[points[2][1] + i,points[2][0]+j]=imageRot[i,j]
                
        cv2.addWeighted(overlay,0.25,framecopy2,1.0,0,framecopy2)
        #cv2.circle(frame, (int(ax), int(ay)), 15, (0, 255, 255), thickness=-1)
        #print(ax,ay)
    cv2.imshow('Output-Skeleton', framecopy2)
    cv2.imshow('Output-Skeleton2', frame)
   
