########################################################################
#
# File:   FinalProject.py
# Author: S. B. Nashed, A. L. Steele
# Date:   May 2015
#
########################################################################

import cv2
import numpy
import sys
import struct
import math

import bounce_mod

sys.path.append('../examples')
import cvk2

w = 1880
h = 1024

black = (0,0,0)
white = (255,255,255)
blue = (255,0,0)
red = (0,0,255)
yellow = (0, 255, 255)


########################################################################
# This is a helper function that reads a frame from camera, converts to grayscale
# and then check if frame is ok

def getFrame():
    ok, frame = capture.read()

    # The image is converted to grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype('uint8')

    #check if ok
    if not ok or frame is None:
        print 'No frames in video'
        sys.exit(1)

    return frame


########################################################################
# This is a helper function that takes an image, modifies it by adding
# some text, and displays it on the screen

def labelAndWaitForKey(frame, text1, text2):

    # Get the image height, and width and make a temp copy to edit text
    h = frame.shape[0]
    w = frame.shape[1]
    
    # Note that even though shapes are represented as (h, w), pixel
    # coordinates below are represented as (x, y). Confusing!
    cv2.putText(frame, "Interactive Physics-Based Sandbox", (w/4, h/6), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0,0,0), 3, cv2.CV_AA)

    cv2.putText(frame, "Interactive Physics-Based Sandbox", (w/4, h/6), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255,255,255), 1, cv2.CV_AA)

    cv2.putText(frame, text1, (w/4, 2*h/6), 
                cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                (0,0,0), 6, cv2.CV_AA)

    cv2.putText(frame, text1, (w/4, 2*h/6), 
                cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                (255,255,255), 2, cv2.CV_AA)

    cv2.putText(frame, text2, (w/4, 3*h/6), 
                cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                (0,0,0), 6, cv2.CV_AA)

    cv2.putText(frame, text2, (w/4, 3*h/6), 
                cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                (255,255,255), 2, cv2.CV_AA)

    cv2.imshow('Final Project', frame)


########################################################################
# This is a helper function that calculates a homography by mapping
# out a series of points and comparing their location in the camera frame
# and the projection frame.

def establishHomography(w,h,color,frame):
    cameraDisplayCircle = []
    calibrationProjection = []

    # array that stores the centroids of each object.
    calibrationCentroids=[]

    for i in range(9):

        x = int((1.0/6)*w + (1.0/3)*w*(i%3))
        y = int((1.0/6)*h + (1.0/3)*h*int(i/3))

        calibrationProjection.append([x,y])

        new = blank.copy()
        cv2.circle( new, ( x, y), 40, color, -1)

        cv2.imshow("Final Project",new)

        # Delay for 500ms and get a key
        k = cv2.waitKey(500)

         # Get the frame.
        ok, frame = capture.read(frame)
        #newFrame = frame - baseFrame
        newFrame = cv2.absdiff(frame,baseFrame)

        # An identity matrix of size 8x8 is created in order to determine
        # the size of the morphological transformation
        kernel = numpy.ones((8,8),numpy.uint8)
        morph = cv2.morphologyEx(newFrame,cv2.MORPH_CLOSE,kernel)

        # An identity matrix of size 5x5 is created in order to determine
        # the size of the dilation
        kernel = numpy.ones((15,15),numpy.uint8)
        morph = cv2.dilate(morph,kernel,iterations=1)

        # The image is converted to grayscale
        morph = cv2.cvtColor(morph, cv2.COLOR_RGB2GRAY).astype('uint8')

        # The image is thresholded to remove remaining noise
        cv2.threshold(morph,100,255, cv2.THRESH_BINARY,morph)

        # An identity matrix of size 8x8 is created in order to determine
        # the size of the dilation
        kernel = numpy.ones((8,8),numpy.uint8)
        morph = cv2.erode(morph,kernel,iterations=1)

        temp = morph.copy()
        # The outlines of the objects are determined
        contours = cv2.findContours(temp, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        
        # Iterates through each object found in frame.
        for j in range(len(contours[0])):

            # Compute some statistics about this contour.
            info = cvk2.getcontourinfo(contours[0][j])

            # Mean location of every objects' centroid.
            mu = info['mean']
            calibrationCentroids.append(mu)
        

        cameraDisplayCircle.append(morph)

        # Bail if none.
        if not ok or frame is None:
            break

    calibrationCentroids = numpy.asarray(calibrationCentroids, dtype='float32')
    calibrationProjection = numpy.asarray(calibrationProjection, dtype='float32')

    M = cv2.findHomography(calibrationCentroids,calibrationProjection,cv2.RANSAC)[0]

    return M

########################################################################
# This is a helper function that rectifies an image.
# The image is then warped to display the region of interest.
def rectifyImage(img, w, h, M):
    # Construct an array of points on the border of the image.
    p = numpy.array( [ [[ 0, 0 ]],
                       [[ w, 0 ]],
                       [[ w, h ]],
                       [[ 0, h ]] ], dtype='float32' )


    # Send the points through the transformation matrix.
    pp = cv2.perspectiveTransform(p, M)

    # Compute the bounding rectangle for all points (note this gives
    # integer coordinates).
    box = cv2.boundingRect(pp)

    dims = box[2:4]

     # Warp the image to the destination in the temp image.
    rectifiedImage = cv2.warpPerspective(img, M, tuple(dims))

    return rectifiedImage


########################################################################
# This is a helper function that outlines user created lines
# on the board.
def userDraw(rectifiedBase, w, h , M, sim):

    while 1:
        frame = getFrame()

        # Warp the image to the destination in the temp image.
        frame = rectifyImage(frame, w, h, M)
        
        newFrame = cv2.absdiff(rectifiedBase,frame)

        # The image is thresholded to remove remaining noise
        cv2.threshold(newFrame,35,255, cv2.THRESH_BINARY,newFrame)

        # An identity matrix of size 5x5 is created in order to determine
        # the size of the dilation
        kernel = numpy.ones((6,6),numpy.uint8)
        newFrame = cv2.erode(newFrame,kernel,iterations=1)
        newFrame = cv2.dilate(newFrame,kernel,iterations=1)

        #isolate region of interest in newFrame
        roi = newFrame[0:h,0:w]

        #update the scene in ball sim
        sim.updateScene(roi)

        #run ball sim and update ball positions
        sim.run()

       
# Try to get an integer argument:
try:
    device = int(sys.argv[1])
    del sys.argv[1]
except (IndexError, ValueError):
    device = 0

# If we have no further arguments, open the device. Otherwise, get the
# filename.
if len(sys.argv) == 1:
    capture = cv2.VideoCapture(device)
    if capture:
        print 'Opened device number', device, '- press Esc to stop capturing.'
else:
    capture = cv2.VideoCapture(sys.argv[1])
    if capture:
        print 'Opened file', sys.argv[1]

# Bail if error.
if not capture:
    print 'Error opening video capture!'
    sys.exit(1)


#display blank black screen
blank = numpy.empty(shape=(h,w))
blank.fill(0)
cv2.imshow("Final Project",blank)
cv2.moveWindow("Final Project",0,0)

rgbArray = numpy.zeros((h,w,3), 'uint8')

# Delay for .5 seconds, let camera callibrate
k = cv2.waitKey(500)

ok, frame = capture.read()
baseFrame = frame.copy()

#establish homography
M = establishHomography(w,h,white,frame)

cv2.imshow("Final Project",blank)
k = cv2.waitKey(1000)

frame = getFrame()

# Warp the image to the destination in the temp image.
rectifiedBase = rectifyImage(frame, w, h, M)


labelAndWaitForKey(rgbArray,"Final Project", "Samer Nashed & Andrew Steele")
cv2.waitKey(1000)
cv2.imshow("Final Project",blank)
k = cv2.waitKey(500)

frame = getFrame()
# Warp the image to the destination in the temp image.
frame = rectifyImage(frame, w, h, M)
newFrame = cv2.absdiff(rectifiedBase,frame)

# The image is thresholded to remove remaining noise
cv2.threshold(newFrame,30,255, cv2.THRESH_BINARY,newFrame)

# An identity matrix of size 6x6 is created in order to determine
# the size of the erosion and dilation
kernel = numpy.ones((6,6),numpy.uint8)

#image is eroded and dilated to get rid of noise
newFrame = cv2.erode(newFrame,kernel,iterations=1)
newFrame = cv2.dilate(newFrame,kernel,iterations=1)

#isolate region of interest in newFrame
roi = newFrame[0:h,0:w]

#create BallSim object
sim = bounce_mod.BallSim(roi,5)
userDraw(rectifiedBase, w, h , M, sim)



