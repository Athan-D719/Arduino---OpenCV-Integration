import cv2
import numpy as np
import serial

img1 = cv2.imread('C:/Users/Asus/OneDrive/Comp/Programación Soft/Python/opencv/cv2_1/captured_frame.png', -1) #Loads a color image
ser = serial.Serial('COM3', 9600)
# cv2.imshow("ORIGINAL", img1)




hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)

green_low = np.array([76,255,200], dtype=np.uint8) #Maximum and Minimum values for each color
green_high = np.array([77, 255, 255], dtype=np.uint8)
# green_low = np.array([112,202,0], dtype=np.uint8)
# green_high = np.array([112,202,0], dtype=np.uint8)


green_mask = cv2.inRange(hsv, green_low, green_high)

#Debugging the noise on the masks
kernel = np.ones((6,6), np.uint8) #Kernel, a matrix that as a mask is an array smaller to put it onto the masks.

#the morph function help us filte the masks

green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

#The close is a solution for black noise pixels
#The open is a solution for white noise pixels

#we also need a general mask that recovers all the image
# mask = cv2.add(green_mask) #permited just two parameters for the add function

moment = cv2.moments(green_mask)
area = moment['m00']

res = cv2.bitwise_and(img1,img1,mask = green_mask)


gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY) # grayscale on the diff in order to find the contours easier
blur = cv2.GaussianBlur(gray, (5,5), 0,)  #secon arg being the kernel size
_, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY) # _, because we dont need this first variable and the second is tresh, TRESHHOLD=limit
dilated = cv2.dilate(thresh, None, iterations=3) #iterations = repetition
contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

if (True):
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour) #understands the x & y coordinates w as width and h as hight

            # if (cv2.contourArea(contour) < 500 or cv2.contourArea(contour) > 2000):
            if (cv2.contourArea(contour) < 2000 or cv2.contourArea(contour) > 2180):
                continue # if the area of the any contour is less than 700 it will not recognize it as one
            cv2.rectangle(res, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(res, "Status: {}, [{}]".format('Movement', w*h), (10, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

cv2.imshow("MASKED", res)
cv2.waitKey(2000) # One second before showing the next image



k = cv2.waitKey(0) & 0xFF#waits till... mask for 64x
if k == 27: #if someone presses the skip key(skip value)
    cv2.destroyAllWindows() #Collapses all the windows at the end
elif k == ord('s'): #elif someone presses the 's' key
                    #saves the copied image
    cv2.imwrite('lena_copy.png',img1) #'image_name' - write the image in a file(imwrite)
                                  #creates the new file
