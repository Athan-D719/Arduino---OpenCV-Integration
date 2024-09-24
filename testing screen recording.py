import cv2
import numpy as np
import pyautogui
import time

SCREEN_SIZE = (1920, 1080)

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("screen_recorded.avi", fourcc, 20.0, (SCREEN_SIZE))
out1 = cv2.VideoCapture()

fps = 120
prev = 0

while True:
    
    print("STARTED")
    time_elapsed = time.time() - prev

    img = pyautogui.screenshot()

    if time_elapsed > 1.0/fps:
        prev = time.time()
        hsv = np.array(img)
        hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2RGB)

        out.write(hsv)
        cv2.waitKey(100)
            #For the red color its different because of the begining and ending of the color in the color chart
    #Thats why theres gonna be red one and red two
    red_low1 = np.array([0,65,75], dtype=np.uint8)
    red_high1 = np.array([12,255,255], dtype=np.uint8)
    red_low2 = np.array([240,65,75], dtype=np.uint8)
    red_high2 = np.array([256,255,255], dtype=np.uint8)


    red_mask1 = cv2.inRange(hsv, red_low1, red_high1)
    red_mask2 = cv2.inRange(hsv, red_low2, red_high2)

    #Debugging the noise on the masks
    kernel = np.ones((6,6), np.uint8) #Kernel, a matrix that as a mask is an array smaller to put it onto the masks.

    #the morph function help us filte the masks

    red_mask1 = cv2.morphologyEx(red_mask1, cv2.MORPH_CLOSE, kernel)
    red_mask1 = cv2.morphologyEx(red_mask1, cv2.MORPH_OPEN, kernel)
    red_mask2 = cv2.morphologyEx(red_mask2, cv2.MORPH_CLOSE, kernel)
    red_mask2 = cv2.morphologyEx(red_mask2, cv2.MORPH_OPEN, kernel)

    #The close is a solution for black noise pixels
    #The open is a solution for white noise pixels

    #we also need a general mask that recovers all the image
    mask = cv2.add(red_mask1, red_mask2) #permited just two parameters for the add function
    
    moment = cv2.moments(mask)
    area = moment['m00']

    moment_r1 = cv2.moments(red_mask1)
    area_r1 = moment_r1['m00']
    moment_r2 = cv2.moments(red_mask2)
    area_r2 = moment_r2['m00']

    ret, frame1 = out.read(mask)
    ret, frame2 = out.read(mask)


    diff = cv2.absdiff(frame1, frame2) # absdiff: Absulute difference method, inthis case between frame1 & frame2
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) # grayscale on the diff in order to find the contours easier
    blur = cv2.GaussianBlur(gray, (5,5), 0,)  #secon arg being the kernel size
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY) # _, because we dont need this first variable and the second is tresh, TRESHHOLD=limit
    dilated = cv2.dilate(thresh, None, iterations=3) #iterations = repetition
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #cv2.drawContours(frame1, contours, -1 ,(0, 255, 0), 2)

    #Drawig and tracking figures
    if (area < 2000):
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour) #understands the x & y coordinates w as width and h as hight

            if (cv2.contourArea(contour) < 500 or cv2.contourArea(contour) > 2000):
                continue # if the area of the any contour is less than 700 it will not recognize it as one
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
            
            # conn, addr = server.accept() #server part
            #cmnd = conn.recv(4)  # The default size of the command packet is 4 bytes
            
            #print(cmnd)
            z = str(y)
            s = str(x)
            g = 0
            j = str(g)
            if len(z) == 1 and len(s) == 1: #Case 1 EQ
                z1 = j+""+j+""+z
                s1 = j+""+j+""+s
                v = s1 +""+z1
                u = v.encode('utf-8')
            elif len(z) == 1 and len(s) == 2: #Case 1,2
                z1 = j+""+j+""+z
                s2 = j+""+s
                v = s2 +""+z1
                u = v.encode('utf-8')
            elif len(z) == 2 and len(s) == 1: #Case 2,1
                z2 = j+""+z
                s1 = j+""+j+""+s
                v = s1 +""+z2
                u = v.encode('utf-8')
            elif len(z) == 1 and len(s) == 3: #Case 1,3
                z1 = j+""+j+""+z
                s3 = s
                v = s3 +""+z1
                u = v.encode('utf-8')
            elif len(z) == 3 and len(s) == 1: #Case 3,1
                z3 = z
                s1 = j+""+j+""+s
                v = s1 +""+z3
                u = v.encode('utf-8')
            elif len(z) == 2 and len(s) == 2: #Case 2 EQ
                z2 = j+""+z
                s2 = j+""+s
                v = s2 +""+z2
                u = v.encode('utf-8')
            elif len(z) == 2 and len(s) == 3: #Case 2,3
                z2 = j+""+z
                s3 = s
                v = s3+""+z2
                u = v.encode('utf-8')
            elif len(z) == 3 and len(s) == 2: #Case 3,2
                z3 = z
                s2 = j+""+s
                v = s2+""+z3
                u = v.encode('utf-8')
            elif len(z) == 3 and len(s) == 3: #Case 3 EQ
                v = s +""+z
                u = v.encode('utf-8')
            #print(s)
            #conn.sendall(b'%s'%(c))
            
            #TCPIP
                # if 'XY' in str(cmnd):
                    # Do the play action
                    # conn.sendall(b"%s"%(u))   #TCPIP
                    # print("X,Y: %s"%(u))
    else:
        continue
            
            #if (area_r1 > 1000000 or area_r2 > 1000000):
                #ser.write(str.encode('r'))
            #    print('RED')

            #if (area > 1000000): #General
            #    print('OBJECT DETECTED')
            #    print("%s , %s" % (x, y))
            #else:
            #    print('NOTHING')

    cv2.imshow("MASKED", frame1)
    frame1 = frame2
    ret, frame2 = cap.read(mask)
   
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    #cv2.imshow('CAMERA', image)
    #cv2.imshow('MASK', mask)
    if cv2.waitKey(40) == 27 : #ESC key
        break

        # out.write(hsv)

    cv2.waitKey(1000)
    cv2.imshow("MASKED", hsv)

cv2.destroyAllWindows()
out.release()
