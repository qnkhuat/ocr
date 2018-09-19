import cv2
import numpy as np



def nothing(x):
    pass
# Creating a window for later use
cv2.namedWindow('result')

# Starting with 100's to prevent error while masking
h,s,v = 100,100,100

# Creating track bar
cv2.createTrackbar('h_l', 'result',0,255,nothing)
cv2.createTrackbar('s_l', 'result',0,255,nothing)
cv2.createTrackbar('v_l', 'result',0,255,nothing)

cv2.createTrackbar('h_u', 'result',0,360,nothing)
cv2.createTrackbar('s_u', 'result',0,255,nothing)
cv2.createTrackbar('v_u', 'result',0,255,nothing)


while(1):

    frame = cv2.imread('../resources/test_images/doc/output.jpg')
    #converting to HSV
    hsv = cv2.cvtColor(frame,cv2.COLOR_RGB2HSV)

    # get info from track bar and appy to result
    h_l = cv2.getTrackbarPos('h_l','result')
    s_l = cv2.getTrackbarPos('s_l','result')
    v_l = cv2.getTrackbarPos('v_l','result')

    h_u = cv2.getTrackbarPos('h_u','result')
    s_u = cv2.getTrackbarPos('s_u','result')
    v_u = cv2.getTrackbarPos('v_u','result')

    # Normal masking algorithm
    lower_blue = np.array([h_l,s_l,v_l])
    upper_blue = np.array([h_u,s_u,v_u])

    result = cv2.inRange(hsv,lower_blue, upper_blue)

    # result = cv2.bitwise_and(frame,frame,mask = mask)
    result =cv2.resize(result,(500,600))

    cv2.imshow('result',~result)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()

cv2.destroyAllWindows()
