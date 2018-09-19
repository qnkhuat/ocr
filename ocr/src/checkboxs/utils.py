import cv2 

def normalize(img):
    if len(img.shape)> 2 :
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img,(28,28))
    img = img * (1. / 255) - 0.5
    # img = img /255.
    

    return img

