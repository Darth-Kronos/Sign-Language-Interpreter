import cv2, pickle  #Python pickle module is used for serializing and de-serializing a Python object structure
import numpy as np
import tensorflow as tf
import os
from keras.models import load_model
x, y, w, h = 300, 100, 250, 250
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
kernel1 = np.ones((9,9),np.uint8)
kernel2 = np.ones((7,7),np.uint8)
prediction = None
model = load_model('cnn_model_keras3.h5')
kernel = np.ones((1,1),np.uint8)
def skin_segementation(frame):
    save_img = None
    text = ""
    mask1 = None
    frame = frame[y:y+h, x:x+w]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0,0.23*255,0])
    upper_skin = np.array([60,0.68*255,255])
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask,kernel2,iterations = 1)
    #res = cv2.bitwise_and(frame,frame, mask= mask)
    blur = cv2.GaussianBlur(mask, (5,5), 0)
    mask = cv2.medianBlur(blur, 9)
    contours= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[0]
    
    if len(contours) > 0:
        contour = max(contours, key = cv2.contourArea)
        if cv2.contourArea(contour) > 10000:
            mask1 = np.zeros(mask.shape,np.uint8)
            cv2.drawContours(mask1,[contour],0,255,-1)
            x1, y1, w1, h1 = cv2.boundingRect(contour)
            save_img = mask1[y1:y1+h1, x1:x1+w1]
            if w1 > h1:
                save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))  #top, bottom, left, right 
            elif h1 > w1:                                                                                                        
                save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))
            cv2.imshow('dwd',save_img)
            cv2.imshow('mask1',mask1)
            text = output(save_img)
    return save_img,mask,text


def get_image_size():
	img = cv2.imread('gestures/0/100.jpg', 0)
	return img.shape

image_x, image_y = get_image_size()

def keras_process_image(img):
	img = cv2.resize(img, (image_x, image_y))
	img = np.array(img, dtype=np.float32)
	img = np.reshape(img, (1, image_x, image_y, 1))
	return img

def keras_predict(model, image):
	processed = keras_process_image(image)
	pred_probab = model.predict(processed)[0]
	pred_class = list(pred_probab).index(max(pred_probab))
	return max(pred_probab), pred_class

def get_pred_text(pred_class):
    alpha = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
    return alpha[pred_class]
def output(image):
    pred_probab, pred_class = keras_predict(model, image)
    text = ""
    if pred_probab*100 > 80:
        text = get_pred_text(pred_class)
    return text

def get_hand_hist():
	with open("hist", "rb") as f:
		hist = pickle.load(f)
	return hist
def histt(img,hist):
    text = ""
    save_img = None
    imgCrop = img[y:y+h, x:x+w]
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    cv2.filter2D(dst,-1,disc,dst)
    blur = cv2.GaussianBlur(dst, (11,11), 0)
    blur = cv2.medianBlur(blur, 15)
    #cv2.imshow('blur',blur)
    thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    thresh = cv2.merge((thresh,thresh,thresh))
    thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    thresh = thresh[y:y+h, x:x+w]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    if len(contours) > 0:
        contour = max(contours, key = cv2.contourArea)
        #print(cv2.contourArea(contour))
        if cv2.contourArea(contour) > 10000:
            x1, y1, w1, h1 = cv2.boundingRect(contour)
            save_img = thresh[y1:y1+h1, x1:x1+w1]
            if w1 > h1:
                save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
            elif h1 > w1:
                save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))
            text = output(save_img)
    return save_img,save_img,text
def recognize():
    text = ""
    global prediction
    cam = cv2.VideoCapture(0)
    hist = get_hand_hist()
    while True:
        img = cam.read()[1]
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))
        #image,mask,text = skin_segementation(img) 
        image,mask,text = histt(img,hist)
        blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blackboard, text, (30, 200), cv2.FONT_HERSHEY_TRIPLEX, 1.3, (255, 255, 255))
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        res = np.hstack((img, blackboard)) #function is used to stack the sequence of input arrays horizontally (i.e. column wise) to make a single array.
        cv2.imshow("Recognizing gesture", res)
        cv2.imshow("mask", mask)
        
        if cv2.waitKey(1) == ord('q'):
            break

keras_predict(model, np.zeros((50, 50), dtype=np.uint8))		
recognize()