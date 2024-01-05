import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def standard(img):
    imgh, imgw = img.shape[:2]
    imgr = imgh/imgw
    new_w = 450
    result = cv.resize(img, (new_w,int(new_w*imgr)))
    return result

def preprocess(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 1)
    thresh = cv.adaptiveThreshold(blur,255,1,1,11,2)
    return thresh    
    
def findsudokuincont(img):
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    area = []
    for contour in contours:
        area.append(cv.contourArea(contour))
        
    maxarea_i = area.index(max(area))
    x0,y0,w,h = cv.boundingRect(contours[maxarea_i])
    sudokucorners = np.float32([[x0,y0],
                     [x0+w,y0],
                     [x0,y0+h],
                     [x0+w,y0+h]])    
    return sudokucorners

def wrap(img, corners):
    w,h = 450,450
    warpdim = np.float32([[0,0],
               [w,0],
               [0,h],
               [w,h]])
    matrix = cv.getPerspectiveTransform(corners, warpdim)
    imgwarp = cv.warpPerspective(img, matrix,(h,w))   
    return imgwarp

def split_numbers(img):
    r = np.vsplit(img,9)
    num = []
    for i in r:
        c = np.hsplit(i,9)
        for n in c:
            num.append(n)            
    return num

def load_model():
    model = tf.keras.models.load_model('D:\Projects\SudokuSolver\my_model_digits_1_9.keras')
    return model

def iswhite(image, perc = 95):
    relwhite = np.sum(image>=200)/image.size * 100
    return relwhite >= perc

def onlysolution(sudoku,solution):
    solonly = []
    for i in range(9):
        for j in range(9):
            if sudoku[i][j] != 0:
                solonly.append(0)
            else:
                solonly.append(solution[i][j])
                
    return np.array(solonly).reshape((9,9))

def plot_s(cells, pred):
    fig, axs = plt.subplots(9, 9, figsize=(5, 5))

    for i in range(9):
        for j in range(9):
            index = i * 9 + j
            axs[i, j].imshow(cells[index])  # You can specify the colormap if needed
            axs[i, j].set_title(pred[index])
            axs[i, j].axis('off')  # Turn off axis labels for cleaner display
    return fig
