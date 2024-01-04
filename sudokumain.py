import numpy as np
import cv2 as cv
import functions as f
import tensorflow as tf
import sudokusolver as ss
import matplotlib.pyplot as plt

# Read the sudoku image:
sudoku_img = './sudokuimg/sudoku2.png'
img_orig = cv.imread(sudoku_img)

# Resize original sudoku:
img_stand = f.standard(img_orig)

# Preprocessing the original sudoku:
img_process = f.preprocess(img_stand)

# Finding a Sudoku Game in the image and wrap it:
s_cont = f.findsudokuincont(img_process)
s_wrap = f.wrap(img_stand, s_cont)

# Get each number cell and clean it:
numbers = f.split_numbers(s_wrap)

# Cut coordinates:
y0,y1,x0,x1 = 5,45,5,45
numbers2 = []

for image in numbers:
    aux_img = image[y0:y1,x0:x1]
    aux_img = cv.resize(aux_img, (50,50))
    numbers2.append(aux_img)
    
# Load the model    
model = f.load_model()

# Get sudoku array:
pred = []
for i, cropped in enumerate(numbers2):
    input_image = cv.resize(cv.cvtColor(cropped, cv.COLOR_BGR2GRAY), (28, 28))
    input_image = input_image.reshape((1, 28, 28, 1)) / 255.0
    if f.iswhite(input_image * 255):
        pred.append(0)
    else:
        prediction = np.argmax(model.predict(input_image))+1
        pred.append(prediction)
        
sudoku = np.array(pred).reshape((9,9))
sudoku_for_solution = sudoku.copy()

# Solution, and array with only complete numbers.
sol = ss.solver(sudoku_for_solution)
only_sol = f.onlysolution(sudoku, sol)

# Some coordinates to put the numbers in the image:
top_left_x, top_left_y = s_cont[0][0], s_cont[0][1]
orig_w, orig_h = s_cont[1][0] - s_cont[0][0], s_cont[2][1] - s_cont[0][1]
cell_y, cell_x = orig_w // 9, orig_h // 9

# Copy of original image
image_with_solution = img_stand.copy()

# Font type, scale, color:
font = cv.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2
font_color = (0, 255, 0)

# Printing numbers in the original image:
for i in range(9):
    for j in range(9):
        value = only_sol[i][j]
        if value != 0:
            text_size = cv.getTextSize(str(value), font, font_scale, font_thickness)[0]
            x_position = top_left_x + j * cell_x + (cell_x // 2 - text_size[0] // 2)
            y_position = top_left_y + i * cell_y + (cell_y // 2 + text_size[1] // 2)
            cv.putText(image_with_solution, str(value), (int(x_position), int(y_position)), font, font_scale, font_color, font_thickness)

# Mostrar la imagen con la solución
cv.imshow("",image_with_solution)
cv.waitKey(0)

