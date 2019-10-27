import numpy as numP
from decimal import Decimal
from PIL import Image
import glob
import xlrd
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from numpy.linalg import inv
import gc
import random
import math

numP.seterr(all='ignore')
flat_arr_images_X_matrix=[]#all images in a folder for trainning X
flat_arr_images_Y1_matrix=[]#all images in a folder for trainning predict Y
flat_arr_images_Y2_matrix=[]#all images in a folder for trainning predict Y
flat_arr_images_XY_matrix = []
n=0#iteration of scans
tData = pd.read_excel(r"C:\Users\mhossa12\Desktop\DataScience\finalProj\train.xlsx", sheet_name='Sheet1')
print("Column headings:", tData.columns)
for filename in glob.glob(r"C:\Users\mhossa12\Desktop\DataScience\finalProj\*.JPG"):
    im=Image.open(filename).resize((56,56)).convert('RGBA')#resize to smallest due memroy and matrix issue bestfit 64，64
        #print(filename)
        #arr = numP.array(im)#print(arr.size)
        # flat_arr = arr.ravel()# print(flat_arr)
    flat_arr_images_X_matrix.append(numP.array(im).ravel())#  here we can add bias, 
    flat_arr_images_Y1_row = []#read row in array struc
    flat_arr_images_Y2_row = []#read row in array struc
    flat_arr_images_Y1_row.append(tData[tData.columns[0]][n])
    flat_arr_images_Y2_row.append(tData[tData.columns[1]][n])
    flat_arr_images_Y1_matrix.append(flat_arr_images_Y1_row)
    flat_arr_images_Y2_matrix.append(flat_arr_images_Y2_row)
    n=n+1
    # applying algo1 lin_reg, this to find training Beta at first. inverse((X_Trans * X))* X_Trans * Y
    flat_arr_images_X_matrix = numP.matrix(flat_arr_images_X_matrix,dtype='float64')
    x_rowSize = flat_arr_images_X_matrix[0].size
    flat_arr_images_Y1_matrix = numP.matrix(flat_arr_images_Y1_matrix,dtype='float64')
    flat_arr_images_Y2_matrix = numP.matrix(flat_arr_images_Y2_matrix,dtype='float64')
    Transform_flat_arr_images_X_matrix = flat_arr_images_X_matrix.T
    #print(flat_arr_images_X_matrix.shape,'\n')print(flat_arr_images_X_matrix,'\n')
    #print(flat_arr_images_Y_matrix.shape,'\n')print(flat_arr_images_Y_matrix,'\n')
    #print(Transform_flat_arr_images_X_matrix,'\n')    
    dot_flat_arr_images_XandY_matrix= Transform_flat_arr_images_X_matrix * flat_arr_images_X_matrix
##    print(dot_flat_arr_images_XandY_matrix.shape,'\n')
##    print(dot_flat_arr_images_XandY_matrix[0].size,'\n')
##    print(dot_flat_arr_images_XandY_matrix[0,0],'\n')
##    print(dot_flat_arr_images_XandY_matrix[16383,16383],'\n')
    for i in range(dot_flat_arr_images_XandY_matrix[0].size):
        for j in range(dot_flat_arr_images_XandY_matrix[0].size):
              dot_flat_arr_images_XandY_matrix[i,j] =dot_flat_arr_images_XandY_matrix[i,j]+random.uniform(0, 1)#add w
    #print(flat_arr_images_X_matrix.shape,'\n')print(flat_arr_images_X_matrix.dtype,'\n')
    #print(flat_arr_images_X_matrix,'\n')  
    inverse_dot_flat_arr_images_XandY_matrix = dot_flat_arr_images_XandY_matrix .I 
    #print(inverse_dot_flat_arr_images_XandY_matrix)
    beta_Training_NN_Xt = inverse_dot_flat_arr_images_XandY_matrix * Transform_flat_arr_images_X_matrix
    beta_Training1 = beta_Training_NN_Xt * flat_arr_images_Y1_matrix 
    beta_Training2 = beta_Training_NN_Xt * flat_arr_images_Y2_matrix
    print("Beta1:",beta_Training1,beta_Training1.shape,"\nBeta2:",beta_Training2,beta_Training2.shape,'\n')
# getting trainning model then to validate 
    test_image_X_matrix=[]
    test_image_Y1_matrix=[]
    test_image_Y2_matrix=[]
    n=0
    test_Data = pd.read_excel(r"C:\Users\mhossa12\Desktop\DataScience\finalProj\train.xlsx", sheet_name='test')
    for filename in glob.glob(r"C:\Users\mhossa12\Desktop\DataScience\finalProj\test\t1.*"):
        im=Image.open(filename).resize((56,56)).convert('RGBA') #resize to smallest due memroy and matrix issue
        test_image_X_matrix.append(numP.array(im).ravel())#  here we can add bias,
        test_image_Y1_row = []#read row in array struc
        test_image_Y2_row = []#read row in array struc        
        test_image_Y1_row.append(test_Data[test_Data.columns[0]][n])
        test_image_Y2_row.append(test_Data[test_Data.columns[1]][n])
        test_image_Y1_matrix.append(test_image_Y1_row)
        test_image_Y2_matrix.append(test_image_Y2_row)
        n=n+1
    test_image_X_matrix = numP.matrix(test_image_X_matrix,dtype='float64')
    test_image_Y1_matrix = numP.matrix(test_image_Y1_matrix,dtype='float64')
    test_image_Y2_matrix = numP.matrix(test_image_Y2_matrix,dtype='float64')
    print("test_image:",test_image_X_matrix,test_image_X_matrix.shape,'\n')
    predict_image_Y1_matrix =  test_image_X_matrix * beta_Training1
    predict_image_Y2_matrix =  test_image_X_matrix * beta_Training2
    print("predict_value_Y1 carbenicillin resistance:",predict_image_Y1_matrix , ",\tpredict_value_Y2 tobramycin resistance:",predict_image_Y2_matrix ,'\n')
    print("actual_value_Y1 carbenicillin resistance:",test_image_Y1_matrix[0,0],",\tactual_value_Y2 tobramycin resistance:",test_image_Y2_matrix[0,0],'\n')    
    print("accuracy :", (test_image_Y1_matrix[0,0]-predict_image_Y1_matrix)/test_image_Y1_matrix[0,0]*100,"%,",
          (test_image_Y2_matrix[0,0]-predict_image_Y2_matrix)/test_image_Y2_matrix[0,0]*100,"%" )

