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

################################# read all test image and predict column 
numP.seterr(all='ignore')
flat_arr_images_X_matrix=[]#all images in a folder for trainning X
flat_arr_images_Y1_matrix=[]#all data in a folder for trainning predict Y
flat_arr_images_Y2_matrix=[]#all data in a folder for trainning predict Y
flat_arr_images_XY_matrix=[]
beta_initial =[0]*12544
beta_tenXn = []
n=0#iteration of scans
tData = pd.read_excel(r"C:\Users\mhossa12\Desktop\DataScience\finalProj\train.xlsx", sheet_name='Sheet1')
print("Column:", tData.columns)
for filename in glob.glob(r"C:\Users\mhossa12\Desktop\DataScience\finalProj\*.JPG"):
    im=Image.open(filename).resize((56,56)).convert('RGBA')#resize to smallest due memroy and matrix issue bestfit 64，64
        #print(filename)
        #arr = numP.array(im)#print(arr.size)
        # flat_arr = arr.ravel()# print(flat_arr)
        flat_arr_images_X_matrix.append(numP.array(im).ravel()/255)#  here we can add bias, 
        flat_arr_images_Y1_row = []#read row in array struc
        flat_arr_images_Y2_row = []#read row in array struc
        flat_arr_images_Y1_row.append(tData[tData.columns[6]][n])
        flat_arr_images_Y2_row.append(tData[tData.columns[7]][n])
        flat_arr_images_Y1_matrix.append(flat_arr_images_Y1_row)
        flat_arr_images_Y2_matrix.append(flat_arr_images_Y2_row)
        n=n+1
    # applying algo1 lin_reg, this to find training Beta at first. inverse((X_Trans * X))* X_Trans * Y
    beta_initial = numP.matrix(beta_initial,dtype='float64')
    flat_arr_images_X_matrix = numP.matrix(flat_arr_images_X_matrix,dtype='float64')
    flat_arr_images_Y1_matrix = numP.matrix(flat_arr_images_Y1_matrix,dtype='float64')
    flat_arr_images_Y2_matrix = numP.matrix(flat_arr_images_Y2_matrix,dtype='float64')
    x_rowSize = flat_arr_images_X_matrix[0].size
    print(beta_initial,beta_initial[0,12543],"row: ",len(beta_initial),"shape: ",beta_initial.shape)        
    print(flat_arr_images_X_matrix[0],flat_arr_images_X_matrix[0,12543],"row: ",len(flat_arr_images_X_matrix),"shape: ",flat_arr_images_X_matrix.shape,"\n")
################################# read all trainning image and predict column
    test_image_X_matrix=[]
    test_image_Y1_matrix=[]
    test_image_Y2_matrix=[]
    n=0
    test_Data = pd.read_excel(r"C:\Users\mhossa12\Desktop\DataScience\finalProj\train.xlsx", sheet_name='test')
    for filename in glob.glob(r"C:\Users\mhossa12\Desktop\DataScience\finalProj\test\t1.*"):
        im=Image.open(filename).resize((56,56)).convert('RGBA') #resize to smallest due memroy and matrix issue
        test_image_X_matrix.append(numP.array(im).ravel()/255)#  here we can add bias,
        test_image_Y1_row = []#read row in array struc
        test_image_Y2_row = []#read row in array struc        
        test_image_Y1_row.append(test_Data[test_Data.columns[3]][n])
        test_image_Y2_row.append(test_Data[test_Data.columns[4]][n])
        test_image_Y1_matrix.append(test_image_Y1_row)
        test_image_Y2_matrix.append(test_image_Y2_row)
        n=n+1
    test_image_X_matrix = numP.matrix(test_image_X_matrix,dtype='float64')
    test_image_Y1_matrix = numP.matrix(test_image_Y1_matrix,dtype='float64')
    test_image_Y2_matrix = numP.matrix(test_image_Y2_matrix,dtype='float64')
    print(test_image_X_matrix,test_image_X_matrix[0,12543],"row: ",len(test_image_X_matrix),"shape: ",test_image_X_matrix.shape)        
    print(test_image_Y1_matrix[0],test_image_Y1_matrix[0,0],"row: ",len(test_image_Y1_matrix),"shape: ",test_image_Y1_matrix.shape)
    print(test_image_Y2_matrix[0],test_image_Y2_matrix[0,0],"row: ",len(test_image_Y2_matrix),"shape: ",test_image_Y2_matrix.shape,"\n")  
################################# prepare build Log_Reg model, read image and predict label.
    Gradient_Xi_to_Y1=[]
    Gradient_Xi_to_Y2=[]
    beta_tenXn = []
    prev_step_size=[]
    precision_tenXn = []
    for i in range(10):
        y1=(flat_arr_images_Y1_matrix[i,0]-(1/(1+math.exp(-beta_initial *(flat_arr_images_X_matrix[i].T) )))*flat_arr_images_X_matrix[i])
        y2=(flat_arr_images_Y2_matrix[i,0]-(1/(1+math.exp(-beta_initial *(flat_arr_images_X_matrix[i].T) )))*flat_arr_images_X_matrix[i])
        Gradient_Xi_to_Y1.append(y1)
        Gradient_Xi_to_Y2.append(y2)
        x=[0]*12544
        y=[1]*12544
        z=[0.0005]*12544
        beta_tenXn.append(x)
        prev_step_size.append(y)
        precision_tenXn.append(z)
    Gradient_Xi_to_Y1 = numP.asarray(Gradient_Xi_to_Y1)
    Gradient_Xi_to_Y2 = numP.asarray(Gradient_Xi_to_Y2)
    Gradient_Xi_to_Y1 =numP.matrix(Gradient_Xi_to_Y1,dtype='float64')
    Gradient_Xi_to_Y2 =numP.matrix(Gradient_Xi_to_Y2,dtype='float64') 
    beta_tenXn = numP.matrix(beta_tenXn,dtype='float64')
    prev_step_size = numP.matrix(prev_step_size,dtype='float64')
    precision_tenXn = numP.matrix(precision_tenXn,dtype='float64')
    print(beta_tenXn[0],beta_tenXn[0][0][0,0],beta_tenXn[0][0][0,12543])
    print(beta_tenXn[0].size,"row: ",len(beta_tenXn),"\n")
    print(Gradient_Xi_to_Y1[0],Gradient_Xi_to_Y1[0][0][0,0],Gradient_Xi_to_Y1[0][0][0,12543])
    print(Gradient_Xi_to_Y1[0].size,"row: ",len(Gradient_Xi_to_Y1),"\n")
    print(Gradient_Xi_to_Y2[0],Gradient_Xi_to_Y2[0][0][0,0],Gradient_Xi_to_Y2[0][0][0,12543])
    print(Gradient_Xi_to_Y2[0].size,"row: ",len(Gradient_Xi_to_Y2),"\n")
################################# implement gradient ascent on initial beta and max likelihood
    cur_x = beta_tenXn # The algorithm starts at x=0
    rate = 0.001 # Learning rate
    precision = precision_tenXn #This tells us when to stop the algorithm
    previous_step_size = prev_step_size #
    max_iters = x_rowSize # maximum number of iterations
    iters = 0 #iteration counter
    print(type(cur_x),cur_x[0].size,cur_x[0],"row: ",len(cur_x))
    print(type(precision),precision[0].size,precision[0,0],"row: ",len(precision))
    print(type(previous_step_size),previous_step_size[0].size,previous_step_size[0,0],"row: ",len(previous_step_size))
    while(previous_step_size[0][0][0,0] > precision[0][0][0,0] )and( iters < 2000): # gradient ascent to find MLE
        prev_x = cur_x #Store current x value in prev_x
        cur_x = cur_x + rate * Gradient_Xi_to_Y1 #Grad ascent
        previous_step_size = abs(cur_x - prev_x) #Change in x
        iters = iters+1 #iteration count                          # loop thru the colunm value
        print("Iteration",iters,",\nX value is",cur_x,'\n') #Print iterations
    max_predict=[]
    for i in range(10):
        betaT = cur_x[i]
##        print(betaT.shape)
##        print((test_image_X_matrix.T).shape)
##        print(betaT * (test_image_X_matrix.T))
        predict_y1 = 1/(1+math.exp( -betaT * (test_image_X_matrix.T)  ) + random.uniform(0, 1))# add bias
        max_predict.append(predict_y1)
        print("predict image: ",i,"\t value: ",predict_y1)
    print("MAX predict carbenicillin resistance:",max(max_predict), ",\tMAX predict tobramycin resistance:",1-max(max_predict) ,'\n')
    print("actual_value_Y1 carbenicillin resistance:",test_image_Y1_matrix[0,0],",\tactual_value_Y2 tobramycin resistance:",test_image_Y2_matrix[0,0],'\n')    
    print("accuracy :", max(max_predict)*100,"%,\t",-(1-max(max_predict))*100,"%" )

