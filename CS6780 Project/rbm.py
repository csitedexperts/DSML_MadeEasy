from __future__ import print_function
import numpy as np
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
class RBM:
  
  def __init__(self, num_visible, num_hidden):
    self.num_hidden = num_hidden
    self.num_visible = num_visible
    self.debug_print = True

    # Initialize a weight matrix, of dimensions (num_visible x num_hidden), using
    # a uniform distribution between -sqrt(6. / (num_hidden + num_visible))
    # and sqrt(6. / (num_hidden + num_visible)). One could vary the 
    # standard deviation by multiplying the interval with appropriate value.
    # Here we initialize the weights with mean 0 and standard deviation 0.1. 
    # Reference: Understanding the difficulty of training deep feedforward 
    # neural networks by Xavier Glorot and Yoshua Bengio
    np_rng = np.random.RandomState(1234)

    self.weights = np.asarray(np_rng.uniform(
			low=-0.1 * np.sqrt(3136. / (num_hidden + num_visible)),
                       	high=0.1 * np.sqrt(3136. / (num_hidden + num_visible)),
                       	size=(num_visible, num_hidden)))


    # Insert weights for the bias units into the first row and first column.
    self.weights = np.insert(self.weights, 0, 0, axis = 0)
    self.weights = np.insert(self.weights, 0, 0, axis = 1)

  def train(self, data, max_epochs = 1000, learning_rate = 0.1):
    """
    Train the machine.
    Parameters
    ----------
    data: A matrix where each row is a training example consisting of the states of visible units.    
    """

    num_examples = data.shape[0]

    # Insert bias units of 1 into the first column.
    data = np.insert(data, 0, 1, axis = 1)

    for epoch in range(max_epochs):      
      # Clamp to the data and sample from the hidden units. 
      # (This is the "positive CD phase", aka the reality phase.)
      pos_hidden_activations = np.dot(data, self.weights)      
      pos_hidden_probs = self._logistic(pos_hidden_activations)
      pos_hidden_probs[:,0] = 1 # Fix the bias unit.
      pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
      # Note that we're using the activation *probabilities* of the hidden states, not the hidden states       
      # themselves, when computing associations. We could also use the states; see section 3 of Hinton's 
      # "A Practical Guide to Training Restricted Boltzmann Machines" for more.
      pos_associations = np.dot(data.T, pos_hidden_probs)

      # Reconstruct the visible units and sample again from the hidden units.
      # (This is the "negative CD phase", aka the daydreaming phase.)
      neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
      neg_visible_probs = self._logistic(neg_visible_activations)
      neg_visible_probs[:,0] = 1 # Fix the bias unit.
      neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
      neg_hidden_probs = self._logistic(neg_hidden_activations)
      # Note, again, that we're using the activation *probabilities* when computing associations, not the states 
      # themselves.
      neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

      # Update weights.
      self.weights += learning_rate * ((pos_associations - neg_associations) / num_examples)

      error = np.sum((data - neg_visible_probs) ** 2)
      if self.debug_print:
        print("Epoch %s: error is %s" % (epoch, error))

  def run_visible(self, data):
    """
    Assuming the RBM has been trained (so that weights for the network have been learned),
    run the network on a set of visible units, to get a sample of the hidden units.
    
    Parameters
    ----------
    data: A matrix where each row consists of the states of the visible units.
    
    Returns
    -------
    hidden_states: A matrix where each row consists of the hidden units activated from the visible
    units in the data matrix passed in.
    """
    
    num_examples = data.shape[0]
    
    # Create a matrix, where each row is to be the hidden units (plus a bias unit)
    # sampled from a training example.
    hidden_states = np.ones((num_examples, self.num_hidden + 1))
    
    # Insert bias units of 1 into the first column of data.
    data = np.insert(data, 0, 1, axis = 1)

    # Calculate the activations of the hidden units.
    hidden_activations = np.dot(data, self.weights)
    # Calculate the probabilities of turning the hidden units on.
    hidden_probs = self._logistic(hidden_activations)
    # Turn the hidden units on with their specified probabilities.
    hidden_states[:,:] = hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
    # Always fix the bias unit to 1.
    # hidden_states[:,0] = 1
  
    # Ignore the bias units.
    hidden_states = hidden_states[:,1:]
    return hidden_states
    
  # TODO: Remove the code duplication between this method and `run_visible`?
  def run_hidden(self, data):
    """
    Assuming the RBM has been trained (so that weights for the network have been learned),
    run the network on a set of hidden units, to get a sample of the visible units.
    Parameters
    ----------
    data: A matrix where each row consists of the states of the hidden units.
    Returns
    -------
    visible_states: A matrix where each row consists of the visible units activated from the hidden
    units in the data matrix passed in.
    """

    num_examples = data.shape[0]

    # Create a matrix, where each row is to be the visible units (plus a bias unit)
    # sampled from a training example.
    visible_states = np.ones((num_examples, self.num_visible + 1))

    # Insert bias units of 1 into the first column of data.
    data = np.insert(data, 0, 1, axis = 1)

    # Calculate the activations of the visible units.
    visible_activations = np.dot(data, self.weights.T)
    # Calculate the probabilities of turning the visible units on.
    visible_probs = self._logistic(visible_activations)
    # Turn the visible units on with their specified probabilities.
    visible_states[:,:] = visible_probs > np.random.rand(num_examples, self.num_visible + 1)
    # Always fix the bias unit to 1.
    # visible_states[:,0] = 1

    # Ignore the bias units.
    visible_states = visible_states[:,1:]
    return visible_states
    
  def daydream(self, num_samples):
    """
    Randomly initialize the visible units once, and start running alternating Gibbs sampling steps
    (where each step consists of updating all the hidden units, and then updating all of the visible units),
    taking a sample of the visible units at each step.
    Note that we only initialize the network *once*, so these samples are correlated.
    Returns
    -------
    samples: A matrix, where each row is a sample of the visible units produced while the network was
    daydreaming.
    """

    # Create a matrix, where each row is to be a sample of of the visible units 
    # (with an extra bias unit), initialized to all ones.
    samples = np.ones((num_samples, self.num_visible + 1))

    # Take the first sample from a uniform distribution.
    samples[0,1:] = np.random.rand(self.num_visible)

    # Start the alternating Gibbs sampling.
    # Note that we keep the hidden units binary states, but leave the
    # visible units as real probabilities. See section 3 of Hinton's
    # "A Practical Guide to Training Restricted Boltzmann Machines"
    # for more on why.
    for i in range(1, num_samples):
      visible = samples[i-1,:]

      # Calculate the activations of the hidden units.
      hidden_activations = np.dot(visible, self.weights)      
      # Calculate the probabilities of turning the hidden units on.
      hidden_probs = self._logistic(hidden_activations)
      # Turn the hidden units on with their specified probabilities.
      hidden_states = hidden_probs > np.random.rand(self.num_hidden + 1)
      # Always fix the bias unit to 1.
      hidden_states[0] = 1

      # Recalculate the probabilities that the visible units are on.
      visible_activations = np.dot(hidden_states, self.weights.T)
      visible_probs = self._logistic(visible_activations)
      visible_states = visible_probs > np.random.rand(self.num_visible + 1)
      samples[i,:] = visible_states

    # Ignore the bias units (the first column), since they're always set to 1.
    return samples[:,1:]        
      
  def _logistic(self, x):
    return 1.0 / (1 + np.exp(-x))

if __name__ == '__main__':
    flat_arr_images_X_matrix=[]#all images in a folder for trainning X
    flat_arr_images_Y1_matrix=[]#all images in a folder for trainning predict Y
    flat_arr_images_Y2_matrix=[]#all images in a folder for trainning predict Y
    flat_arr_images_XY_matrix = []
    test_x_matrix = []
    maxX = 255
    maxY = 30
    n=0#iteration of scans
    bias = []
    tData = pd.read_excel(r"C:\Users\sheng\Desktop\DataScience\finalProj\train.xlsx", sheet_name='Sheet1')
    print("Column headings:", tData.columns)
    for filename in glob.glob(r"C:\Users\sheng\Desktop\DataScience\finalProj\*.JPG"):
        im=Image.open(filename).resize((28,28)).convert('RGBA')#resize to smallest due memroy and matrix issue bestfit 64，64
        #print(filename)
        #arr = numP.array(im)#print(arr.size)
        # flat_arr = arr.ravel()# print(flat_arr)
        flat_arr_images_X_matrix.append(np.array(im).ravel())#  here we can add bias, 
        flat_arr_images_Y1_row = []#read row in array struc
        flat_arr_images_Y2_row = []#read row in array struc
        flat_arr_images_Y1_row.append(tData[tData.columns[0]][n])
        flat_arr_images_Y2_row.append(tData[tData.columns[1]][n])
        flat_arr_images_Y1_matrix.append(flat_arr_images_Y1_row)
        flat_arr_images_Y2_matrix.append(flat_arr_images_Y2_row)
        n=n+1
    for filename in glob.glob(r"C:\Users\sheng\Desktop\DataScience\finalProj\test\t1.*"):
        im=Image.open(filename).resize((28,28)).convert('RGBA')#resize to smallest due memroy and matrix issue bestfit 64，64
        test_x_matrix.append(np.array(im).ravel())
    print(test_x_matrix)
    
    # applying algo1 lin_reg, this to find training Beta at first. inverse((X_Trans * X))* X_Trans * Y
    flat_arr_images_X_matrix = np.matrix(flat_arr_images_X_matrix,dtype='float64')
    test_x_matrix = np.matrix(test_x_matrix,dtype='float64') /255
    T_test_x_matrix = np.array(test_x_matrix)
    print(T_test_x_matrix)
    print(T_test_x_matrix.shape)
    flat_arr_images_Y1_matrix = np.matrix(flat_arr_images_Y1_matrix,dtype='float64')
    flat_arr_images_Y2_matrix = np.matrix(flat_arr_images_Y2_matrix,dtype='float64')
    Transform_flat_arr_images_X_matrix = flat_arr_images_X_matrix.T
    dot_flat_arr_images_XandY_matrix= Transform_flat_arr_images_X_matrix * flat_arr_images_X_matrix /65025
    for i in range(dot_flat_arr_images_XandY_matrix[0].size):
        for j in range(dot_flat_arr_images_XandY_matrix[0].size):
              dot_flat_arr_images_XandY_matrix[i,j] =dot_flat_arr_images_XandY_matrix[i,j]+random.uniform(0, 1)#add w
    inverse_dot_flat_arr_images_XandY_matrix = dot_flat_arr_images_XandY_matrix .I
    inverse_dot_flat_arr_images_XandY_matrix = np.array(inverse_dot_flat_arr_images_XandY_matrix)
    #print(inverse_dot_flat_arr_images_XandY_matrix.ndim)
##    print(inverse_dot_flat_arr_images_XandY_matrix[0,0])
    #print(inverse_dot_flat_arr_images_XandY_matrix[0].size)
    #print(inverse_dot_flat_arr_images_XandY_matrix[0,3135])
##    print(inverse_dot_flat_arr_images_XandY_matrix[3135,3135])
    print(inverse_dot_flat_arr_images_XandY_matrix,'\n')

    r = RBM(num_visible = 3136, num_hidden = 1)
    training_data = inverse_dot_flat_arr_images_XandY_matrix
    r.train(training_data, max_epochs = 10000)
    print(r.weights)
    user = T_test_x_matrix
    print(r.run_visible(user))
