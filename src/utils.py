import tensorflow as tf #We need a framework called "Tensorflow"
from tensorflow.keras.layers import Lambda #We need "lambda" Layer inside "keras.layers" in Tensorflow 
from tensorflow.keras.regularizers import l1,l2,l1_l2 #We have seen this "Regularization" which is "L1 regularizer and L2 regularizer" and combination of "L1 and L2"
from kerastuner import HyperModel  #In "kerastuner" we have one thing called "HyperModel"that is were we have Important.We are "Tuning" the models here Now
import kerastuner as kt # Keras Tuner is a Framework in Tensorflow 
import json
import os 
from config import  *
import pandas as pd
import numpy as np


#Here Iam building "Keras Tuner" Model.This "Keras Tuner" Model for Neural Network
class MyHyperModel(HyperModel):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        
#Applying  input ML Model to "Keras-Tuner" for Neural network
    def build(self, hp):
        inputs = tf.keras.Input(shape=INPUT_SHAPE) #Giving Input layer to the Neural Network
        MIN_VALUE = FINE_TUNE_LAYERS['MIN_VAL']
        MAX_VALUE = FINE_TUNE_LAYERS['MAX_VAL']
        flatten = tf.keras.layers.Flatten()
        x = flatten(inputs)
        MIN_VAL = FC_LAYERS['MIN_VAL']
        MAX_VAL = FC_LAYERS['MAX_VAL']
        #Giving How many "Number of hidden" Layers and How many "Number of nodes" in each hidden layer as an Hyper-Parameter.In Each Experiment it takes one number
        #from 1 to 10 as we mentioned in "confif.py" and applys and perform experiment on it.It experiments Randomly from 1 to 20 values/Layers.In below "hp" is the
        #Hyper-parameter.
        for i in range(hp.Int('num_layers', MIN_VAL, MAX_VAL)): # "num_layers" and "MIN_VAL" and "MAX_VAL" which we have defined in "config.py" inside "FC_LAYERS"
            layer = tf.keras.layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=128,
                                            max_value=4096,
                                            step=32))
            x = layer(x)
            x = tf.keras.layers.GaussianNoise(stddev = hp.Float('gaussian_noise_'+str(i),
                                                                min_value = 0.1,
                                                                max_value = 0.9,
                                                                sampling = 'log'))(x)
            if MODEL_CONF['BATCH_NORM'] == True:
                batch_norm = tf.keras.layers.BatchNormalization()
                x = batch_norm(x)
            x = tf.keras.activations.relu(x)
            if MODEL_CONF['REGULARIZATION'] == True:
                gaussian_dropout = tf.keras.layers.GaussianDropout(rate=hp.Float('gaussian_dropout_'+str(i),
                                                                                 min_value = 0.1,
                                                                                 max_value = 0.9,
                                                                                 sampling = 'log'))
                x = gaussian_dropout(x)
            dropout = tf.keras.layers.Dropout(0.2)
            x = dropout(x)
            
        if len(N_LABELS) == 2:
            prediction_layer = tf.keras.layers.Dense(1,activation='sigmoid')
            outputs = prediction_layer(x) #Here Prediction Layer is our Output Layer
            model = tf.keras.Model(inputs,outputs)
            model.compile(loss = tf.keras.losses.BinaryCrossentropy(from_logits=False),
                    optimizer = "adam",
                    metrics = ['accuracy'])
            
            return model
        elif len(N_LABELS) > 2:
            prediction_layer = tf.keras.layers.Dense(N_LABELS,activation='softmax')
            outputs = prediction_layer(x) #Here Prediction Layer is our Output Layer
            model = tf.keras.Model(inputs,outputs)
            model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                    optimizer = "adam",
                    metrics = ['accuracy'])
            
            return model
    

def prepare_dataset():
    data = pd.read_csv(DATAPATH)
    x_columns = list(set(data.columns)-set(LABEL_NAME))
    y_columns = [LABEL_NAME]

    x_data = data[x_columns].to_numpy()
    y_data = data[y_columns].to_numpy()

    return x_data,y_data