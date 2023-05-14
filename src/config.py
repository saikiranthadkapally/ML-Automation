import os
INPUT_SHAPE = (14,)
FINE_TUNE_LAYERS = {
        'MIN_VAL':1,
        'MAX_VAL':15
    }
FC_LAYERS = {
        #"Number of hidden Layers" from minimum value "1" to maximum value "10".We can change our "Max value" to any number of layers according to Us.It means we are 
        #improving our search as we increases the Number of Layers.So, Here number of layers is 1 to 20.
        'MIN_VAL':1,
        'MAX_VAL':10 #We can keep even 200, etc..
    }

MODEL_CONF = {
    "BATCH_NORM":True,
    "REGULARIZATION":True
}
OUTPUT_FOLDER = "../assets/output"
INPUT_FOLDER = "../assets/input"
N_LABELS = 2
DATAPATH = os.path.join(INPUT_FOLDER,"heart.csv")
LABEL_NAME  = "output"
