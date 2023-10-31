import pandas as pd
import numpy as np
import plotly.express as px
import tensorflow as tf 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
from keras.models import Sequential
from keras.layers import Input
from keras.layers import Conv1D, Conv2D
from keras.layers import MaxPool1D, MaxPool2D
from keras.layers import Flatten
from tensorflow.keras.preprocessing import image
import os 
from datetime import datetime

time1 = datetime.now()


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'



try :
    loaded_model = tf.keras.models.load_model('my_modelUPDATE.h5')
except:
    print('Error importing the module')


def predict_image_label(model, input_image_path):
    img_array = None  # Initialize img_array
    try:
        # Load and preprocess the test image
        img = image.load_img(input_image_path, target_size=(227, 227))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Rescale to match training data preprocessing
        print('Successfully loaded the image in the algorithm')
    except Exception as e:
        print(f'Error: {e}')

    # Check if img_array is defined before making a prediction
    if img_array is not None:
        # Make a prediction
        prediction = model.predict(img_array)

        # Assuming it's binary classification, '1' indicates positive and '0' indicates negative
        if prediction > 0.5:
            return "Positive (CRACK)"
        else:
            return "Negative (CLEAN)"
    else:
        return "Image loading failed"
    

import os

# Define the folder containing the images
image_folder = '/Users/ivanbelgacem/Desktop/Coding/imageML/IMAGES2'

# Get a list of all files in the folder
image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
x=0
y=0
# Loop through the image files
for image_file in image_files:
    # Generate the full path to the image file
    image_path = os.path.join(image_folder, image_file)
    
    # Call the predict_image_label function
    result = predict_image_label(loaded_model, image_path)

    if 'Positive' in result :
        print(f"{bcolors.OKGREEN}Crack Detected for Image : {image_file} {bcolors.ENDC}")
        x += 1
    else:
        print(f"{bcolors.FAIL }No Crack for Image :{image_file} {bcolors.ENDC}")
        y +=1 

print(f"\n{bcolors.OKGREEN}Found {x} crack image{bcolors.ENDC}")
print(f"{bcolors.FAIL}Found {y} non cracked image{bcolors.ENDC}")
    
    
    # Print the result for each image
    #print(f'Result for {image_file}: {result}')



time2 = datetime.now()

print(f"\n{bcolors.BOLD}Elapsed running time {time2-time1}{bcolors.ENDC}")

