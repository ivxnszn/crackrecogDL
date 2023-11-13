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
import cv2


time1 = datetime.now()
image_folder = '/Users/ivanbelgacem/Desktop/Coding/imageML/IMAGES3'

import cv2
import numpy as np
import os
import glob

def adjust_contrast(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Applique une égalisation de l'histogramme adaptative pour réduire le contraste excessif.
    """
    # Conversion de l'image en LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # Séparation des canaux
    l, a, b = cv2.split(lab)
    # Création d'un objet CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    # Application de CLAHE sur le canal L (Luminance)
    cl = clahe.apply(l)
    # Fusion des canaux en retour
    limg = cv2.merge((cl, a, b))
    # Conversion en BGR color space
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def reduce_pixellation(image):
    """
    Réduit la pixellisation par un flou et une interpolation.
    """
    # Floutage léger pour réduire le bruit
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    # Redimensionnement pour réduire la pixellisation, avec interpolation
    height, width = image.shape[:2]
    resized = cv2.resize(blur, (2 * width, 2 * height), interpolation=cv2.INTER_CUBIC)
    # Floutage pour lisser l'image redimensionnée
    smoothed = cv2.GaussianBlur(resized, (5, 5), 0)
    return smoothed
def sharpen_image(image):
    """
    Applique un filtre de netteté à l'image.
    """
    # Création d'un kernel de sharpening plus accentué
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
    # Application du kernel à l'image
    sharpened = cv2.filter2D(image, -1, kernel_sharpening)
    return sharpened


def edge_enhancement(image):
    """
    Améliore les contours de l'image.
    """
    # Conversion de l'image en gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Détection des contours
    edges = cv2.Canny(gray, 100, 200)
    # Conversion de nouveau en BGR
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # Combinaison des contours avec l'image originale
    image_enhanced = cv2.addWeighted(image, 0.8, edges_colored, 0.2, 0)
    return image_enhanced


def unsharp_mask(image, sigma=1.0, strength=1.5):
    """
    Applique un masque flou inversé à l'image.
    """
    # Floutage de l'image
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    # Application du masque flou inversé
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return sharpened


# [Inclure toutes les fonctions précédentes ici (adjust_contrast, reduce_pixellation, etc.)]

def process_image(image):
    # Applique les différentes améliorations
    sharpened = sharpen_image(image)
    edge_enhanced = edge_enhancement(sharpened)
    unsharped = unsharp_mask(edge_enhanced)
    contrast_adjusted = adjust_contrast(unsharped)
    denoised = reduce_pixellation(contrast_adjusted)
    return denoised

def process_images_in_folder(folder_path):
    """
    Applique le traitement d'image à toutes les images du dossier spécifié.
    """
    # Vérification de l'existence du dossier
    if not os.path.exists(folder_path):
        print("Le dossier spécifié n'existe pas.")
        return

    # Construction du chemin pour toutes les images du dossier
    images_path = os.path.join(folder_path, '*.jpg')  # Assurez-vous que le format correspond à vos images

    for image_file in glob.glob(images_path):
        # Lecture de l'image
        image = cv2.imread(image_file)

        # Vérification si l'image a été chargée correctement
        if image is None:
            print(f"Erreur lors de la lecture de l'image: {image_file}")
            continue

        # Traitement de l'image
        processed_image = process_image(image)

        # Nom du fichier de sortie
        file_name = os.path.basename(image_file)
        output_file = os.path.join('C/Users/ivanbelgacem/Desktop/Coding/imageML/go', file_name)

        # Création du dossier de sortie s'il n'existe pas
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Sauvegarde de l'image traitée
        cv2.imwrite(output_file, processed_image)
        print(f"L'image traitée a été sauvegardée: {output_file}")

# Chemin vers le dossier contenant les images
folder_path = '/Users/ivanbelgacem/Desktop/Coding/imageML/IMAGES3'


#process_images_in_folder(folder_path)



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

try:
    loaded_model = tf.keras.models.load_model('my_modelAUGMENTEDlast10nov.h5')
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
            # Add code for image segmentation here
            # Find contours in the heatmap

            print(prediction)
            return "Positive (CRACK)"
        else:
            print(prediction)
            return "Negative (CLEAN)"
    else:
        return "Image loading failed"

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
    print(result)

    if 'Positive' in result:
        print(f"{bcolors.OKGREEN}Crack Detected for Image: {image_file} {bcolors.ENDC}")
        x += 1
    else:
        print(f"{bcolors.FAIL}No Crack for Image: {image_file} {bcolors.ENDC}")
        y += 1 

print(f"\n{bcolors.OKGREEN}Found {x} crack image{bcolors.ENDC}")
print(f"{bcolors.FAIL}Found {y} non-cracked image{bcolors.ENDC}")
    
time2 = datetime.now()

print(f"\n{bcolors.BOLD}Elapsed running time {time2-time1}{bcolors.ENDC}")
