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

#---------------------------------------------------------------------------------DATA----------------------------------------------------------------------------------------------
#Classification en positive(avec fissure) et négative(sans fissure)

positive_fissure_dir = Path('/Users/ivanbelgacem/Desktop/Coding/imageML/archive/Positive')
negative_fissure_dir = Path('/Users/ivanbelgacem/Desktop/Coding/imageML/archive/Negative')


#Generate Dataframe

def generate_df(image_dir, label):
    filepaths = pd.Series(list(image_dir.glob(r'*.jpg')), name='Filepath').astype(str)
    lables = pd.Series(label, name='Label', index=filepaths.index)
    df = pd.concat([filepaths, lables], axis=1)#Retourne un dataframe avec l'image et le nom de label sur une autre colonne
    return df

positive_df = generate_df(positive_fissure_dir,label="CRACK")
negative_df = generate_df(negative_fissure_dir,label="CLEAN")

df = pd.concat([positive_df, negative_df], axis=0).sample(frac=1.0, random_state=1).reset_index(drop=True)




train_df,test_df = train_test_split(df.sample(20000,random_state=98),train_size=0.7,shuffle=True,random_state=89)

train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,  # Randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.1,  # Randomly zoom image 
    width_shift_range=0.1,  # Randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # Randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # Randomly flip images horizontally
    vertical_flip=False  # Randomly flip images vertically
) #Création d'une liste d'image train

test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255) #Créationd d'une liste d'image pour le test

train_data = train_gen.flow_from_dataframe(
    train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(227,277),
    color_mode='rgb',
    class_mode='binary',
    batch_size=64,
    shuffle=True,
    seed=96,
    subset='training'
)

val_data = train_gen.flow_from_dataframe(
    train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(227, 227),
    color_mode='rgb',
    class_mode='binary',
    batch_size=64,
    shuffle=True,
    seed=96,
    subset='validation'
)

test_data = train_gen.flow_from_dataframe(
    test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(227, 227),
    color_mode='rgb',
    class_mode='binary',
    batch_size=64,
    shuffle=False,
    seed=96,
)

#-------------------------------------------------------------------------------MODEL-----------------------------------------------------------------------------------------------
#Model part


inputs = tf.keras.Input(shape=(227,227,3))
x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', name='conv1')(inputs)
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu', name='conv2')(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', name='conv3')(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dense(200, activation='relu')(x)
x = tf.keras.layers.Dense(200, activation='relu')(x)

outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1,
    mode='min',
    restore_best_weights=True
)

history = model.fit(train_data, validation_data=val_data, epochs=20, callbacks=[early_stopping])
print(history)


model.save("my_modelAUGMENTEDlast10nov.h5")

