# -*- coding: utf-8 -*-
"""
Created on Mon May  8 23:38:33 2023

@author: user
"""


#Transformer 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalMaxPooling1D
from tensorflow.keras.layers import Embedding, Conv1D, Concatenate, LayerNormalization
from tensorflow.keras import Model

#이미지 df
dataset_df = final_df[['image_path','label','eeg','ecg','eeg_1','ppg']]

from sklearn.preprocessing import LabelEncoder

# # 레이블 인코딩
# le = LabelEncoder()
# le.fit(dataset_df['label'])
# encoded_label = le.transform(dataset_df['label'])

# dataset_df['label'] = encoded_label

from sklearn.model_selection import train_test_split

train, test= train_test_split(dataset_df,test_size=0.2, random_state=42,stratify=dataset_df['label'])



#%%이미지 데이터 전처리
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_dataframe(
    dataframe=train,
    x_col='image_path',
    y_col='label',
    target_size=(224, 224),
    batch_size=64,
    class_mode='categorical',
    subset='training',shuffle=False
)


#%%
from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator(rescale=1./255)


test_generator = test_datagen.flow_from_dataframe(
    dataframe=test,
    x_col='image_path',
    y_col='label',
    target_size=(224, 224),
    batch_size=64,
    class_mode='categorical',shuffle=False
)

#%%
#Vision Transformer
from vit_keras import vit

vit_model = vit.vit_b32(
        image_size = 224,
        activation = 'softmax',
        pretrained = True,
        include_top = False,
        pretrained_top = False,
        classes = 6)

import tensorflow_addons as tfa
from tensorflow.keras import models

model_vision_transformer = models.Sequential()
model_vision_transformer.add(vit_model)
model_vision_transformer.add(layers.Flatten())
model_vision_transformer.add(layers.Dense(512, activation='relu'))
model_vision_transformer.add(layers.Dropout(0.35))
model_vision_transformer.add(layers.Dense(128, activation='relu'))
model_vision_transformer.add(layers.Dropout(0.35))
model_vision_transformer.add(layers.Dense(32, activation='relu'))
model_vision_transformer.add(layers.Dense(6, activation='softmax'))

learning_rate = 1e-4

optimizer = tfa.optimizers.RectifiedAdam(learning_rate = learning_rate)

model_vision_transformer.compile(optimizer = optimizer, 
              loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing = 0.2), 
              metrics = ['accuracy'])

with tf.device('/device:GPU:0'): 
    model_vision_transformer.fit(train_generator,
                        epochs=20,
                        steps_per_epoch = 150,
                        verbose=1)
    
#%% 이미지 모델
#Test
import numpy as np
with tf.device('/device:GPU:0'): 
    vision_transformer_predict = model_vision_transformer.predict(test_generator)


from sklearn.metrics import classification_report

vision_transformer_pred_classes = np.argmax(vision_transformer_predict,axis=1)

# classification report 출력
target_names = list(test_generator.class_indices.keys())



print(classification_report(test_generator.classes, vision_transformer_pred_classes, target_names=target_names))
