# -*- coding: utf-8 -*-
"""
Created on Mon May  8 23:38:57 2023

@author: user
"""


#%%123
bio_signal_embedding = tf.keras.models.Model(inputs=transformer.input_layer, outputs=transformer.call(transformer.input_layer))


bio_signal = train[['ecg','eeg','eeg_1','ppg','label']]

# 바이오 신호 데이터를 numpy array로 변환
ecg_data = np.array(bio_signal['ecg'].tolist())
eeg_0_data = np.array(bio_signal['eeg'].tolist())
eeg_1_data = np.array(bio_signal['eeg_1'].tolist())
ppg_data = np.array(bio_signal['ppg'].tolist())

# 모든 바이오 신호 데이터를 하나의 numpy array로 결합
data = np.stack([ecg_data, eeg_0_data, eeg_1_data, ppg_data], axis=2)


train_bio_signal_embedding = bio_signal_embedding.predict(data)


bio_signal_test = test[['ecg','eeg','eeg_1','ppg','label']]

# 바이오 신호 데이터를 numpy array로 변환
ecg_data = np.array(bio_signal_test['ecg'].tolist())
eeg_0_data = np.array(bio_signal_test['eeg'].tolist())
eeg_1_data = np.array(bio_signal_test['eeg_1'].tolist())
ppg_data = np.array(bio_signal_test['ppg'].tolist())

# 모든 바이오 신호 데이터를 하나의 numpy array로 결합
test_data = np.stack([ecg_data, eeg_0_data, eeg_1_data, ppg_data], axis=2)

test_bio_signal_embedding = bio_signal_embedding.predict(test_data)



#%%

train_image_vit = vit_model.predict(train_generator)

test_image_vit = vit_model.predict(test_generator)


#%%
train_bio_signal_embedding
train_image_vit

import numpy as np

train_multi_modal_feature = np.concatenate([train_image_vit, train_bio_signal_embedding], axis=1)

from keras.utils import to_categorical

train_classes = to_categorical(train_classes, num_classes=6)

from tensorflow.keras import layers, models

model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(train_multi_modal_feature.shape[1],)))
model.add(layers.Dropout(0.35))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.35))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(6, activation='softmax'))

learning_rate = 1e-4

optimizer = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate)

model.compile(optimizer=optimizer,
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
              metrics=['accuracy'])



model.fit(train_multi_modal_feature,train_classes,
          epochs=20,
          batch_size=64,
          verbose=1)


#%%
#TEST
test_multi_modal_feature = np.concatenate([test_image_vit, test_bio_signal_embedding], axis=1)

from keras.utils import to_categorical

test_classes = to_categorical(test_generator.classes, num_classes=6)


with tf.device('/device:GPU:0'): 
    predict = model.predict(test_multi_modal_feature)


from sklearn.metrics import classification_report

y_pred_classes = np.argmax(predict,axis=1)

# classification report 출력
target_names = list(test_generator.class_indices.keys())


print(classification_report(test_generator.classes, y_pred_classes, target_names=target_names))
