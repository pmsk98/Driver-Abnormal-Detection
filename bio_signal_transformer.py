# -*- coding: utf-8 -*-
"""
Created on Mon May  8 23:37:43 2023

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



bio_signal = final_df[['ecg','eeg','eeg_1','ppg','label']]

# 바이오 신호 데이터를 numpy array로 변환
ecg_data = np.array(bio_signal['ecg'].tolist())
eeg_0_data = np.array(bio_signal['eeg'].tolist())
eeg_1_data = np.array(bio_signal['eeg_1'].tolist())
ppg_data = np.array(bio_signal['ppg'].tolist())

# 모든 바이오 신호 데이터를 하나의 numpy array로 결합
data = np.stack([ecg_data, eeg_0_data, eeg_1_data, ppg_data], axis=2)


#-----------
    
# 하이퍼파라미터 설정
num_classes = 6
d_model = 256
num_heads = 16
dff = 512
input_vocab_size = data.shape[1]
maximum_position_encoding = data.shape[1]
dropout_rate = 0.3
num_features = data.shape[2]

from tensorflow.keras.layers import MultiHeadAttention
# Transformer 모델 구현
class TransformerModel(tf.keras.Model):
    def __init__(self, num_classes, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, dropout_rate):
        super(TransformerModel, self).__init__()
        self.d_model = d_model

        # Input layer
        self.input_layer = tf.keras.layers.Input(shape=(input_vocab_size, num_features))

        # Encoder
        self.encoder = tf.keras.layers.Dense(d_model, activation='relu')
        self.pos_encoding = self.positional_encoding(maximum_position_encoding,
                                                     self.d_model)

        self.encoder_layers = [self.encoder_layer(d_model, num_heads, dff, dropout_rate)
                               for _ in range(2)]
        self.encoder_dropout = tf.keras.layers.Dropout(dropout_rate)

        # Output
        self.out = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.encoder(inputs)
        seq_len = tf.shape(x)[1]
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.encoder_dropout(x)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        x = tf.reduce_mean(x, axis=1)
        return self.out(x)
        
    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)
        # apply sin to even indices in the array; 2i
        sines = np.sin(angle_rads[:, 0::2])
        # apply cos to odd indices in the array; 2i+1
        cosines = np.cos(angle_rads[:, 1::2])

        pos_encoding = np.concatenate([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, position, i, d_model):
        angles = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return position * angles
    
    def encoder_layer(self, d_model, num_heads, dff, rate):
        inputs = tf.keras.Input(shape=(None, d_model))
        attention = MultiHeadAttention(num_heads, d_model)
        attention_output = attention(inputs, inputs)
        attention_output = tf.keras.layers.Dropout(rate)(attention_output)
        attention_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output)
        outputs = tf.keras.layers.Dense(dff, activation='relu')(attention_output)
        outputs = tf.keras.layers.Dense(d_model)(outputs)
        outputs = tf.keras.layers.Dropout(rate)(outputs)
        outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention_output + outputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs)


# Transformer 모델 생성
transformer = TransformerModel(num_classes=num_classes,
                               d_model=d_model,
                               num_heads=num_heads,
                               dff=dff,
                               input_vocab_size=input_vocab_size,
                               maximum_position_encoding=maximum_position_encoding,
                               dropout_rate=dropout_rate)

#--------------------

# # 모델 컴파일
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
# metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
# transformer.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

# # 모델 학습
# transformer.fit(X_train, y_train, epochs=30, batch_size=32)

#%%
#Embadding 추출
# # 학습된 모델을 이용하여 바이오 신호 데이터의 임베딩 추출
# bio_signal_embedding = tf.keras.models.Model(inputs=transformer.input_layer, outputs=transformer.call(transformer.input_layer))

# # train 데이터셋의 임베딩 추출
# bio_signal_signal_embedding = bio_signal_embedding.predict(data)
# print('Train 데이터셋 임베딩:', train_signal_embedding.shape)
