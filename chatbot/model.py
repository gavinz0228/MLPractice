import os
import numpy as np
import nltk
from keras.models import Sequential, Model
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent, Embedding,Dropout, Bidirectional, Input, merge, Flatten, Reshape
from keras.preprocessing.sequence import pad_sequences
import keras
import json


def create_model(x_vocab_len, x_max_len, y_vocab_len, y_max_len, hidden_size, num_layers):
    model = Sequential()

    # Creating encoder network
    model.add(Embedding(x_vocab_len, 1024, input_length=x_max_len, mask_zero=True))
    model.add(Bidirectional(GRU(hidden_size)))
    model.add(Dropout(0.2))
    model.add(RepeatVector(y_max_len))

    # Creating decoder network
    for _ in range(num_layers):
        model.add(GRU(hidden_size * 2, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(y_vocab_len)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])
    return model
    
def attention(inputs, time_step):
    flatten = Flatten()(inputs)
    repeat = RepeatVector(time_step)(flatten)
    att_prob = Dense(int(inputs.shape[-1]), activation = 'sigmoid')(repeat)
    output_att = merge([inputs, att_prob ], mode='mul')
    return output_att

def create_model_with_attention(x_vocab_len, x_max_len, y_vocab_len, y_max_len, embedding_size, hidden_size, num_layers):

    inputs = Input(shape = (x_max_len,))
    # Creating encoder network
    embedding = Embedding(x_vocab_len, embedding_size, input_length=x_max_len, mask_zero=False,)(inputs)
    encoder = Bidirectional(LSTM(hidden_size, return_sequences = True))(embedding)
    attention_mul = attention(encoder, x_max_len)
    last_layer = attention_mul
    # Creating decoder network
    for _ in range(num_layers):
        last_layer = LSTM(hidden_size * 2, return_sequences=True)(last_layer)
    
    one_hot_output = TimeDistributed(Dense(y_vocab_len))(last_layer)
    activation_output = Activation('softmax')(one_hot_output)
    model = Model(inputs = inputs, outputs = activation_output)
    
    model.compile(loss='categorical_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])
    return model
