import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
import keras.utils
import keras.backend as K
from keras.optimizers import Adam, SGD
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Flatten, LSTM

default_hyperparams = {
    'threshold': {
        'value': 0.55, 'min': 0, 'max': 1, 'step': 0.01
    },
    'num_layers': {
        'value': 3, 'min': 1, 'max': 10, 'step': 1
    }
}

def deep_variable_lstm(num_inputs):
    input = Input(shape=(None, num_inputs))
    lstm1 = LSTM(64, return_sequences=True)(input)
    lstm2 = LSTM(64, return_sequences=True)(lstm1)
    lstm3 = LSTM(64)(lstm2)
    output = Dense(1, activation='sigmoid')(lstm3)

    model = Model(inputs=input, outputs = output)
    # model.add(Dense(1))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def deep_fixed_lstm(lookback, num_inputs):
    lstmInput = Input(shape=(lookback, num_inputs))
    denseInput = Input(2)

    lstm1 = LSTM(64, return_sequences=True)(lstmInput)
    lstm2 = LSTM(64, return_sequences=True)(lstm1)
    lstm3 = LSTM(64)(lstm2)

    # merged = keras.layers.Concatenate([lstm3, denseInput])
    # dense1 = Dense(6)(merged)
    output = Dense(1, activation='sigmoid')(lstm3)

    model = Model(inputs=lstmInput, outputs = output)
    # model.add(Dense(1))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def deep_dense(hyperparams, lookback, num_inputs, steps_per_epoch, output_bias=None):
    # Get hyperparameters
    num_layers = hyperparams['num_layers'].value
    threshold = hyperparams['threshold'].value

    # layer_widths = []
    # for i in range(num_layers):
    #     layer_widths.append(hyperparam.Float(f'layer_%d' % i, min_value=0.5, max_value=5, step=0.1))

    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    initial_learning_rate = 0.0005
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.9,
        staircase=True
    )

    clr_schedule = tfa.optimizers.CyclicalLearningRate(
        initial_learning_rate,
        maximal_learning_rate = initial_learning_rate * 10,
        scale_fn = lambda x: 1/(2.**(x-1)),
        step_size=steps_per_epoch
    )


    input = Input(shape=(lookback, num_inputs))
    flatten = Flatten()(input)

    # Build Model
    prev_layer = flatten
    for i in range(num_layers):
        # int(num_inputs * layer_widths[i])
        prev_layer = Dense(lookback * num_inputs * 3)(prev_layer)
    output = Dense(1, activation='sigmoid', bias_initializer=output_bias)(prev_layer)

    model = Model(inputs=input, outputs = output)
    model.compile(loss='binary_crossentropy', optimizer=SGD(learning_rate=clr_schedule),
        metrics=[
            'accuracy',
            keras.metrics.Precision(thresholds=threshold, name='precision'),
            keras.metrics.Recall(name='recall')
            ])

    keras.utils.plot_model(model, to_file='model.png', show_shapes=True)    
    return model