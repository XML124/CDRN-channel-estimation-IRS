from __future__ import print_function
import datetime
import time
import keras
import scipy.io as io
import numpy as np
import os
import scipy.io as sio
import matplotlib.pyplot as plt
from keras.regularizers import l2
from keras.layers import Dense, Dropout, Activation, Flatten, Subtract, Input, BatchNormalization
from keras import backend as K
from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.utils import np_utils
from keras.utils import plot_model
from keras.optimizers import SGD, RMSprop, Adam
import mat73

# loss function
def NMSE_IRS(y_true, y_pred):
    r_true = y_true[:,:,:,0] #shape: (batchsize, 8, 8)
    i_true = y_true[:,:,:,1]
    r_pred = y_pred[:,:,:,0]
    i_pred = y_pred[:,:,:,1]
    mse_r_sum = K.sum(K.sum(K.square(r_pred - r_true), -1), -1)
    r_sum = K.sum(K.sum(K.square(r_true), -1), -1)
    mse_i_sum = K.sum(K.sum(K.square(i_pred - i_true), -1), -1)
    i_sum = K.sum(K.sum(K.square(i_true), -1), -1)
    num = mse_r_sum + mse_i_sum
    den = r_sum + i_sum
    return num/den


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('NMSE'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_NMSE'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('NMSE'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_NMSE'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('NMSE-loss')
        plt.legend(loc="upper right")
        plt.title('Training and Validation loss')
        plt.savefig('loss curve.png')
        plt.show()


def train_model(model, train, test, epochs, batch_size):
    # X-data
    xx_train = train[0]
    xx_train = xx_train.astype('float32')
    X_train = xx_train

    xx_test = test[0]
    xx_test = xx_test.astype('float32')
    X_test = xx_test

    # Y-data
    yy_train = train[1]
    yy_train = yy_train.astype('float32')
    Y_train = yy_train

    yy_test = test[1]
    yy_test = yy_test.astype('float32')
    Y_test = yy_test

    print('x_train shape:', X_train.shape)
    print('y_train shape:', Y_train.shape)
    print('x_test shape:', X_test.shape)
    print('y_test shape:', Y_test.shape)
    print(X_train.shape[0], 'input samples')
    print(Y_train.shape[0], 'label samples')

    model.compile(loss=NMSE_IRS,
                  optimizer=Adam(0.001),
                  metrics=[NMSE_IRS])

    history = LossHistory()

    callbacks_list = [
        # this callback will stop the training when there is no improvement in the validation loss
        keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='min'),
        # save checkpoint
        keras.callbacks.ModelCheckpoint(
            filepath='Model_210106.h5',  # 文件路径
            verbose=1,
            save_best_only=True,
            mode='min'),
        history]

    start_time = time.clock()

    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_data=(X_test, Y_test),
              callbacks=callbacks_list)

    # print training time cost
    end_time = time.clock()
    elapsed_time_train = end_time - start_time
    print(elapsed_time_train)

    # serialize model to JSON
    model_json = model.to_json()
    with open("Model_210106.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.load_weights('Model_210106.h5')
    model_best = model
    print("Saved best model to disk")

    # print evaluating time cost
    start_time = time.clock()
    score = model_best.evaluate(X_test, Y_test, verbose=1)
    end_time = time.clock()
    elapsed_time = end_time - start_time
    print('Inference time for each sample:', elapsed_time / 20000)


    print('Test best epoch NMSE:', score[1])
    history.loss_plot('epoch')


    # Prediction. the output is saved as .mat files.
    residual_predict = []
    for test_sample in X_test: #X_test shape: (20000, 8, 8, 2)  test_sample shape:(8, 8, 2)
        test_sample = test_sample[np.newaxis, :] #shape: (1,8,8,2)
        y = model_best.predict(test_sample)  #output shape: (1,8,8,2)
        residual = test_sample - y
        residual_ = np.squeeze(residual) #output shape: (8,8,2)
        residual_predict.append(residual_)  #output shape: (20000,8,8,2)

    predict_np = np.array(residual_predict)
    io.savemat('Model_CDRN_20dB_Residual_Prediction210106.mat', {'data': predict_np})
    print('prediction saved to .mat file')


def DnCNN_MultiBlock(block, depth, image_channels, filters=64,  use_bnorm=True):

    input = Input(shape=(None, None, image_channels))
    input_ = input
    x = input_

    B = block

    while B:

        input_ = x
        #For depth-1 layers, Conv+BN+relu
        for i in range(depth-1):

            x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal', \
                       padding='same', use_bias=False, data_format='channels_last')(input_)

            if use_bnorm:
                x = BatchNormalization(axis=3, momentum=0.0, epsilon=0.0001)(x)

            x = Activation('relu')(x)

        #For last layer, Conv
        x = Conv2D(filters=image_channels, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal', \
                   padding='same', use_bias=False, data_format='channels_last')(x)

        x = Subtract()([input_, x])  # input - noise

        B -= 1

    model = Model(inputs=input, outputs=x)
    return model


def main():

    batch_size = 64
    epochs = 400

    model = DnCNN_MultiBlock(block=3, depth=16, image_channels=2, use_bnorm=True)
    model.summary()

    #load .mat data
    data_xtr = sio.loadmat('x_train_Rician_CSCG_K10dB_60000_M8_N32_5dB.mat')
    data_ytr = sio.loadmat('y_train_Rician_CSCG_K10dB_60000_M8_N32_5dB.mat')
    data_xtest = sio.loadmat('x_test_Rician_CSCG_K10dB_20000_M8_N32_5dB.mat')
    data_ytest = sio.loadmat('y_test_Rician_CSCG_K10dB_20000_M8_N32_5dB.mat')

    #rename data
    xa_train = data_xtr['x_train']
    ya_train = data_ytr['y_train']

    xa_test = data_xtest['x_test']
    ya_test = data_ytest['y_test']


    train_model(model,
                (xa_train, ya_train),
                (xa_test, ya_test),
                epochs,
                batch_size)

if __name__ == '__main__':
    main()
