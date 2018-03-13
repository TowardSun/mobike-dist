# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from const import *
from keras.layers import Input, Dense, Conv2D, Flatten, Dropout, BatchNormalization, Activation
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import logging
from keras.models import load_model
import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def mobike_distribution(y_train, y_test):
    print('========================================================================')
    print('Mobike Train & Test Distribution:')
    print('Y_Train Size: ', len(y_train))
    print('Y_Train Range (%.2f, %.2f): ' % (np.min(y_train), np.max(y_train)))
    print('Y_Train Mean: ', np.mean(y_train))
    print('Y_Train Std: ', np.std(y_train))

    print('Y_Test Size: ', len(y_test))
    print('Y_Test Range (%.2f, %.2f): ' % (np.min(y_test), np.max(y_test)))
    print('Y_Test Mean: ', np.mean(y_test))
    print('Y_Test Std: ', np.std(y_test))


def mse_evaluation(model_name, y_test, y_predict, label='Test'):
    mse = mean_squared_error(y_test, y_predict)
    rmse = np.sqrt(mse)
    print('========================================================================')
    print('Model %s Performance:' % (model_name,))
    print(label, 'MSE: ', mse)
    print(label, 'RMSE: ', rmse)
    return mse, rmse


def entropy_evaluation(model_name, y_test, y_predict, label='Test', baseline_flag=False):
    y_predict = y_predict.flatten()

    y_predict_ratio = y_predict / np.sum(y_predict)
    y_test_ratio = y_test / np.sum(y_test)

    y_test_ratio = np.clip(y_test_ratio, 1e-08, 1)
    y_predict_ratio = np.clip(y_predict_ratio, 1e-08, 1)

    # cross_entropy = np.sum(y_test_ratio * (np.log(1 / y_predict_ratio)))
    kl_divergence = np.sum(y_test_ratio * (np.log(y_test_ratio / y_predict_ratio)))
    print('========================================================================')
    print('Model %s Performance:' % (model_name,))
    # print label, 'Cross Entropy: ', cross_entropy
    if baseline_flag:
        y_base_ratio = np.ones_like(y_test) / len(y_test)
        print('Baseline KL Divergence: ', np.sum(y_test_ratio * (np.log(y_test_ratio / y_base_ratio))))
        print('Baseline RMLSE: ', np.sqrt(mean_squared_error(np.log(y_test_ratio), np.log(y_base_ratio))))

    print(label, 'KL Divergence: ', kl_divergence)
    print(label, 'RMLSE: ', np.sqrt(mean_squared_error(np.log(y_test_ratio), np.log(y_predict_ratio))))


def transform_2_conv(lat_steps, lng_steps, x, y, size):
    """
    transform the 2-D arrays to 3-D arrays
    :return:
    """
    wd = size // 2

    # transform feature x
    conv_samples = []
    x = x.reshape((lat_steps, lng_steps, x.shape[1]))

    for i in range(wd, lat_steps - wd):
        for j in range(wd, lng_steps - wd):
            sample = x[i - wd:i + wd + 1, j - wd:j + wd + 1, :]
            conv_samples.append(sample)

    # transform label y
    y_conv = y.reshape(lat_steps, lng_steps)[wd:lat_steps - wd, wd:lng_steps - wd]
    y_conv = y_conv.flatten()

    return np.array(conv_samples), y_conv


def build_model(input_shape=(5, 5, 2), dropout=0.5, lr=0.001):
    """
    build CNN model
    :param input_shape:
    :param dropout:
    :param lr:
    :return:
    """
    window = input_shape[0]

    inputs = Input(shape=input_shape)
    layer = Conv2D(10, (window, window), activation='relu')(inputs)
    layer = Flatten()(layer)

    layer = Dense(32)(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(dropout)(layer)

    layer = Dense(32)(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(dropout)(layer)
    predictions = Dense(1, activation='relu')(layer)

    nn_model = Model(inputs=inputs, outputs=predictions)
    optimizer = Adam(lr=lr)
    nn_model.compile(optimizer=optimizer, loss='mse')

    return nn_model


def get_conv_data(cities, x, y, window, filter_count=1):
    """
    transfer the 2-D features from multi cities to 3-D feature maps
    :param cities:
    :param x: input features x
    :param y: input target y
    :param window: transformed feature map size
    :param filter_count:
    :return:
    """
    start_index = 0
    conv_xs = []
    conv_ys = []
    for city in cities:
        assert city in city_block_dict
        lat_steps, lng_steps, _ = city_block_dict[city]
        samples = lat_steps * lng_steps

        city_x = x[start_index: start_index + samples, :]
        city_y = y[start_index: start_index + samples]

        conv_x, conv_y = transform_2_conv(lat_steps, lng_steps, city_x, city_y, window)
        conv_x = conv_x[conv_y >= filter_count, :, :, :]
        conv_y = conv_y[conv_y >= filter_count]
        conv_xs.append(conv_x)
        conv_ys.append(conv_y)

        start_index += samples

    return np.vstack(conv_xs), np.concatenate(conv_ys)


def get_train_val_test(train_cities, test_cities, window=5, n_components=2, mode='common', y_scale=False):
    train_dfs = [pd.read_csv(PATH_PATTERN % city) for city in train_cities]
    train_df = pd.concat(train_dfs)

    test_dfs = [pd.read_csv(PATH_PATTERN % city) for city in test_cities]
    test_df = pd.concat(test_dfs)

    x_train = train_df[FEATURES].values
    y_train = train_df[TARGET].values

    x_test = test_df[FEATURES].values
    y_test = test_df[TARGET].values

    estimator = [
        ('scaler', StandardScaler()),
        ('reducer', FactorAnalysis(n_components=n_components))
    ]

    pipe = Pipeline(estimator)
    if mode == 'common':
        x = pipe.fit_transform(np.vstack([x_train, x_test]))
        split_index = len(x_train)
        x_train = x[:split_index, :]
        x_test = x[split_index:, :]
    else:
        x_train = pipe.fit_transform(x_train)
        x_test = pipe.transform(x_test)

    y_scaler = StandardScaler().fit(y_train[:, np.newaxis])
    if y_scale:
        y_train = y_scaler.transform(y_train[:, np.newaxis]).ravel()
        y_test = y_scaler.transform(y_test[:, np.newaxis]).ravel()

    filter_count = y_scaler.transform(np.array([[1]])).ravel() if y_scale else 0
    x_train, y_train = get_conv_data(train_cities, x_train, y_train, window, filter_count=filter_count)
    x_test, y_test = get_conv_data(test_cities, x_test, y_test, window, filter_count=filter_count)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, shuffle=True)
    return x_train, x_val, x_test, y_train, y_val, y_test, y_scaler


def run(train_cities, test_cities, window=5, n_components=10, mode='common', y_scale=False, lr=0.001, epochs=40):
    x_train, x_val, x_test, y_train, y_val, y_test, y_scaler = get_train_val_test(
        train_cities, test_cities, window, n_components, mode, y_scale)

    best_val_loss = np.inf
    model_save_path = './model/best_model_%s.h5' % datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    nn_model = build_model(x_train.shape[1:], lr=lr)
    history_dict = {
        'loss': [],
        'val_loss': []
    }
    for _ in range(epochs):
        history = nn_model.fit(
            x_train, y_train, validation_data=(x_val, y_val), batch_size=32, epochs=1, verbose=2).history
        for key, value in history.items():
            history_dict[key].append(value[-1])

        if history['val_loss'][-1] < best_val_loss:
            logger.info('Update model, last best loss {}, current val loss {}'.format(
                best_val_loss, history['val_loss'][-1]))
            best_val_loss = history['val_loss'][-1]
            nn_model.save(model_save_path)

    nn_model = load_model(model_save_path)

    y_train_pred = nn_model.predict(x_train)
    y_test_pred = nn_model.predict(x_test)

    if y_scale:
        y_train = y_scaler.inverse_transform(y_train)
        y_test = y_scaler.inverse_transform(y_test)
        y_train_pred = y_scaler.inverse_transform(y_train_pred)
        y_test_pred = y_scaler.inverse_transform(y_test_pred)

    mobike_distribution(y_train, y_test)
    mse_evaluation('conv_nn', y_train, y_train_pred, 'Train')
    mse_evaluation('conv_nn', y_test, y_test_pred)
    entropy_evaluation('conv_nn', y_train, y_train_pred, 'Train')
    entropy_evaluation('conv_nn', y_test, y_test_pred)


if __name__ == '__main__':
    run(train_cities=('sh',), test_cities=('nb',), y_scale=False, epochs=40)
