# -*- coding: utf-8 -*-


import datetime
import logging
from enum import IntEnum
import argparse
import pandas as pd
import tensorflow as tf
from keras.layers import Input, Dense, Conv2D, Flatten, Dropout, BatchNormalization, Activation
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from decomposition import TCA
from sklearn.model_selection import ParameterGrid
from const import *
from evaluation.metrics import *
from models.dense_conv import DenseConvModel
import os


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class ModelChoice(IntEnum):
    cnn = 0
    dense_cnn = 1


class ReducerChoice(IntEnum):
    fa = 0
    pca = 1
    tca = 2


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


def build_model(input_shape=(5, 5, 2), model_choice=ModelChoice.cnn, lr=0.001, dropout=0.5,
                first_filter=6, nb_dense_block_layers=(6, ), growth_rate=6, compression=0.5):
    if model_choice not in (ModelChoice.cnn, ModelChoice.dense_cnn):
        raise ValueError('not valid model choice')

    if model_choice == ModelChoice.cnn:
        model = build_cnn_model(input_shape, lr=lr, dropout=dropout)
    elif model_choice == ModelChoice.dense_cnn:
        nn_model = DenseConvModel(
            input_shape, first_filters=first_filter, nb_dense_block_layers=nb_dense_block_layers,
            growth_rate=growth_rate, compression=compression,
            lr=lr, dropout=dropout
        )
        model = nn_model.build_model()
    else:
        model = None
    return model


def build_cnn_model(input_shape=(5, 5, 2), dropout=0.5, lr=0.001):
    """
    build CNN model
    :param input_shape:
    :param dropout:
    :param lr:
    :return:
    """
    inputs = Input(shape=input_shape)
    layer = Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    layer = Conv2D(32, kernel_size=(3, 3), activation='relu')(layer)
    layer = Flatten()(layer)

    layer = Dense(32)(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(dropout)(layer)

    # layer = Dense(32)(layer)
    # layer = BatchNormalization()(layer)
    # layer = Activation('relu')(layer)
    # layer = Dropout(dropout)(layer)
    outputs = Dense(1)(layer)

    nn_model = Model(inputs=inputs, outputs=outputs)
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


def get_train_val_test(train_cities, test_cities, window=5, reducer_choice=None,
                       n_components=2, y_scale=False):
    """
    Get train, validation set from train cities,
    get test set from test cities
    :param train_cities:
    :param test_cities:
    :param window:
    :param n_components:
    :param reducer_choice: Factor Analysis | PCA | TCA
    :param y_scale:
    :return:
    """
    train_dfs = [pd.read_csv(PATH_PATTERN % city) for city in train_cities]
    train_df = pd.concat(train_dfs)

    test_dfs = [pd.read_csv(PATH_PATTERN % city) for city in test_cities]
    test_df = pd.concat(test_dfs)

    x_train = train_df[FEATURES].values
    y_train = train_df[TARGET].values

    x_test = test_df[FEATURES].values
    y_test = test_df[TARGET].values

    if reducer_choice == ReducerChoice.pca:
        reducer = PCA(n_components=n_components)
    elif reducer_choice == ReducerChoice.fa:
        reducer = FactorAnalysis(n_components=n_components)
    elif reducer_choice == ReducerChoice.tca:
        reducer = TCA(n_src=len(x_train), n_tar=len(x_test), n_components=n_components)
    else:
        reducer = None

    # preprocess the data
    estimator = [
        ('scaler', StandardScaler()),
        ('reducer', reducer)
    ]

    # common scale and dimension reduce
    pipe = Pipeline(estimator)
    x = pipe.fit_transform(np.vstack([x_train, x_test]))
    split_index = len(x_train)
    x_train = x[:split_index, :]
    x_test = x[split_index:, :]

    # transform into conv data
    x_train, y_train = get_conv_data(train_cities, x_train, y_train, window, filter_count=1)
    x_test, y_test = get_conv_data(test_cities, x_test, y_test, window, filter_count=0)

    # scale the target y
    y_scaler = StandardScaler().fit(y_train[:, np.newaxis])
    if y_scale:
        y_train = y_scaler.transform(y_train[:, np.newaxis]).ravel()
        y_test = y_scaler.transform(y_test[:, np.newaxis]).ravel()

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, shuffle=True, random_state=0)
    return x_train, x_val, x_test, y_train, y_val, y_test, y_scaler


def run(train_cities, test_cities, window=5,
        model_choice=ModelChoice.cnn, epochs=40, y_scale=False, early_stopping=True, early_stop_epoch=10):
    """
    main entrance to run the model
    :param train_cities: train cities' names
    :param test_cities: test cities' names
    :param window: neighbor window size
    :param model_choice: cnn model choice, simple cnn | dense net | res net
    :param epochs: max fit epochs
    :param y_scale: whether scale the target y
    :param early_stopping: whether early stopping
    :param early_stop_epoch: early stopping epoch counts
    :return:
    """
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    config.gpu_options.allow_growth = True

    data_param_grid = dict(
        n_components=list(range(5, 31, 5)),
        reducer_choice=[ReducerChoice.pca]
    )
    model_param_dict = {
        ModelChoice.cnn: dict(
            lr=[0.001, 0.0001],
            dropout=[0.2, 0.5]
        ),
        ModelChoice.dense_cnn: dict(
            lr=[0.001, 0.0001],
            dropout=[0.2, 0.5],
            first_filter=[16],
            nb_dense_block_layers=[(4,)],
            growth_rate=[6], compression=[0.5]
        )
    }
    model_param_grid = model_param_dict[model_choice]
    candidate_data_params = list(ParameterGrid(param_grid=data_param_grid))
    candidate_model_params = list(ParameterGrid(param_grid=model_param_grid))
    results = []

    task_dir = os.path.join('./logs', datetime.datetime.now().strftime('%m%d%H%M%S'))
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)

    for i, data_param in enumerate(candidate_data_params):
        logger.info('Data config: %s' % data_param)
        x_train, x_val, x_test, y_train, y_val, y_test, y_scaler = get_train_val_test(
            train_cities, test_cities, window, y_scale=y_scale, **data_param)

        best_val_loss = np.inf
        model_save_path = os.path.join(task_dir, 'best_model_%d.h5' % i)
        for model_param in candidate_model_params:
            logger.info('Model config: %s' % model_param)
            nn_model = build_model(x_train.shape[1:], model_choice=model_choice, **model_param)
            history_dict = {
                'loss': [],
                'val_loss': []
            }

            early_stop_counter = 0
            for j in range(epochs):
                print('\nEpoch %s/%s' % (j + 1, epochs))
                history = nn_model.fit(
                    x_train, y_train, validation_data=(x_val, y_val), batch_size=32, epochs=1, verbose=2).history
                for key, value in history.items():
                    history_dict[key].append(value[-1])

                if history['val_loss'][-1] < best_val_loss:
                    logger.info('Update model, last best loss {}, current val loss {}'.format(
                        best_val_loss, history['val_loss'][-1]))
                    best_val_loss = history['val_loss'][-1]
                    nn_model.save(model_save_path)
                    early_stop_counter = 0
                early_stop_counter += 1
                if early_stopping and early_stop_counter > early_stop_epoch:
                    break

        nn_model = load_model(model_save_path)

        y_train_pred = nn_model.predict(x_train)
        y_val_pred = nn_model.predict(x_val)
        y_test_pred = nn_model.predict(x_test)

        if y_scale:
            y_train = y_scaler.inverse_transform(y_train).ravel()
            y_val = y_scaler.inverse_transform(y_val).ravel()
            y_test = y_scaler.inverse_transform(y_test).ravel()
            y_train_pred = y_scaler.inverse_transform(y_train_pred).ravel()
            y_val_pred = y_scaler.inverse_transform(y_val_pred).ravel()
            y_test_pred = y_scaler.inverse_transform(y_test_pred).ravel()

        train_rmse = mse_evaluation(model_choice.name, y_train, y_train_pred, 'Train')
        val_rmse = mse_evaluation(model_choice.name, y_val, y_val_pred, 'Val')
        test_rmse = mse_evaluation(model_choice.name, y_test, y_test_pred)
        s_kl, s_rmlse = entropy_evaluation(
            model_choice.name, np.concatenate([y_train, y_val]), np.concatenate([y_train_pred, y_val_pred]), 'Train')
        t_kl, t_rmlse = entropy_evaluation(model_choice.name, y_test, y_test_pred)

        result_dict = data_param.copy()
        result_dict['train_rmse'] = train_rmse
        result_dict['val_rmse'] = val_rmse
        result_dict['test_rmse'] = test_rmse
        result_dict['s_kl'] = s_kl
        result_dict['s_rmlse'] = s_rmlse
        result_dict['t_kl'] = t_kl
        result_dict['t_rmlse'] = t_rmlse
        results.append(result_dict)

        res_df = pd.DataFrame(results)
        res_df = res_df[['reducer_choice', 'n_components', 'train_rmse', 'val_rmse', 'test_rmse', 's_kl', 's_rmlse'
                         't_kl', 't_rmlse']]
        res_df.sort_values(by='t_kl', inplace=True)
        res_df.to_csv(os.path.join(task_dir, 'results.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mobike dist')
    parser.add_argument('--train_cities', required=True, default='bj', type=str)
    parser.add_argument('--test_cities', required=True, default='nb', type=str)
    parser.add_argument('--y_scale', action='store_true', help='std scale the target label')
    parser.add_argument('--model_choice', default=0, type=int, help='model choice: 0 -> cnn, 1-> dense cnn',
                        choices=[0, 1])
    parser.add_argument('--epochs', default=100, type=int)

    args = parser.parse_args()
    train_city = args.train_cities.split(',')
    test_city = args.test_cities.split(',')

    run(train_cities=train_city, test_cities=test_city, y_scale=args.y_scale, epochs=args.epochs,
        model_choice=ModelChoice(args.model_choice))
