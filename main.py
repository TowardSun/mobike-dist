# -*- coding: utf-8 -*-


import logging
import pandas as pd
import tensorflow as tf
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, Activation
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from decomposition import TCA
from sklearn.model_selection import ParameterGrid
from const import ModelChoice, CITY_BLOCK_DICT, FeatureChoice, ReducerChoice, PATH_PATTERN, TARGET, FEATURE_DICT
from const import LOG_DIR, ScaleChoice
from evaluation.metrics import *
from models import DenseConvModel, conv_block
import os
import keras.backend as K
import uuid

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


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
                first_filter=6, nb_dense_block_layers=(6,), growth_rate=6, compression=0.5):
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
    # layer = Conv2D(10, kernel_size=(5, 5), activation='relu')(inputs)
    layer = conv_block(inputs, filters=16, kernel_size=(1, 1), padding='valid', dropout_rate=dropout, conv_first=True)
    layer = conv_block(layer, filters=32, kernel_size=(3, 3), padding='same', dropout_rate=dropout, conv_first=True)
    layer = conv_block(layer, filters=16, kernel_size=(1, 1), padding='valid', dropout_rate=dropout, conv_first=True)
    layer = conv_block(layer, filters=32, kernel_size=(3, 3), padding='valid', dropout_rate=dropout, conv_first=True)
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
        assert city in CITY_BLOCK_DICT
        lat_steps, lng_steps, _ = CITY_BLOCK_DICT[city]
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


def get_train_val_test(train_cities, test_cities, features, window=5, reducer_choice=None,
                       n_components=2, scale_choice=ScaleChoice.origin):
    """
    Get train, validation set from train cities,
    get test set from test cities
    :param train_cities:
    :param test_cities:
    :param features: feature names
    :param window:
    :param n_components: reduced dimension
    :param reducer_choice: Factor Analysis | PCA | TCA
    :param scale_choice: target scale choice
    :return:
    """
    train_dfs = [pd.read_csv(PATH_PATTERN % city) for city in train_cities]
    train_df = pd.concat(train_dfs)

    test_dfs = [pd.read_csv(PATH_PATTERN % city) for city in test_cities]
    test_df = pd.concat(test_dfs)

    x_train = train_df[features].values
    y_train = train_df[TARGET].values

    x_test = test_df[features].values
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
    x_test, y_test = get_conv_data(test_cities, x_test, y_test, window, filter_count=1)

    # scale the target y
    y_scaler = None
    if scale_choice in (ScaleChoice.std, ScaleChoice.min_max):
        if scale_choice == ScaleChoice.std:
            y_scaler = StandardScaler().fit(y_train[:, np.newaxis])
        else:
            y_scaler = MinMaxScaler().fit(y_train[:, np.newaxis])

        y_train = y_scaler.transform(y_train[:, np.newaxis]).ravel()
        y_test = y_scaler.transform(y_test[:, np.newaxis]).ravel()

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, shuffle=True, random_state=0)
    return x_train, x_val, x_test, y_train, y_val, y_test, y_scaler


def write_results(result_file_path, result_item):
    columns = [
        'model_choice', 'feature_choice', 'reducer_choice', 'n_components', 'y_scale',
        'train_rmse', 'val_rmse', 'test_rmse', 's_kl', 's_rmlse',
        't_kl', 't_rmlse', 'model_path'
    ]

    res_list = []
    for col in columns:
        if col in result_item:
            res_list.append(str(result_item[col]))
        else:
            res_list.append('none')

    if not os.path.exists(result_file_path):
        with open(result_file_path, 'w') as f:
            f.write(','.join(columns) + '\n')
            f.write(','.join(res_list) + '\n')
    else:
        with open(result_file_path, 'a') as f:
            f.write(','.join(res_list) + '\n')

    res_df = pd.read_csv(result_file_path)
    res_df.sort_values(by='t_kl', inplace=True)
    res_df.to_csv(result_file_path, index=False)


def run(train_cities, test_cities, data_param_grid, model_param_dict, window=5,
        model_choice=ModelChoice.cnn, feature_choice=FeatureChoice.all, epochs=40,
        scale_choice=ScaleChoice.origin, early_stopping=True,
        early_stop_epoch=10):
    """
    main entrance to run the model
    :param train_cities: train cities' names
    :param test_cities: test cities' names
    :param data_param_grid: data grid search config
    :param model_param_dict: model grid search config
    :param window: neighbor window size
    :param model_choice: cnn model choice, simple cnn | dense net | res net
    :param feature_choice: multi-source feature selection
    :param epochs: max fit epochs
    :param scale_choice: target scale choice, no scale | std scale | min max scale
    :param early_stopping: whether early stopping
    :param early_stop_epoch: early stopping epoch counts
    :return:
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))

    # make log dir
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
    task_dir = os.path.join('./logs', str(uuid.uuid1()))
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)

    model_param_grid = model_param_dict[model_choice]
    features = FEATURE_DICT[feature_choice]
    candidate_data_params = list(ParameterGrid(param_grid=data_param_grid))
    candidate_data_params.append({
        'n_components': -1,
        'reducer_choice': ReducerChoice.all
    })
    candidate_model_params = list(ParameterGrid(param_grid=model_param_grid))

    result_file_path = './results/s_%s_t_%s_%s_%s_%s.csv' % (
        '_'.join(train_cities), '_'.join(test_cities), model_choice.name, feature_choice.name, scale_choice.name)
    for i, data_param in enumerate(candidate_data_params):
        if data_param['n_components'] > len(features):
            continue

        logger.info('Data config: %s' % data_param)
        x_train, x_val, x_test, y_train, y_val, y_test, y_scaler = get_train_val_test(
            train_cities, test_cities, features, window, scale_choice=scale_choice, **data_param)

        data_best_loss = np.inf
        data_best_model_path = ''
        for j, model_param in enumerate(candidate_model_params):
            logger.info('Model config: %s' % model_param)
            model_save_path = os.path.join(task_dir, 'best_model_%d_%d.h5' % (i, j))
            config_save_path = os.path.join(task_dir, 'model_config_%d_%d.config' % (i, j))
            with open(config_save_path, 'w') as f:
                f.write(str(model_param))

            nn_model = build_model(x_train.shape[1:], model_choice=model_choice, **model_param)
            history_dict = {
                'loss': [],
                'val_loss': []
            }

            best_val_loss = np.inf
            early_stop_counter = 0
            for k in range(epochs):
                print('\nEpoch %s/%s' % (k + 1, epochs))
                history = nn_model.fit(
                    x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=1, verbose=2).history
                for key, value in history.items():
                    history_dict[key].append(value[-1])

                if history['val_loss'][-1] < best_val_loss:
                    logger.info('Update model, last best loss {}, current val loss {}'.format(
                        best_val_loss, history['val_loss'][-1]))
                    best_val_loss = history['val_loss'][-1]
                    early_stop_counter = 0
                    nn_model.save(model_save_path)

                early_stop_counter += 1
                if early_stopping and early_stop_counter > early_stop_epoch:
                    break

            if best_val_loss < data_best_loss:
                data_best_loss = best_val_loss
                data_best_model_path = model_save_path

        nn_model = load_model(data_best_model_path)

        y_train_pred = nn_model.predict(x_train)
        y_val_pred = nn_model.predict(x_val)
        y_test_pred = nn_model.predict(x_test)

        if y_scaler is not None:
            y_train = y_scaler.inverse_transform(y_train.reshape(-1, 1)).ravel()
            y_val = y_scaler.inverse_transform(y_val.reshape(-1, 1)).ravel()
            y_test = y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
            y_train_pred = y_scaler.inverse_transform(y_train_pred.reshape(-1, 1)).ravel()
            y_val_pred = y_scaler.inverse_transform(y_val_pred.reshape(-1, 1)).ravel()
            y_test_pred = y_scaler.inverse_transform(y_test_pred.reshape(-1, 1)).ravel()

        train_rmse = mse_evaluation(model_choice.name, y_train, y_train_pred, 'Train')
        val_rmse = mse_evaluation(model_choice.name, y_val, y_val_pred, 'Val')
        test_rmse = mse_evaluation(model_choice.name, y_test, y_test_pred)
        s_kl, s_rmlse = entropy_evaluation(
            model_choice.name, np.concatenate([y_train, y_val]), np.concatenate([y_train_pred, y_val_pred]), 'Train')
        t_kl, t_rmlse = entropy_evaluation(model_choice.name, y_test, y_test_pred)

        result_dict = data_param.copy()
        result_dict['reducer_choice'] = result_dict['reducer_choice'].name
        result_dict['feature_choice'] = feature_choice.name
        result_dict['model_choice'] = model_choice.name
        result_dict['y_scale'] = scale_choice.name
        result_dict['train_rmse'] = train_rmse
        result_dict['val_rmse'] = val_rmse
        result_dict['test_rmse'] = test_rmse
        result_dict['s_kl'] = s_kl
        result_dict['s_rmlse'] = s_rmlse
        result_dict['t_kl'] = t_kl
        result_dict['t_rmlse'] = t_rmlse
        result_dict['model_path'] = data_best_model_path
        logger.info(str(result_dict))

        write_results(result_file_path, result_dict)


if __name__ == '__main__':
    data_param_config = dict(
        n_components=[10, 20],
        reducer_choice=[ReducerChoice.fa]
    )
    model_param_config = {
        ModelChoice.cnn: dict(
            lr=[0.001],
            dropout=[0.5]
        ),
        ModelChoice.dense_cnn: dict(
            lr=[0.001],
            dropout=[0.2],
            first_filter=[16],
            nb_dense_block_layers=[(4,)],
            growth_rate=[6], compression=[0.5]
        )
    }

    run(
        train_cities=('sh',), test_cities=('nb',), data_param_grid=data_param_config,
        feature_choice=FeatureChoice.poi,
        model_param_dict=model_param_config,
        scale_choice=ScaleChoice.origin, epochs=100,
        model_choice=ModelChoice.cnn
    )

    # run(
    #     train_cities=('bj', ), test_cities=('nb',), data_param_grid=data_param_config,
    #     model_param_dict=model_param_config,
    #     y_scale=True, epochs=100,
    #     model_choice=ModelChoice.cnn
    # )
    # run(
    #     train_cities=('bj', ), test_cities=('nb',), data_param_grid=data_param_config,
    #     model_param_dict=model_param_config,
    #     y_scale=False, epochs=100,
    #     model_choice=ModelChoice.dense_cnn
    # )
    # run(
    #     train_cities=('bj', ), test_cities=('nb',), data_param_grid=data_param_config,
    #     model_param_dict=model_param_config,
    #     y_scale=True, epochs=100,
    #     model_choice=ModelChoice.dense_cnn
    # )
