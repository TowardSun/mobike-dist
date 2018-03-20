# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import mean_squared_error


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
    return rmse


def entropy_evaluation(model_name, y_test, y_predict, label='Test', baseline_flag=False):
    y_predict = y_predict.flatten()

    y_predict_ratio = y_predict / np.sum(y_predict)
    y_test_ratio = y_test / np.sum(y_test)

    y_test_ratio = np.clip(y_test_ratio, 1e-08, 1)
    y_predict_ratio = np.clip(y_predict_ratio, 1e-08, 1)

    # cross_entropy = np.sum(y_test_ratio * (np.log(1 / y_predict_ratio)))
    kl_divergence = np.sum(y_test_ratio * (np.log(y_test_ratio / y_predict_ratio)))
    rmlse = np.sqrt(mean_squared_error(np.log(y_test_ratio), np.log(y_predict_ratio)))
    print('========================================================================')
    print('Model %s Performance:' % (model_name,))
    # print label, 'Cross Entropy: ', cross_entropy
    if baseline_flag:
        y_base_ratio = np.ones_like(y_test) / len(y_test)
        print('Baseline KL Divergence: ', np.sum(y_test_ratio * (np.log(y_test_ratio / y_base_ratio))))
        print('Baseline RMLSE: ', np.sqrt(mean_squared_error(np.log(y_test_ratio), np.log(y_base_ratio))))

    print(label, 'KL Divergence: ', kl_divergence)
    print(label, 'RMLSE: ', rmlse)
    return kl_divergence, rmlse
