# -*- coding: utf-8 -*-

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class TCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_src, n_tar, n_components=5, kernel_type='rbf', kernel_param=1, mu=1):
        """
        Init function
        :param n_components: dims after tca (n_components <= d)
        :param kernel_type: 'rbf' | 'linear' | 'poly' (default is 'rbf')
        :param kernel_param: kernel param
        :param mu: param
        """
        self.n_src = n_src
        self.n_tar = n_tar
        self.n_components = n_components
        self.kernel_param = kernel_param
        self.kernel_type = kernel_type
        self.mu = mu
        self.K = None
        self.V = None
        self.eig_values = None

    @staticmethod
    def get_L(n_src, n_tar):
        """
        Get index matrix
        :param n_src: num of source domain
        :param n_tar: num of target domain
        :return: index matrix L
        """
        L_ss = (1. / (n_src * n_src)) * np.full((n_src, n_src), 1)
        L_st = (-1. / (n_src * n_tar)) * np.full((n_src, n_tar), 1)
        L_ts = (-1. / (n_tar * n_src)) * np.full((n_tar, n_src), 1)
        L_tt = (1. / (n_tar * n_tar)) * np.full((n_tar, n_tar), 1)
        L_up = np.hstack((L_ss, L_st))
        L_down = np.hstack((L_ts, L_tt))
        L = np.vstack((L_up, L_down))
        return L

    @staticmethod
    def get_kernel(kernel_type, kernel_param, x1, x2=None):
        """
        Calculate kernel for TCA (inline func)
        :param kernel_type: 'rbf' | 'linear' | 'poly'
        :param kernel_param: param
        :param x1: x1 matrix (n1,d)
        :param x2: x2 matrix (n2,d)
        :return: Kernel K
        """
        n1, dim = x1.shape
        K = None
        if kernel_type == 'linear':
            if x2 is not None:
                K = np.dot(x2, x1.T)
            else:
                K = np.dot(x1, x1.T)
        elif kernel_type == 'poly':
            if x2 is not None:
                K = np.power(np.dot(x1, x2.T), kernel_param)
            else:
                K = np.power(np.dot(x1, x1.T), kernel_param)
        elif kernel_type == 'rbf':
            if x2 is not None:
                n2 = x2.shape[0]
                sum_x2 = np.sum(np.multiply(x2, x2), axis=1)
                sum_x2 = sum_x2.reshape((len(sum_x2), 1))
                K = np.exp(-1 * (
                    np.tile(np.sum(np.multiply(x1, x1), axis=1).T, (n2, 1)) +
                    np.tile(sum_x2, (1, n1)) - 2 * np.dot(x2, x1.T)) / (dim * 2 * kernel_param))
            else:
                P = np.sum(np.multiply(x1, x1), axis=1)
                P = P.reshape((len(P), 1))
                K = np.exp(
                    -1 * (np.tile(P.T, (n1, 1)) + np.tile(P, (1, n1)) - 2 * np.dot(x1, x1.T)) / (dim * 2 * kernel_param))
        # more kernels can be added
        return K

    def fit(self, X, y=None):
        L = self.get_L(self.n_src, self.n_tar)
        L[np.isnan(L)] = 0
        self.K = self.get_kernel(self.kernel_type, self.kernel_param, X)
        self.K[np.isnan(self.K)] = 0

        H = np.identity(self.n_src + self.n_tar) - 1. / (self.n_src + self.n_tar) * np.ones(
            (self.n_src + self.n_tar, 1)) * np.ones((self.n_src + self.n_tar, 1)).T
        pinv = self.mu * np.identity(self.n_src + self.n_tar) + np.dot(np.dot(self.K, L), self.K)
        pinv[np.isnan(pinv)] = 0
        Kc = np.dot(np.dot(np.dot(np.linalg.pinv(pinv), self.K), H), self.K)
        Kc[np.isnan(Kc)] = 0

        D, V = np.linalg.eig(Kc)
        eig_values = D.reshape(len(D), 1)
        self.eig_values = np.sort(eig_values[::-1], axis=0)
        index_sorted = np.argsort(-eig_values, axis=0)
        V = V[:, index_sorted]
        self.V = V.reshape((V.shape[0], V.shape[1]))
        return self

    def transform(self, X):
        """
        TCA main method. Wrapped from Sinno J. Pan and Qiang Yang's "Domain adaptation via transfer component analysis.
        IEEE TNN 2011"
        :return: transformed x_src_tca,x_tar_tca,x_tar_o_tca
        """
        x_src_tca = np.dot(self.K[:self.n_src, :], self.V)
        x_tar_tca = np.dot(self.K[self.n_src:, :], self.V)

        x_src_tca = np.asarray(x_src_tca[:, :self.n_components], dtype=float)
        x_tar_tca = np.asarray(x_tar_tca[:, :self.n_components], dtype=float)

        return np.vstack([x_src_tca, x_tar_tca])
