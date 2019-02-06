# coding: utf-8
import os
import sys

# import caffe
import numpy as np
from pylab import *
import pickle

def read_pickle_data(pickle_path):
    pickle_file = open(pickle_path, 'r')
    [train_loss, scratch_train_loss, test_loss, scratch_test_loss, 
            avg_train_rmse, avg_test_rmse, avg_scratch_train_rmse, 
            avg_scratch_test_rmse] = pickle.load(pickle_file)
    return [train_loss, scratch_train_loss, test_loss, scratch_test_loss, 
            avg_train_rmse, avg_test_rmse, avg_scratch_train_rmse, 
            avg_scratch_test_rmse]

def calc_avg_error_after_iter(avg_train_rmse, avg_test_rmse, 
        avg_scratch_train_rmse, avg_scratch_test_rmse, iteration):
    start_idx = iteration/10000 - 1
    avg_train_rmse_after_iter = np.mean(avg_train_rmse[start_idx:])
    avg_test_rmse_after_iter = np.mean(avg_test_rmse[start_idx:])
    avg_scratch_train_rmse_after_iter = \
            np.mean(avg_scratch_train_rmse[start_idx:])
    avg_scratch_test_rmse_after_iter = \
            np.mean(avg_scratch_test_rmse[start_idx:])

    print('avg_train_rmse_after_iter = ' + str(avg_train_rmse_after_iter))
    print('avg_test_rmse_after_iter = ' + str(avg_test_rmse_after_iter))
    print('avg_scratch_train_rmse_after_iter = ' + \
            str(avg_scratch_train_rmse_after_iter))
    print('avg_scratch_test_rmse_after_iter = ' + \
            str(avg_scratch_test_rmse_after_iter))

    return [avg_train_rmse_after_iter, avg_test_rmse_after_iter, 
            avg_scratch_train_rmse_after_iter, 
            avg_scratch_test_rmse_after_iter]

def print_avg_loss_table_latex(avg_train_rmse, avg_test_rmse, 
        avg_scratch_train_rmse, avg_scratch_test_rmse):
    for i in range(len(avg_train_rmse)):
        print(str(i * 10000) + '-' + str(i * 10000 + 10000) + ' & ' + '{0:.3f}'.format(avg_train_rmse[i][0]) + ' & ' + '{0:.3f}'.format(avg_scratch_train_rmse[i][0])+ ' & ' + '{0:.3f}'.format(avg_test_rmse[i][0]) + ' & ' + '{0:.3f}'.format(avg_scratch_test_rmse[i][0]) + ' \\\\ \\hline')

def print_avg_loss_table_latex_no_scratch(avg_train_rmse, avg_test_rmse):
    for i in range(len(avg_train_rmse)):
        print(str(i * 10000) + '-' + str(i * 10000 + 10000) + ' & ' + '{0:.3f}'.format(avg_train_rmse[i][0]) + ' & ' + '{0:.3f}'.format(avg_test_rmse[i][0]) + ' \\\\ \\hline')

    


def main():
    # with scratch net, one test run
    # pickle_path_prefix = './result/result_oversample_cv_20160219_fold'
    # pickle_path_suffix = 'train_test_errors_oversample_cv_20160219_fold'

    # no scratch net, multiple test runs
    pickle_path_prefix = './result/result_cv_multi_test_runs_20160301_fold'
    pickle_path_suffix = 'train_test_errors_cv_multi_test_runs_20160301_fold'
    num_fold = 5
    total_train_error = 0
    total_test_error = 0
    total_scratch_train_error = 0
    total_scratch_test_error = 0

    for i in range(num_fold):
        print('Fold ' + str(i) + 
                '  ====================================')

        pickle_path = pickle_path_prefix + str(i) + '/' + \
                pickle_path_suffix + str(i) + '.pickle'
        [train_loss, scratch_train_loss, test_loss, scratch_test_loss, 
            avg_train_rmse, avg_test_rmse, avg_scratch_train_rmse, 
            avg_scratch_test_rmse] = read_pickle_data(pickle_path)

        [avg_train_rmse_after_iter, avg_test_rmse_after_iter, 
            avg_scratch_train_rmse_after_iter, 
            avg_scratch_test_rmse_after_iter] = \
                calc_avg_error_after_iter(
                avg_train_rmse, avg_test_rmse, 
                avg_scratch_train_rmse, avg_scratch_test_rmse, 40000)

        total_train_error += avg_train_rmse_after_iter
        total_test_error += avg_test_rmse_after_iter
        total_scratch_train_error += avg_scratch_train_rmse_after_iter
        total_scratch_test_error += avg_scratch_test_rmse_after_iter

        # print_avg_loss_table_latex(avg_train_rmse, avg_test_rmse, 
        # avg_scratch_train_rmse, avg_scratch_test_rmse)
        print_avg_loss_table_latex_no_scratch(avg_train_rmse, avg_test_rmse)

    print('===============================================================')
    print('Cross-validation average: \n' 
            + 'train = ' + str(total_train_error/num_fold) + '\n' 
            + 'test = ' + str(total_test_error/num_fold) + '\n'
            + 'scratch_train = ' + str(total_scratch_train_error/num_fold) + '\n'
            + 'scratch_test = ' + str(total_scratch_test_error/num_fold) + '\n')



if __name__ == '__main__':
    main()















