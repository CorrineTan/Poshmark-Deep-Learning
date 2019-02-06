# coding: utf-8
import os
# os.chdir('..')
import sys
# sys.path.insert(0, './python')

import caffe
import numpy as np
from pylab import *
import pickle

# get_ipython().magic(u'matplotlib inline')
# get_ipython().system(u'diff models/bvlc_reference_caffenet/train_val.prototxt models/poshmark_734/train_val.prototxt')
niter = 100000
test_interval = 200

calc_avg_error_interval = 10000

big_plot_interval = niter
zoomed_in_plot_interval = niter/10
train_error_show_interval = 200

result_suffix = '_oversample_cv_20160219_fold1_test'

with open('result/result' + result_suffix + \
        '/train_test_errors' + result_suffix + '.pickle', 'r') as f:
    [train_loss, scratch_train_loss, test_loss, scratch_test_loss, 
            avg_train_rmse, avg_test_rmse, avg_scratch_train_rmse, 
            avg_scratch_test_rmse] = pickle.load(f)

# get_ipython().magic(u'matplotlib qt')
plt.figure(0)
x_range = range(0, zoomed_in_plot_interval, train_error_show_interval)
fine_tune_train_plot, = plt.plot(x_range, (train_loss[x_range]*2)**0.5, label='Fine tune train loss')
scratch_train_plot, = plt.plot(x_range, (scratch_train_loss[x_range]*2)**0.5, label='Build from scratch train loss')
# plt.legend(handles=[fine_tune_train_plot, scratch_train_plot])
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('RMSE')
plt.show(block=False)
plt.savefig('result/result' + result_suffix + \
        '/train_errors_zoomed_in' + result_suffix + '.png')

# get_ipython().magic(u'matplotlib qt')
plt.figure(1)
x_range = range(0, big_plot_interval, train_error_show_interval)
fine_tune_train_plot, = plt.plot(x_range, (train_loss[x_range]*2)**0.5, label='Fine tune train loss')
scratch_train_plot, = plt.plot(x_range, (scratch_train_loss[x_range]*2)**0.5, label='Build from scratch train loss')
# plt.legend(handles=[fine_tune_train_plot, scratch_train_plot])
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('RMSE')
plt.show(block=False)
plt.savefig('result/result' + result_suffix + \
        '/train_errors' + result_suffix + '.png')

# get_ipython().magic(u'matplotlib qt')
plt.figure(2)
start_ite = 0
end_ite = zoomed_in_plot_interval
iterations = range(start_ite, end_ite, test_interval)
x_range = range(start_ite/test_interval, end_ite/test_interval)

fine_tune_test_plot, = plt.plot(iterations, (test_loss[x_range]*2)**0.5, label='Fine tune test loss')
scratch_test_plot, = plt.plot(iterations, (scratch_test_loss[x_range]*2)**0.5, label='Build from scratch test loss')
# plt.legend(handles=[fine_tune_test_plot, scratch_test_plot])
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('RMSE')
plt.show(block=False)
plt.savefig('result/result' + result_suffix + \
        '/test_errors_zoomed_in' + result_suffix + '.png')

# get_ipython().magic(u'matplotlib qt')
plt.figure(3)
start_ite = 0
end_ite = big_plot_interval
iterations = range(start_ite, end_ite, test_interval)
x_range = range(start_ite/test_interval, end_ite/test_interval)

fine_tune_test_plot, = plt.plot(iterations, (test_loss[x_range]*2)**0.5, label='Fine tune test loss')
scratch_test_plot, = plt.plot(iterations, (scratch_test_loss[x_range]*2)**0.5, label='Build from scratch test loss')
# plt.legend(handles=[fine_tune_test_plot, scratch_test_plot])
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('RMSE')
plt.show(block=False)
plt.savefig('result/result' + result_suffix + \
        '/test_errors' + result_suffix + '.png')