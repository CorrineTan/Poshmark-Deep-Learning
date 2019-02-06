# coding: utf-8
import os
# os.chdir('..')
import sys
# sys.path.insert(0, './python')

import caffe
import numpy as np
from pylab import *
import pickle

def fine_tune(solver_path, trained_model_path, scratch_solver_path, 
            result_suffix):
    # get_ipython().magic(u'matplotlib inline')
    # get_ipython().system(u'diff models/bvlc_reference_caffenet/train_val.prototxt models/poshmark_734/train_val.prototxt')
    niter = 100000
    test_interval = 200

    calc_avg_error_interval = 10000

    big_plot_interval = niter
    zoomed_in_plot_interval = niter/10
    train_error_show_interval = 200

    # solver_path = 'models/poshmark_734_no_oversample/solver.prototxt'
    # trained_model_path = 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    # scratch_solver_path = 'models/poshmark_734_no_oversample/solver.prototxt'
    # result_suffix = '_no_oversample_20160210'

    result_directory = 'result/result' + result_suffix

    if not os.path.exists(result_directory):
        os.makedirs(result_directory)

    # losses will also be stored in the log
    train_loss = np.zeros(niter)
    test_loss = np.zeros(niter/test_interval + 1)
    scratch_train_loss = np.zeros(niter)
    scratch_test_loss = np.zeros(niter/test_interval + 1)

    caffe.set_device(0)
    caffe.set_mode_gpu()
    # We create a solver that fine-tunes from a previously trained network.
    solver = caffe.SGDSolver(solver_path)
    solver.net.copy_from(trained_model_path)
    # For reference, we also create a solver that does no finetuning.
    scratch_solver = caffe.SGDSolver(scratch_solver_path)

    # We run the solver for niter times, and record the training loss.
    for it in range(niter):
        solver.step(1)  # SGD by Caffe
        scratch_solver.step(1)
        # store the train loss
        train_loss[it] = solver.net.blobs['loss'].data
        scratch_train_loss[it] = scratch_solver.net.blobs['loss'].data
        if it % 10 == 0:
            print 'iter %d, finetune_loss=%f, scratch_loss=%f' % (it, train_loss[it], scratch_train_loss[it])
        if it % test_interval == 0:
            solver.test_nets[0].forward()
            test_loss[it/test_interval] = solver.test_nets[0].blobs['loss'].data
            scratch_solver.test_nets[0].forward()
            scratch_test_loss[it/test_interval] = scratch_solver.test_nets[0].blobs['loss'].data
            print 'iter %d, test_finetune_loss=%f, test_scratch_loss=%f' % (it, test_loss[it/test_interval], scratch_test_loss[it/test_interval])
    print 'done'

    # calculate the average rmse after certain numbers of iterations
    avg_train_rmse = np.empty((niter/calc_avg_error_interval, 1), dtype=np.float64)
    avg_test_rmse = np.empty((niter/calc_avg_error_interval, 1), dtype=np.float64)
    avg_scratch_train_rmse = np.empty((niter/calc_avg_error_interval, 1), dtype=np.float64)
    avg_scratch_test_rmse = np.empty((niter/calc_avg_error_interval, 1), dtype=np.float64)
    for idx, i in enumerate(range(0, niter, calc_avg_error_interval)):
        avg_train_rmse[idx] = np.mean((train_loss[i:(i + calc_avg_error_interval)]*2)**0.5, axis = 0)
        avg_test_rmse[idx] = np.mean((test_loss[(i/test_interval):\
                ((i+calc_avg_error_interval)/test_interval)]*2)**0.5, axis = 0)
        avg_scratch_train_rmse[idx] = np.mean((scratch_train_loss[i:(i + calc_avg_error_interval)]*2)**0.5, axis = 0)
        avg_scratch_test_rmse[idx] = np.mean((scratch_test_loss[(i/test_interval):\
                ((i+calc_avg_error_interval)/test_interval)]*2)**0.5, axis = 0)
    print(avg_train_rmse)
    print(avg_test_rmse)
    print(avg_scratch_train_rmse)
    print(avg_scratch_test_rmse)

    with open(result_directory + 
            '/train_test_errors' + result_suffix + '.pickle', 'w') as f:
        pickle.dump([train_loss, scratch_train_loss, test_loss, scratch_test_loss, 
                avg_train_rmse, avg_test_rmse, avg_scratch_train_rmse, 
                avg_scratch_test_rmse], f)

    solver.net.save(result_directory + 
            '/fine_tune' + result_suffix + '.caffemodel')
    scratch_solver.net.save(result_directory + 
            '/build_from_scratch' + result_suffix + '.caffemodel')

    # get_ipython().magic(u'matplotlib qt')
    plt.figure(0)
    x_range = range(0, zoomed_in_plot_interval, train_error_show_interval)
    fine_tune_train_plot, = plt.plot(x_range, (train_loss[x_range]*2)**0.5, label='Fine tune train loss')
    scratch_train_plot, = plt.plot(x_range, (scratch_train_loss[x_range]*2)**0.5, label='Build from scratch train loss')
    plt.legend(handles=[fine_tune_train_plot, 
            scratch_train_plot])
    plt.grid()
    plt.xlabel('Number of iterations')
    plt.ylabel('RMSE')
    plt.show(block=False)
    plt.savefig(result_directory + 
            '/train_errors_zoomed_in' + result_suffix + '.png')

    # get_ipython().magic(u'matplotlib qt')
    plt.figure(1)
    x_range = range(0, big_plot_interval, train_error_show_interval)
    fine_tune_train_plot, = plt.plot(x_range, (train_loss[x_range]*2)**0.5, label='Fine tune train loss')
    scratch_train_plot, = plt.plot(x_range, (scratch_train_loss[x_range]*2)**0.5, label='Build from scratch train loss')
    plt.legend(handles=[fine_tune_train_plot, scratch_train_plot])
    plt.grid()
    plt.xlabel('Number of iterations')
    plt.ylabel('RMSE')
    plt.show(block=False)
    plt.savefig(result_directory + 
            '/train_errors' + result_suffix + '.png')

    # get_ipython().magic(u'matplotlib qt')
    plt.figure(2)
    start_ite = 0
    end_ite = zoomed_in_plot_interval
    iterations = range(start_ite, end_ite, test_interval)
    x_range = range(start_ite/test_interval, end_ite/test_interval)

    fine_tune_test_plot, = plt.plot(iterations, (test_loss[x_range]*2)**0.5, label='Fine tune test loss')
    scratch_test_plot, = plt.plot(iterations, (scratch_test_loss[x_range]*2)**0.5, label='Build from scratch test loss')
    plt.legend(handles=[fine_tune_test_plot, scratch_test_plot])
    plt.grid()
    plt.xlabel('Number of iterations')
    plt.ylabel('RMSE')
    plt.show(block=False)
    plt.savefig(result_directory + 
            '/test_errors_zoomed_in' + result_suffix + '.png')

    # get_ipython().magic(u'matplotlib qt')
    plt.figure(3)
    start_ite = 0
    end_ite = big_plot_interval
    iterations = range(start_ite, end_ite, test_interval)
    x_range = range(start_ite/test_interval, end_ite/test_interval)

    fine_tune_test_plot, = plt.plot(iterations, (test_loss[x_range]*2)**0.5, label='Fine tune test loss')
    scratch_test_plot, = plt.plot(iterations, (scratch_test_loss[x_range]*2)**0.5, label='Build from scratch test loss')
    plt.legend(handles=[fine_tune_test_plot, scratch_test_plot])
    plt.grid()
    plt.xlabel('Number of iterations')
    plt.ylabel('RMSE')
    plt.show(block=False)
    plt.savefig(result_directory + 
            '/test_errors' + result_suffix + '.png')

def cross_validation(model_dir, trained_model_path, result_suffix, numFold):
    for i in range(numFold):
        solver_path = model_dir + '/solver_' + str(i) + '.prototxt'
        scratch_solver_path = solver_path
        iter_result_suffix = result_suffix + '_fold' + str(i)
        fine_tune(solver_path, trained_model_path, scratch_solver_path, 
            result_suffix)

def main():
    # for one fine tune task
    # solver_path = 'models/poshmark_734_no_oversample/solver.prototxt'
    # trained_model_path = 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    # scratch_solver_path = solver_path
    # result_suffix = '_no_oversample_20160210'
    # fine_tune(solver_path, trained_model_path, scratch_solver_path, 
    #         result_suffix)
    # for cross validation
    model_dir = 'models/poshmark_734_oversample_cross_validation'
    trained_model_path = \
        'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    result_suffix = 'oversample_cv_20160211'
    cross_validation(model_dir, trained_model_path, result_suffix, 5)


if __name__ == '__main__':
    main()