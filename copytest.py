# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 23:44:11 2017

@author: jrbtaylor
"""

from __future__ import print_function

import numpy
import theano
from theano import tensor as T

import timeit
import sys

import recurrent
from optim import adam


# -----------------------------------------------------------------------------
# Common copy task
# n_in is the number of words + 2 (one for pause, one for copy)
# the n_in-2 words are randomly chosen and 1-hot encoded sequence_length times
# then the blank character is input pause times
# then the copy character is input (once)
# then the output repeats the original sequence (minus the pause & copy words)
# the input during the copy is the blank character again
# -----------------------------------------------------------------------------

def data(n_in,n_train,n_val,sequence_length,pause):
    rng = numpy.random.RandomState(1)
    def generate_data(examples):
        x = numpy.zeros((examples,2*sequence_length+pause+1,n_in),dtype='float32')
        y = numpy.zeros((examples,2*sequence_length+pause+1,n_in-1),dtype='float32')
        for ex in range(examples):
            # original sequence
            oneloc = rng.randint(0,n_in-2,size=(sequence_length))
            x[ex,numpy.arange(sequence_length),oneloc] = 1
            # blank characters before copy
            x[ex,sequence_length+numpy.arange(pause),n_in-2] = 1
            # copy character
            x[ex,sequence_length+pause,n_in-1] = 1
            # blank characters during copy
            x[ex,sequence_length+pause+1+numpy.arange(sequence_length),n_in-2] = 1
            # output is blank character until copy character is input
            y[ex,numpy.arange(sequence_length+pause+1),n_in-2] = 1
            # repeat the original sequence
            y[ex,sequence_length+pause+1+numpy.arange(sequence_length),oneloc] = 1
        return x,y
    x_train,y_train = generate_data(n_train)
    x_val,y_val = generate_data(n_val)
    return [x_train,y_train,x_val,y_val]


# -----------------------------------------------------------------------------
# Experiments
# -----------------------------------------------------------------------------

def l2(x):
    return T.sqrt(T.mean(T.square(x)))
    
def l1(x):
    return T.mean(T.abs_(x))

def experiment(model,lr,lr_decay,batch_size,n_train,n_val,patience):
    seqlens = []
    train_loss = []
    grad_l2 = []
    grad_l1 = []
    weight_l2 = []
    weight_l1 = []
    preact = []
    val_loss = []
    
    best_val = numpy.inf
    epoch = 0
    train_idx = range(n_train)
    
    # set up rest of computational graph
    x = model.x
    y = T.tensor3('y')
    learning_rate = T.scalar('learning_rate')
    loss = model.crossentropy(y)
    grads = [T.grad(loss,param) for param in model.params]
    train_updates = adam(learning_rate,
                         model.params,
                         grads)
    train_fcn = theano.function(inputs=[x,y,learning_rate],
                                outputs=[loss,
                                         T.mean([l2(g) for g in grads]),
                                         T.mean([l1(g) for g in grads]),
                                         T.mean([l2(w) for w in model.W]),
                                         T.mean([l1(w) for w in model.W]),
                                         model.preact],
                                updates=train_updates)
    test_fcn = theano.function(inputs=[x,y],
                               outputs=loss)
    
    seqlen = 1
    init_patience = patience
    while patience>0:
        start_time = timeit.default_timer()
        
        # re-generate data at each epoch (essential as seqlen>10)
        x_train,y_train,x_val,y_val = data(n_in,n_train,n_val,seqlen,0)   
        
        # train
        loss_epoch = 0
        grad_l2_epoch = 0
        grad_l1_epoch = 0
        numpy.random.shuffle(train_idx)
        n_train_batches = int(numpy.floor(x_train.shape[0]/batch_size))
        for batch in range(n_train_batches):
            batch_idx = train_idx[batch*batch_size:(batch+1)*batch_size]
            x_batch = x_train[batch_idx]
            y_batch = y_train[batch_idx]
            loss_batch,grads_l2_batch,grads_l1_batch,weights_l1,weights_l2, \
                preact_batch = train_fcn(x_batch,y_batch,lr)
            loss_epoch += loss_batch
            grad_l2_epoch += numpy.square(grads_l2_batch)
            grad_l1_epoch += grads_l1_batch
            preact.append([p for p in preact_batch])
            
        loss_epoch = loss_epoch/n_train_batches
        grad_l2_epoch = numpy.sqrt(grad_l2_epoch/n_train_batches)
        grad_l1_epoch = grad_l1_epoch/n_train_batches
        end_time = timeit.default_timer()
        print('Epoch %d  -----  sequence: %i  -----  time per example (msec): %f' \
             % (epoch,seqlen,1000*(end_time-start_time)/x_train.shape[0]))
        print('  Training loss  = %f' % loss_epoch)
        print(grad_l1_epoch)
        print(grad_l2_epoch)
        print(weights_l1)
        print(weights_l2)
#        print('    Grads:    L1 = %f    L2 = %f' %(grad_l1_epoch,grad_l2_epoch))
#        print('    Weights:  L1 = %f    L2 = %f' %(weights_l1,weights_l2))
        sys.stdout.flush() # force print to appear
        seqlens.append(seqlen)
        train_loss.append(loss_epoch)
        grad_l2.append(grad_l2_epoch)
        grad_l1.append(grad_l1_epoch)
        weight_l2.append(weights_l1)
        weight_l1.append(weights_l2)
        
        # validate
        val_loss_epoch = 0
        n_val_batches = int(numpy.floor(x_val.shape[0]/batch_size))
        for batch in range(n_val_batches):
            x_batch = x_val[batch*batch_size:(batch+1)*batch_size]
            y_batch = y_val[batch*batch_size:(batch+1)*batch_size]
            val_loss_epoch += test_fcn(x_batch,y_batch)
        val_loss_epoch = val_loss_epoch/n_val_batches
        print('  Validation loss = %f' % val_loss_epoch)
        sys.stdout.flush() # force print to appear
        val_loss.append(val_loss_epoch)
        
        if val_loss_epoch<best_val:
            best_val = val_loss_epoch
            patience = init_patience
        else:
            patience -= 1
        
        # increase seqlen once it gets good enough
        if val_loss_epoch<0.15:
            patience = init_patience
            best_val = numpy.inf
            seqlen += 1
            x_train,y_train,x_val,y_val = data(n_in,n_train,n_val,seqlen,0)
            print('==========================================================')
            print('              Increasing sequence length')
            print('==========================================================')
        
        # set up next epoch
        epoch += 1
        lr = lr*lr_decay
    
    return seqlens, train_loss, val_loss, grad_l2_epoch, grad_l1_epoch, \
           weight_l2, weight_l1, preact


def log_results(filename,header,results):
    import csv
    import os
    if not filename[-4:]=='.csv':
        filename = filename+'.csv'
    
    writeheader = False
    
    # check that the header of the file matches the input
    if os.path.isfile(filename):
        file = open(filename,'r')
        reader = csv.reader(file,delimiter=',')
        fileheader = reader.next()
        file.close()
        if fileheader!=header:
            os.remove(filename)
            writeheader = True
    else:
        writeheader = True
    
    file = open(filename,'a')
    writer = csv.writer(file)
    if writeheader:
        writer.writerow(header)
    writer.writerow(results)
    file.close()


def test_hornn(n_in,n_hidden,n_out,activation,order,
                  lr,lr_decay,n_train,n_val,batch_size,patience):
    x = T.tensor3('x')
    model = recurrent.hornn(x,n_in,n_hidden,n_out,activation,order)
    return experiment(model,lr,lr_decay,batch_size,n_train,n_val,patience)


if __name__ == "__main__":
    import graph
    import argparse
    parser = argparse.ArgumentParser(description='Run higher-order RNN experiments')
    parser.add_argument('--activation',nargs=1,type=str,
                        default='tanh')
    parser.add_argument('--order',nargs=1,type=int,
                        default=1)
    parser.add_argument('--learnrate',nargs=1,type=float,
                        default=7e-5)
    activation = parser.parse_args().activation
    order = parser.parse_args().order
    lr = parser.parse_args().learnrate
    
    # make some data
    n_in = 4 # one-hot encoding, n_in-2 words + pause + copy
    n_out = n_in-1 # n_in-2 words + blank
    n_train = 20*256
    n_val = 256
    
    # experiment params
    lr_decay = 0.995
    patience = 500
    batch_size = 256 # from paper
    n_hidden = 256 # from paper
    
    seqlens, loss, val_loss, grad_l2_epoch, grad_l1_epoch, weight_l2, \
        weight_l1, preact = test_hornn(n_in,n_hidden,n_out,activation,order,lr,
                                       lr_decay,n_train,n_val,batch_size,patience)
    
    # log results
    filename = 'hornn_'+activation+'_order'+str(order)
    graph.make_all(filename,'DNI_scale')
    












