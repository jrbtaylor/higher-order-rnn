# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 23:45:51 2017

@author: jrbtaylor
"""

import numpy

import theano
from theano import tensor as T
from theano.tensor import tanh
from theano.tensor.nnet import sigmoid, relu
from theano.tensor.nnet.nnet import softmax, categorical_crossentropy

rng = numpy.random.RandomState(1)

# from http://deeplearning.net/tutorial/code/lstm.py
def ortho_weight(ndim,rng=rng):
    W = rng.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return theano.shared(u.astype(theano.config.floatX),borrow=True)

def const_bias(n,value=0):
    return theano.shared(value*numpy.ones((n,),dtype=theano.config.floatX),
                         borrow=True)

def uniform_weight(n1,n2,rng=rng):
    limit = numpy.sqrt(6./(n1+n2))
    return theano.shared((rng.uniform(low=-limit,
                                      high=limit,
                                      size=(n1,n2))
                         ).astype(theano.config.floatX),borrow=True)

def layer_norm(h,scale=1,shift=0,eps=1e-5):
    mean = T.mean(h,axis=1,keepdims=True,dtype=theano.config.floatX)
#    std = T.std(h,axis=1,keepdims=True)
    std = T.mean(T.abs_(h-T.mean(h)),axis=1,keepdims=True)
    normed = (h-mean)/(eps+std)
    return scale*normed+shift
    
def dropout(h,p,rng=rng):
    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(99999))
    mask = T.cast(srng.binomial(n=1,p=1-p,size=h.shape),theano.config.floatX)
    # rescale activations at train time to avoid rescaling weights at test
    h = h/(1-p)
    return h*mask

def zoneout(h_t,h_tm1,p,rng=rng):
    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(99999))
    mask = T.cast(srng.binomial(n=1,p=1-p,size=h_t.shape),theano.config.floatX)
    return h_t*mask+h_tm1*(1-mask)
        

class hornn(object):
    def __init__(self,x,n_in,n_hidden,n_out,
                 activation='tanh',order=1):
        self.x = x
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.order = order
        
        if activation.lower()=='tanh':
            act = tanh
        elif activation.lower()=='relu':
            act = relu
        elif activation.lower()=='sigmoid':
            act = sigmoid
        elif activation.lower()=='linear':
            act = lambda x: x
        
        def _slice(x,n):
            return x[:,n*self.n_hidden:(n+1)*self.n_hidden]
        
        # initialize weights
        def ortho_weight(ndim,rng=rng):
            W = rng.randn(ndim, ndim)
            u, s, v = numpy.linalg.svd(W)
            return u.astype(theano.config.floatX)
        def uniform_weight(n1,n2,rng=rng):
            limit = numpy.sqrt(6./(n1+n2))
            return rng.uniform(low=-limit,high=limit,size=(n1,n2)).astype(theano.config.floatX)
        def const_bias(n,value=0):
            return value*numpy.ones((n,),dtype=theano.config.floatX)
        
        if self.order==0:
            # no multiplicative terms
            self.Wx = theano.shared(uniform_weight(n_in,n_hidden),borrow=True)
            self.Wh = theano.shared(ortho_weight(n_hidden),borrow=True)
            self.bh = theano.shared(const_bias(n_hidden,0),borrow=True)
            
            self.Wy = theano.shared(uniform_weight(n_hidden,n_out),borrow=True)
            self.by = theano.shared(const_bias(n_out,0),borrow=True)
            
            self.am = []
            self.ax = []
            self.ah = []
            
            self.params = [self.Wx,self.Wh,self.bh,self.Wy,self.by]
            self.W = [self.Wx,self.Wh,self.Wy]
            self.L1 = numpy.sum([abs(w).sum() for w in self.W])
            self.L2 = numpy.sum([(w**2).sum() for w in self.W])
            
            # forward function
            def forward(x_t,h_tm1,Wx,Wh,bh,am,ax,ah,Wy,by):
                preact = T.dot(x_t,Wx)+T.dot(h_tm1,Wh)+bh
                h_t = act(preact)
                y_t = softmax(T.dot(h_t,Wy)+by)
                return h_t,y_t,preact
        else:
            self.Wx = theano.shared(numpy.concatenate(
                        [uniform_weight(n_in,n_hidden) for i in range(order)],
                         axis=1),borrow=True)
            self.Wh = theano.shared(numpy.concatenate(
                        [ortho_weight(n_hidden) for i in range(order)],
                         axis=1),borrow=True)
            self.am = theano.shared(numpy.concatenate(
                        [const_bias(n_hidden,2) for i in range(order)],
                         axis=0),borrow=True)
            self.ax = theano.shared(numpy.concatenate(
                        [const_bias(n_hidden,0.5) for i in range(order)],
                         axis=0),borrow=True)
            self.ah = theano.shared(numpy.concatenate(
                        [const_bias(n_hidden,0.5) for i in range(order)],
                         axis=0),borrow=True)
            self.bh = theano.shared(numpy.concatenate(
                        [const_bias(n_hidden,0) for i in range(order)],
                         axis=0),borrow=True)
            
            self.Wy = theano.shared(uniform_weight(n_hidden,n_out),borrow=True)
            self.by = theano.shared(const_bias(n_out,0),borrow=True)
            
            self.params = [self.Wx,self.Wh,self.am,self.ax,self.ah,self.bh,
                           self.Wy,self.by]
            self.W = [self.Wx,self.Wh,self.Wy]
            self.L1 = numpy.sum([abs(w).sum() for w in self.W])
            self.L2 = numpy.sum([(w**2).sum() for w in self.W])   
            
            # forward function
            def forward(x_t,h_tm1,Wx,Wh,bh,am,ax,ah,Wy,by):
                h_t = 1
                preact = am*T.dot(x_t,Wx)*T.dot(h_tm1,Wh) \
                        +ax*T.dot(x_t,Wx) \
                        +ah*T.dot(h_tm1,Wh) \
                        +bh
                for i in range(self.order):
                    h_t = h_t*act(_slice(preact,i))
                y_t = softmax(T.dot(h_t,Wy)+by)
                return h_t,y_t,preact
        h0 = T.alloc(T.zeros((self.n_hidden,),dtype=theano.config.floatX),x.shape[0],self.n_hidden)
        ([h,y,p],updates) = theano.scan(fn=forward,
                                      sequences=x.dimshuffle([1,0,2]),
                                      outputs_info=[dict(initial=h0,taps=[-1]),
                                                    None,
                                                    None],
                                      non_sequences=[self.Wx,self.Wh,self.bh,
                                                     self.am,self.ax,self.ah,
                                                     self.Wy,self.by])
        self.output = y
        self.preact = p
        self.pred = T.argmax(self.output,axis=1)
    
    # ----- Classification -----
    def crossentropy(self,y):
        return T.mean(categorical_crossentropy(self.output,y.dimshuffle([1,0,2])))
    
    def errors(self,y):
        return T.mean(T.neq(self.pred,y.dimshuffle([1,0,2])))








