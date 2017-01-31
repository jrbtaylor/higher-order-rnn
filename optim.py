# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 14:45:39 2016

@author: jrbtaylor
"""

import numpy
import theano
from theano import tensor as T

def adam(lr,params,grads):
    """
    Adam optimization
    
    Parameters
    ----------
    lr: Theano SharedVariable
        Initial learning rate
    params: list of Theano SharedVariable
        Model parameters
    grads: list of Theano variables
        Gradients of cost w.r.t. parameters
    
    """
    # default values
    epsilon = 1e-8
    b1 = 0.9
    b2 = 0.999
    
    def floatX(data):
        return numpy.asarray(data,dtype=theano.config.floatX)
    
    updates = []
    t = theano.shared(floatX(1.))
    t_new = t+1
    updates.append((t,t_new))
    for p,g in zip(params,grads):
        m = theano.shared(p.get_value()*floatX(0.))
        m_new = b1*m+(1-b1)*g
        updates.append((m,m_new))
        
        v = theano.shared(p.get_value()*floatX(0.))
        v_new = b2*v+(1-b2)*g**2
        updates.append((v,v_new))
        
        mhat = m_new/(1-b1**t)
        vhat = v_new/(1-b2**t)
        updates.append((p,p-lr*mhat/(T.sqrt(vhat)+epsilon)))
        
    return updates