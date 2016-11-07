#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Impelementation of a soft attention layer
"""

from keras import backend as K
from keras.layers import Layer

class InnerAttention(Layer):
    """Applies soft attention from x to y, i.e. every element of the
       output, z_i, is combination of subelements of x that y_i attends
       to.  Let:
        - x ∈ R^{m x d} and y ∈ R^{n x d},
        - attention matrix e = x y^T ∈ R^{m x n}.
        - returns softmax(e) y ∈ R^{n x d}
    """

    def __init__(self, **kwargs):
        super(InnerAttention, self).__init__(**kwargs)

    def call(self, xy, mask=None):
        if not isinstance(xy, list) or len(xy) != 2:
            raise Exception('Inner attention must be called on 2 inputs.'
                            ' Got: ' + str(xy))
        x, y = xy
        assert K.ndim(x) == 3, "x should be 3d (m x d), but got %d"%(K.ndim(x))
        assert K.ndim(y) == 3, "y should be 3d (n x d), but got %d"%(K.ndim(y))
        #assert d1 == d2, "x and y should be of same dimension, but dim(x)=%d, dim(y)=%d"%(d1, d2)

        z = K.batch_dot(x, y, axes=2)

        # Softmax
        e = K.exp(z - K.max(z, axis=-1, keepdims=True))
        s = K.sum(e, axis=-1, keepdims=True)
        z = e / s

        z = K.batch_dot(z, x, axes=1)

        return z

    def get_output_shape_for(self, input_shapes):
        assert input_shapes and len(input_shapes) == 2
        return input_shapes[1]

def test_inner_attention():
    import numpy as np
    from numpy import eye, array
    from numpy.random import randn
    from keras.layers import Input
    from keras.models import Model

    N1, D = 3, 4
    N2, D = 2, 4

    x1 = Input(shape=(N1,D))
    x2 = Input(shape=(N2,D))
    z = InnerAttention()([x1, x2])
    model = Model(input=[x1,x2], output=[z])

    def compute_attention(x1, x2):
        e = x1.dot(x2.T)
        e = np.exp(e - np.max(e, axis=-1, keepdims=True))
        s = np.sum(e, axis=-1, keepdims=True)
        z = e / s
        return z.T.dot(x1)

    for x1, x2 in [(eye(4)[:N1,:], eye(4)[:N2,:]),
                   (randn(N1,D), randn(N2,D))]:
        z = compute_attention(x1, x2)
        z_ = model.predict([array([x1]),array([x2])])
        assert np.allclose(z, z_), "Implementation of soft attention doesn't match defintion"
