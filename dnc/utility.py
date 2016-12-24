import tensorflow as tf
import numpy as np
from tensorflow.python.ops import gen_state_ops
from tqdm import tqdm
def pairwise_add(u, v=None, is_batch=False):
    """
    performs a pairwise summation between vectors (possibly the same)

    Parameters:
    ----------
    u: Tensor (n, ) | (n, 1)
    v: Tensor (n, ) | (n, 1) [optional]
    is_batch: bool
        a flag for whether the vectors come in a batch
        ie.: whether the vectors has a shape of (b,n) or (b,n,1)

    Returns: Tensor (n, n)
    Raises: ValueError
    """
    u_shape = u.get_shape().as_list()

    if len(u_shape) > 2 and not is_batch:
        raise ValueError("Expected at most 2D tensors, but got %dD" % len(u_shape))
    if len(u_shape) > 3 and is_batch:
        raise ValueError("Expected at most 2D tensor batches, but got %dD" % len(u_shape))

    if v is None:
        v = u
    else:
        v_shape = v.get_shape().as_list()
        if u_shape != v_shape:
            raise VauleError("Shapes %s and %s do not match" % (u_shape, v_shape))

    n = u_shape[0] if not is_batch else u_shape[1]

    column_u = tf.reshape(u, (-1, 1) if not is_batch else (-1, n, 1))
    U = tf.concat(1 if not is_batch else 2, [column_u] * n)

    if v is u:
        return U + tf.transpose(U, None if not is_batch else [0, 2, 1])
    else:
        row_v = tf.reshape(v, (1, -1) if not is_batch else (-1, 1, n))
        V = tf.concat(0 if not is_batch else 1, [row_v] * n)

        return U + V


def decaying_softmax(shape, axis):
    rank = len(shape)
    max_val = shape[axis]

    weights_vector = np.arange(1, max_val + 1, dtype=np.float32)
    weights_vector = weights_vector[::-1]
    weights_vector = np.exp(weights_vector) / np.sum(np.exp(weights_vector))

    container = np.zeros(shape, dtype=np.float32)
    broadcastable_shape = [1] * rank
    broadcastable_shape[axis] = max_val

    return container + np.reshape(weights_vector, broadcastable_shape)


def get_staggered_im(im, size, splits, use_ravel=True):
    stag = []
    for j in range(splits):
        for k in range(splits):
            s = im[j*(size/splits):(j+1)*(size/splits), k*(size/splits):(k+1)*(size/splits)]
            stag.append(s if not use_ravel else s.ravel())
    return stag 

def get_images_dumb(bsize, size=4, splits=2):
    X = []
    Xstag = []
    y = []
    for i in range(bsize):
        pos = np.random.random() < 0.5
        im = np.zeros((size,size)) if pos else np.ones((size,size))
        Xstag.append(get_staggered_im(im, size, splits))
        X.append(im)
        label = np.zeros(2)
        label[pos] = 1
        y.append(label)
    return np.array(Xstag), np.array(X), np.array(y)

def get_rectangle_images_size(bsize, size=8, splits=2):
    X = []
    Xstag = []
    y = []
    for i in range(bsize):
        pos = np.random.random() < 0.5
        im = np.zeros((size,size))
        if pos:
            im[1:size-1, 1:size-1] = 1
        else:
            im[size/2-1:size/2+1, size/2-1:size/2+1] = 1
        
        Xstag.append(get_staggered_im(im, size, splits))
        X.append(im)
        label = np.zeros(2)
        label[pos] = 1
        y.append(label)
    return np.array(Xstag), np.array(X), np.array(y)

def get_lrb_images(bsize, size=8, splits=2):
    X = []
    Xstag = []
    y = []
    for i in range(bsize):
        im = np.zeros((size,size))
        num = np.random.random()
        if num <= 0.25:
            l = 0
        if num > 0.25 and num <= 0.5:
            l = 1
            im[1:size-1,1:2] = 1
        elif num > 0.5 and num <= 0.75:
            l = 2
            im[1:size-1,-2:-1] = 1
        elif num > 0.75:
            l = 3
            im[1:size-1,1:2] = 1
            im[1:size-1,-2:-1] = 1
        Xstag.append(get_staggered_im(im, size, splits))
        X.append(im)
        label = np.zeros(4)
        label[l] = 1
        y.append(label)
    return np.array(Xstag), np.array(X), np.array(y)


def binary_cross_entropy(predictions, targets):

    return tf.reduce_mean(
        -1 * targets * tf.log(predictions) - (1 - targets) * tf.log(1 - predictions)
    )

def get_im_sequence(batch_x, batch_y):
    b = np.reshape(batch_x, (-1, 28,28))

    X = np.reshape(np.vstack((b for b in 
                              (b[:, :7,:7], b[:, :7,7:14], b[:, :7,14:21], b[:, :7,21:28],  b[:, 7:14,:7],  b[:, 7:14,7:14],  
               b[:, 7:14,14:21], b[:, 7:14,21:28], b[:, 14:21,:7], b[:, 14:21,7:14], b[:, 14:21,14:21], 
               b[:, 14:21,21:28], b[:, 21:28,:7], b[:, 21:28,7:14], b[:, 21:28,14:21], b[:, 21:28,21:28]))), 
                   (-1, 16, 49))
    return X, batch_y

def window_vector(X, ind_1, ind_2, window_size=3):
    return X[:, tf.maximum(0, ind_1):tf.minimum(ind_1 + window_size, tf.shape(X)[1]), 
                tf.maximum(0, ind_2):tf.minimum(ind_2 + window_size, tf.shape(X)[1])]    


def get_updt(loss, learning_rate=1e-4, momentum=0.9, clip=10):
    opt_func = tf.train.AdamOptimizer(1e-6)
    # opt_func = tf.train.RMSPropOptimizer(learning_rate, momentum=momentum)
    print "computing gradients..."
    gvs = opt_func.compute_gradients(loss)
    grads = [(tf.clip_by_value(grad, -clip, clip), var)
                     for grad, var in tqdm(gvs) if not grad is None]
    print "applying gradients..."
    return opt_func.apply_gradients(grads)
