import tensorflow as tf
import numpy as np
from tensorflow.python.ops import gen_state_ops
from tqdm import tqdm
import itertools

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


def get_staggered_im(im, sequence_length, size, use_ravel=True):
    assert sequence_length == int(np.sqrt(sequence_length))**2
    stag = []
    for j in range(int(np.sqrt(sequence_length))):
        for k in range(int(np.sqrt(sequence_length))):
            s = im[j*(size/splits):(j+1)*(size/splits), k*(size/splits):(k+1)*(size/splits)]
            stag.append(s if not use_ravel else s.ravel())
    return stag 

# def get_images_dumb(bsize, size=4, splits=2):
#     X = []
#     Xstag = []
#     y = []
#     for i in range(bsize):
#         pos = np.random.random() < 0.5
#         im = np.zeros((size,size)) if pos else np.ones((size,size))
#         Xstag.append(get_staggered_im(im, size, splits))
#         X.append(im)
#         label = np.zeros(2)
#         label[pos] = 1
#         y.append(label)
#     return np.array(Xstag), np.array(X), np.array(y)

# def get_rectangle_images_size(bsize, size=8, splits=2):
#     X = []
#     Xstag = []
#     y = []
#     for i in range(bsize):
#         pos = np.random.random() < 0.5
#         im = np.zeros((size,size))
#         if pos:
#             im[1:size-1, 1:size-1] = 1
#         else:
#             im[size/2-1:size/2+1, size/2-1:size/2+1] = 1
        
#         Xstag.append(get_staggered_im(im, size, splits))
#         X.append(im)
#         label = np.zeros(2)
#         label[pos] = 1
#         y.append(label)
#     return np.array(Xstag), np.array(X), np.array(y)

def get_lrb_images(size, bsize, sequence_length):
    X = []
    y = []
    for i in range(bsize):
        im = np.zeros((size,size))
        num = np.random.random()
        if num <= 0.25:
            l = 0
        if num > 0.25 and num <= 0.5:
            l = 0
            im[1:size-1,1:2] = 1
        elif num > 0.5 and num <= 0.75:
            l = 0
            im[1:size-1,-2:-1] = 1
        elif num > 0.75:
            l = 1
            im[1:size-1,1:2] = 1
            im[1:size-1,-2:-1] = 1
        X.append([np.ravel(im) for i in range(sequence_length)])
        label = np.zeros(2)
        label[l] = 1
        y.append(label)
    return np.array(X), np.array(y)



def get_center_bar_images(size, bsize, sequence_length):
    X = []
    y = []
    for i in range(bsize):
        pos = np.random.random() < 0.5
        im = np.zeros((size,size))
        if pos:
            im[size/8:7*(size/8), size/2-1:size/2+1] = 1
        X.append([np.ravel(im) for i in range(sequence_length)])
        label = np.zeros(2)
        label[pos] = 1
        y.append(label)
    return np.array(X), np.array(y)

def get_right_bar_images(size, bsize, sequence_length):
    X = []
    y = []
    for i in range(bsize):
        pos = np.random.random() < 0.5
        im = np.zeros((size,size))
        if pos:
            im[size/8:7*(size/8), size-1:size] = 1
        X.append([np.ravel(im) for i in range(sequence_length)])
        label = np.zeros(2)
        label[pos] = 1
        y.append(label)
    return np.array(X), np.array(y)

def get_sd_images(size, bsize, sequence_length, half_max_item ):

    # Makes same/different images
    # Images have two square bit patterns of side size
    # Bit patterns are placed about the image's vertical bisector
    # If s_or_d = s, then both are the same; if = d, then different
    X = []
    y = []

    for i in range(bsize):
        label = np.array([0,0])
        s_or_d = np.random.uniform() < .5
        s_or_d *=1
        half_item = np.random.randint(1,half_max_item)
        label[s_or_d] = 1
        im = np.zeros((size, size))

        vert_1 = np.random.randint(0,size-2*half_item + 1)
        vert_2 = np.random.randint(0,size-2*half_item + 1)

        offset = np.random.randint(half_item + 1, .5*size - half_item + 1)

        bit_p1 = np.round(np.random.uniform(0,1,size=(2*half_item, 2*half_item)))
        bit_p2 = np.round(np.random.uniform(0,1,size=(2*half_item, 2*half_item)))

        im[vert_1: vert_1 + 2* half_item, \
           .5*size - half_item - offset:.5*size + half_item - offset] = bit_p1
        im[vert_2 : vert_2 + 2* half_item ,\
           .5*size + offset -  half_item : .5*size + offset + half_item] = bit_p1*(1-s_or_d) + \
        bit_p2*s_or_d

        X.append([np.ravel(im) for j in range(sequence_length)])

        y.append(label)
    return np.array(X), np.array(y)

def get_square_detect_images(size, bsize, sequence_length):
    X = []
    y = []
    
    for i in range(bsize):
        label = np.array([0,0])
        p_or_a = np.random.uniform() < .5
        p_or_a *=1
        label[p_or_a] = 1

        canvas = np.zeros((size,size))

        if not p_or_a:
            X.append([np.ravel(canvas) for j in range(sequence_length)])
            y.append(label)
        else:
            square_side = np.floor(size/6.0)
            upper_left_corner = [np.random.randint(0,size-square_side), np.random.randint(0,size-square_side)]

            square = np.ones((square_side, square_side))
            canvas[upper_left_corner[0]:upper_left_corner[0] + square_side , upper_left_corner[1]:upper_left_corner[1] + square_side] = square

            X.append([np.ravel(canvas) for j in range(sequence_length)])
            y.append(label) 

    return np.array(X), np.array(y)


def get_2_square_detect_images(size, bsize, sequence_length):
    X = []
    y = []
    
    for i in range(bsize):
        label = np.array([0,0])
        two_squares = np.random.uniform() < .5
        two_squares *=1
        label[two_squares] = 1


        stamp1 = np.zeros((size,size))
	stamp2 = np.zeros((size,size))

	flag = 0

        if not two_squares:
            square_side = np.floor(size/6.0)
            upper_left_corner = [np.random.randint(0,size-square_side), np.random.randint(0,size-square_side)]

            square = np.ones((square_side, square_side))
            canvas[upper_left_corner[0]:upper_left_corner[0] + square_side , upper_left_corner[1]:upper_left_corner[1] + square_side] = square

            X.append([np.ravel(canvas) for j in range(sequence_length)])
            y.append(label) 

        else:

	    while flag == 0:
                canvas = np.zeros((size,size))
                square_side = np.floor(size/6.0)
            	square= np.ones((square_side, square_side))

	        upper_left_corner1 = [np.random.randint(0,size-square_side), np.random.randint(0,size-square_side)]
	        upper_left_corner2 = [np.random.randint(0,size-square_side), np.random.randint(0,size-square_side)]


	        stamp1[upper_left_corner[0]:upper_left_corner[0] + square_side , upper_left_corner[1]:upper_left_corner[1] + square_side] = square
	        stamp1[upper_left_corner[0]:upper_left_corner[0] + square_side , upper_left_corner[1]:upper_left_corner[1] + square_side] = square

	        canvas += stamp1 + stamp2

	    if np.max(np.max(canvas)) == 1:
	    	X.append([np.ravel(canvas) for j in range(sequence_length)])
	    	y.append(label) 
		flag = 1
		
    return np.array(X), np.array(y)


def make_ims(params):

    if params["task"] == "center":
        Input, Target_Output = get_center_bar_images(size=params["input_side"], 
                                                       bsize=params["bsize"], sequence_length=params["sequence_length"])
    elif params["task"] == "square_detect":
        Input, Target_Output = get_square_detect_images(size=params["input_side"], 
                                                       bsize=params["bsize"], sequence_length=params["sequence_length"])
    elif params["task"] == "2_square_detect":
        Input, Target_Output = get_square_detect_images(size=params["input_side"], 
                                                       bsize=params["bsize"], sequence_length=params["sequence_length"])
    elif params["task"] == "right":
        Input, Target_Output = get_right_bar_images(size=params["input_side"], 
                                                       bsize=params["bsize"], sequence_length=params["sequence_length"])
    elif params["task"] == "sd":
        Input, Target_Output = get_sd_images(size=params["input_side"], 
                                               bsize=params["bsize"], 
                                               sequence_length=params["sequence_length"],
                                               half_max_item=params["half_max_item"])
    elif params["task"] == "lrb":
        Input, Target_Output = get_lrb_images(size=params["input_side"], 
                                                       bsize=params["bsize"], sequence_length=params["sequence_length"])
        
    Target_Output = np.hstack([np.expand_dims(Target_Output, 1) for i in range(Input.shape[1])])
    return Input, Target_Output


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


def apply_mask(X, row_v, col_v, f_mask, focus_range, focus_type):

    if focus_type == "rowcol":
        num_row = tf.shape(X)[0]
        num_col = tf.shape(X)[1] 

        tmp_row = tf.tile(tf.expand_dims(row_v,1), [1,num_col])
        tmp_col = tf.tile(tf.expand_dims(col_v,0), [num_row,1])

        mask = tf.multiply(tmp_row,tmp_col)
    elif focus_type == "mask":
        mask =  tf.reshape(f_mask, (-1, focus_range, focus_range)) 
    else:
        assert False, ("{} is not a valid focus type".format(self.focus_type)) 
    masked_X = tf.multiply(X, mask)
    

    return masked_X

def apply_spotlight(X,spotlight_row, spotlight_col, spotlight_sigma):
          
    # Center coordinates
    spotlight_row = spotlight_row -  12
    spotlight_col = spotlight_col -  12
    
    # Make axes
    x_axis = np.float32(range(-12,12))
    y_axis = np.float32(range(-12,12))    
    plane = np.float32(tuple(itertools.product(x_axis,y_axis)))

    tmp_id = tf.constant(np.identity(2,dtype=np.float32)) 
    
    # Gaussian parameters 
    mu = tf.concat(0,[spotlight_col,spotlight_row]) 
    sigma = spotlight_sigma*tmp_id
     
    dist = tf.contrib.distributions.MultivariateNormalFull(mu,sigma)
    spotlight = dist.pdf(plane)
    
    spotlight = tf.reshape(spotlight,(24,24))
    spotlit_X = tf.multiply(X,spotlight)
    
     
    return spotlit_X

def apply_spotlight_circle(X,spotlight_row, spotlight_col, spotlight_radius):
    
    height = tf.shape(X)[0]
    width = tf.shape(X)[1]

    # Center coordinates
    spotlight_row = spotlight_row -  12
    spotlight_col = spotlight_col -  12

    center = tf.to_float([spotlight_row, spotlight_col])
    
    # Make axes
    x_axis = np.float32(range(-12,12))
    y_axis = np.float32(range(-12,12))    
    plane = np.float32(tuple(itertools.product(x_axis,y_axis)))
    y_coords = plane[:,0]
    x_coords = plane[:,1]


    # Make circular spotlight

    spotlight = tf.sqrt((y_coords - center[0])**2 + (x_coords - center[1])**2) <= spotlight_radius
    spotlight = tf.to_float(tf.reshape(spotlight,(24,24)))
    spotlit_X = tf.multiply(X,spotlight)
    
     
    return spotlit_X



def get_updt(loss, learning_rate=1e-4, momentum=0.9, clip=10):
    opt_func = tf.train.AdamOptimizer(1e-6)
    # opt_func = tf.train.RMSPropOptimizer(learning_rate, momentum=momentum)
    print "computing gradients..."
    gvs = opt_func.compute_gradients(loss)
    grads = [(tf.clip_by_value(grad, -clip, clip), var)
                     for grad, var in tqdm(gvs) if not grad is None]
    print "applying gradients..."
    return opt_func.apply_gradients(grads)
