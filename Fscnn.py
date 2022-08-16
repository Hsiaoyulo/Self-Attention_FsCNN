import tensorflow as tf
# from tensorflow import keras
import numpy as np
import skfuzzy as fuzz
# from attention_keras import *
# from keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)
# K.set_session(session)

config = tf.compat.v1.ConfigProto() 
config.gpu_options.per_process_gpu_memory_fraction = 0.9 # 占用GPU90%的顯存 
session = tf.compat.v1.Session(config=config)

def Fuzz_dataset(words_data):
        # Generate trapezoidal membership function on range [0, 1]
        x_r = np.arange(-0.9999, 1, 0.1)

        # Generate fuzzy membership functions
        f_p = fuzz.trimf(x_r, [-0.1, 0.5, 1])
        f_n = fuzz.trimf(x_r, [-1, -0.5, 0.1])
        with tf.compat.v1.Session() as sess: 
            sess.run(tf.initialize_all_variables())
            array = words_data.eval(sess)
        z = 0
        k = 0
        for i in range(array.shape[0]):
            z += 1   
            k=0
            for j in range(array.shape[1]):  
                k += 1

        p_level = [list(range(k)) for _ in range(z)]
        p_original = [list(range(k)) for _ in range(z)]
        n_level = [list(range(k)) for _ in range(z)]
        n_original = [list(range(k)) for _ in range(z)]

       # We need the activation of our fuzzy membership functions at these values.
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                p_level[i][j] = fuzz.interp_membership(x_r, f_p, array[i][j])
                n_level[i][j] = fuzz.interp_membership(x_r, f_n, array[i][j])
                if (p_level[i][j]>n_level[i][j]):                    
                    p_original[i][j] = array[i][j]                   
                    n_level[i][j] = 0
                    n_original[i][j] = 0

                elif (p_level[i][j]<=n_level[i][j]):
                    n_original[i][j] = array[i][j]
                    p_level[i][j] = 0
                    p_original[i][j] = 0


        p = np.array(p_level)
        p= p.astype(np.float32)
        p_o = np.array(p_original)
        p_or= p_o.astype(np.float32)
        n = np.array(n_level) 
        n= n.astype(np.float32)
        n_o = np.array(n_original)
        n_or= n_o.astype(np.float32)        
        return p, n, p_or, n_or 
    
class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.compat.v1.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.compat.v1.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.compat.v1.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        
        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random.uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.p, self.n, self.p_o, self.n_o = Fuzz_dataset(self.W)
            
            self.embedded_chars_p = tf.nn.embedding_lookup(self.p, self.input_x)
            self.embedded_chars_expanded_p = tf.expand_dims(self.embedded_chars_p, -1)
            
            self.embedded_chars_n = tf.nn.embedding_lookup(self.n, self.input_x)
            self.embedded_chars_expanded_n = tf.expand_dims(self.embedded_chars_n, -1)
            
            self.embedded_chars_p_o = tf.nn.embedding_lookup(self.p_o, self.input_x)
            self.embedded_chars_expanded_p_o = tf.expand_dims(self.embedded_chars_p_o, -1)
            
            self.embedded_chars_n_o = tf.nn.embedding_lookup(self.n_o, self.input_x)
            self.embedded_chars_expanded_n_o = tf.expand_dims(self.embedded_chars_n_o, -1)
            
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs_p = []
        pooled_outputs_p_o = []
        pooled_outputs_n = []
        pooled_outputs_n_o = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.random.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv_p = tf.nn.conv2d(
                    self.embedded_chars_expanded_p,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",#Valid
                    name="conv_p")
                # Apply nonlinearityleaky_
#                 h_p = tf.nn.relu(tf.nn.bias_add(conv_p, b), name="relu")
#                 h_p = tf.nn.leaky_relu(tf.nn.bias_add(conv_p, b), name="leakyrelu")
                h_p = tf.nn.selu(tf.nn.bias_add(conv_p, b), name="selu")
#                 h_p = tf.nn.elu(tf.nn.bias_add(conv_p, b), name="elu")

                pooled_p = tf.nn.max_pool2d(
                    h_p,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",#Valid
                    name="pool_p") 
                
                # Apply nonlinearity 
                pooled_outputs_p.append(pooled_p)
                
                
                conv_p_o = tf.nn.conv2d(
                    self.embedded_chars_expanded_p_o,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",#Valid
                    name="conv_p_o")
                # Apply nonlinearityleaky_
#                 h_p_o = tf.nn.relu(tf.nn.bias_add(conv_p_o, b), name="relu")
#                 h_p_o = tf.nn.leaky_relu(tf.nn.bias_add(conv_p_o, b), name="leakyrelu")
                h_p_o = tf.nn.selu(tf.nn.bias_add(conv_p_o, b), name="selu")
#                 h_p_o = tf.nn.elu(tf.nn.bias_add(conv_p_o, b), name="elu")
                

                pooled_p_o = tf.nn.max_pool2d(
                    h_p_o,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",#Valid
                    name="pool_p") 
                # Apply nonlinearity 
                pooled_outputs_p_o.append(pooled_p_o)
                
                conv_n = tf.nn.conv2d(
                    self.embedded_chars_expanded_n,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",#Valid
                    name="conv_n")
                # Apply nonlinearity
#                 h_n = tf.nn.relu(tf.nn.bias_add(conv_n, b), name="relu")
#                 h_n = tf.nn.leaky_relu(tf.nn.bias_add(conv_n, b), name="leakyrelu")
                h_n = tf.nn.selu(tf.nn.bias_add(conv_n, b), name="selu")
#                 h_n = tf.nn.elu(tf.nn.bias_add(conv_n, b), name="elu")
                
                pooled_n = tf.nn.max_pool2d(
                    h_n,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",#Valid
                    name="pool_n")                  
                # Maxpooling over the outputs
                pooled_outputs_n.append(pooled_n)
                
                conv_n_o = tf.nn.conv2d(
                    self.embedded_chars_expanded_n_o,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",#Valid
                    name="conv_p_o")
                # Apply nonlinearityleaky_
#                 h_n_o = tf.nn.relu(tf.nn.bias_add(conv_n_o, b), name="relu")
#                 h_n_o = tf.nn.leaky_relu(tf.nn.bias_add(conv_n_o, b), name="leakyrelu")
                h_n_o = tf.nn.selu(tf.nn.bias_add(conv_n_o, b), name="selu")
#                 h_n_o = tf.nn.elu(tf.nn.bias_add(conv_n_o, b), name="elu")
                
                pooled_n_o = tf.nn.max_pool2d(
                    h_n_o,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",#Valid
                    name="pool_p") 
                # Apply nonlinearity 
                pooled_outputs_n_o.append(pooled_n_o)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool_p = tf.concat(pooled_outputs_p, 3)
        self.h_pool_flat_p = tf.reshape(self.h_pool_p, [-1, num_filters_total])
        self.h_pool_p_o = tf.concat(pooled_outputs_p_o, 3)
        self.h_pool_flat_p_o = tf.reshape(self.h_pool_p_o, [-1, num_filters_total])
        
        self.h_pool_n = tf.concat(pooled_outputs_n, 3)
        self.h_pool_flat_n = tf.reshape(self.h_pool_n, [-1, num_filters_total])
        self.h_pool_n_o = tf.concat(pooled_outputs_n_o, 3)
        self.h_pool_flat_n_o = tf.reshape(self.h_pool_n_o, [-1, num_filters_total])
        
        # defuzzification
            
        tf_multiply_p = tf.multiply(self.h_pool_flat_p, self.h_pool_flat_p_o)
        tf_multiply_n = tf.multiply(self.h_pool_flat_n, self.h_pool_flat_n_o)
        tf_add_fraction = tf.add(tf_multiply_p, tf_multiply_n)
        tf_add_denominator = tf.add(self.h_pool_flat_p, self.h_pool_flat_n)

        self.defuzz_pool = tf.divide(tf_add_fraction, tf_add_denominator)
       # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.defuzz_pool, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.compat.v1.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                 initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.compat.v1.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)  + l2_reg_lambda * l2_loss
    
        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")
        
        #cv test value
        with tf.name_scope('num_correct'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.num_correct = tf.reduce_sum(tf.cast(correct_predictions, 'float'), name='num_correct')
                    