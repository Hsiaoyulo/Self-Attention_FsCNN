#! /usr/bin/env python

import tensorflow as tf
import tensorflow.compat.v1 as tfcv
import numpy as np
import os
import time
import datetime
import data_helpers
import matplotlib.pyplot as plt
import random
from Fscnn import *
import pandas as pd
# from cnn_attention import *

from attentions import *
from tensorflow import keras
from tensorflow.keras.optimizers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.models import Sequential, Model
from torch.utils.tensorboard import SummaryWriter
from array import array

from tensorflow.contrib import learn
from tensorflow.keras.layers import Embedding,Input

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold 
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
#gpu_options = tf.GPUOptions(allow_growth=True)
sess = tfcv.Session(config=tfcv.ConfigProto(device_count = {'GPU': 0}))
# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
# tf.flags.DEFINE_string("positive_data_file", "./Dataset/IMDB/plot.tok.gt9.5000", "Data source for the train data.")
# tf.flags.DEFINE_string("negative_data_file", "./Dataset/IMDB/quote.tok.gt9.5000", "Data source for the test data.")

# tf.flags.DEFINE_string("positive_data_file", "./Dataset/SST2/stsa.binarypos.train", "Data source for the positive data.")
# tf.flags.DEFINE_string("negative_data_file", "./Dataset/SST2/stsa.binaryneg.train", "Data source for the negative data.")

# tf.flags.DEFINE_string("positive_data_file", "./Dataset/MR/rt-polarity.pos", "Data source for the positive data.")
# tf.flags.DEFINE_string("negative_data_file", "./Dataset/MR/rt-polarity.neg", "Data source for the negative data.")

# tf.flags.DEFINE_string("positive_data_file", "./Dataset/MPQA/mpqa.pos", "Data source for the positive data.")
# tf.flags.DEFINE_string("negative_data_file", "./Dataset/MPQA/mpqa.neg", "Data source for the negative data.")

# tf.flags.DEFINE_string("positive_data_file", "./Dataset/SUBJ/subj.objective", "Data source for the positive data.")
# tf.flags.DEFINE_string("negative_data_file", "./Dataset/SUBJ/subj.subjective", "Data source for the negative data.")

tf.flags.DEFINE_string("positive_data_file", "./Dataset/CR/custrev.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./Dataset/CR/custrev.neg", "Data source for the negative data.")

# tf.flags.DEFINE_string("positive_data_file", "./Dataset/finance/LoughranMcDonald_Positive.txt", "Data source for the positive data.")
# tf.flags.DEFINE_string("negative_data_file", "./Dataset/finance/LoughranMcDonald_Negative.txt", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 3.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 500)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
#tf.flags.DEFINE_boolean("fine_tune_batch_norm ", False, " set batch size as large as possible")

FLAGS = tfcv.flags.FLAGS

def preprocess():
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file) 

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))    
    
    #if you need caculate cv test value then split the original dataset into train and test sets
    x_, x_test, y_, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

    cv = KFold(n_splits=10, random_state=42, shuffle=True)
    for train_index, dev_index in cv.split(x_):
        x_train, x_dev, y_train, y_dev = x[train_index], x[dev_index], y[train_index], y[dev_index]

    del x, y, cv
    
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev/Test split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    #print(dev_sample_index), x_test, y_test
    return x_train, y_train, vocab_processor, x_dev, y_dev, x_test, y_test

def train(x_train, y_train, vocab_processor, x_dev, y_dev, x_test, y_test):
    # Training
    # ==================================================

    with tf.Graph().as_default():
        session_conf = tf.compat.v1.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.compat.v1.Session(config=session_conf)
        x = np.array((x_train))
        y = np.array((y_train))
        with sess.as_default():  
            
            model = MultiHeadAttention(512,8)
            model = MultiHeadAttention(512,8)

            #Horizontal Concatenation 沿著水平方向將數組堆疊起來
#             a = np.hstack((x,x_train)) 
#             b = np.hstack((y,y_train))   
            #vertical concatenation 沿著垂直方向堆疊
            a = np.vstack((x,x_train))
            b = np.vstack((y,y_train))
                       
            model = TextCNN(
                sequence_length=a.shape[1],
                num_classes=b.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)
            
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            grads_and_vars = optimizer.compute_gradients(model.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)                
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))
            
            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", model.loss)
            acc_summary = tf.summary.scalar("accuracy", model.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            result_loss.append(loss_summary)
            result_acc.append(acc_summary)
            
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())
           

            def train_step(x_batch, y_batch):
                
                #A single training step
                
                feed_dict = {
                  model.input_x: x_batch,
                  model.input_y: y_batch,
                  model.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, model.loss, model.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                
                #Evaluates model on a dev set
                #batches = data_helpers.batch_iter(
                #zip(x_batch, y_batch), 2, 1)
                #for batch in batches:
                x_batch, y_batch = zip(*batch)
                feed_dict = {
                  model.input_x: x_batch,
                  model.input_y: y_batch,
                  model.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy, num_correct = sess.run(
                    [global_step, dev_summary_op, model.loss, model.accuracy, model.num_correct],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)
                    
                return num_correct
            
            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.compat.v1.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
                    
            #if you need caculate cv test value/predict x_test 
            y_true = np.argmax(y_test,1)
            y_pred = sess.run([model.predictions], feed_dict={model.input_x:x_test, model.input_y:y_test, model.dropout_keep_prob: 1.0})
            y_pred = np.array(y_pred)
            y_pred = y_pred.ravel()
            acc= accuracy_score(y_true, y_pred)
            rs= recall_score(y_true, y_pred, average='weighted')
            f1= f1_score(y_true, y_pred, average='weighted')
            
            print("\nAccuracy on test set {:g}".format(acc))
            print("Recall rate on test set {:g}".format(rs))
            print("F1 on test set {:g}".format(f1))            
            

def main(argv=None):
    x_train, y_train, vocab_processor, x_dev, y_dev, x_test, y_test = preprocess() #, x_test, y_test
    train(x_train, y_train, vocab_processor, x_dev, y_dev, x_test, y_test)

if __name__ == '__main__':
    tf.compat.v1.app.run()
