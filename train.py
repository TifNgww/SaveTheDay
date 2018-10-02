#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import io
# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("train_file", "zendesk/cm_zendesk.train.csv", "Data source for the train data.")
tf.flags.DEFINE_string("test_file", "zendesk/cm_zendesk_en.test.csv", "Data source for the test data.")
tf.flags.DEFINE_string("pretrained_embedding_file", "pretrained_embedding/cca-59", "Data source for the test data.")

# Model Hyperparameters
tf.flags.DEFINE_string("filter_sizes", "1,2,3", "Comma-separated filter sizes (default: '1,2,3')")
tf.flags.DEFINE_integer("num_filters", 25, "Number of filters per filter size (default: 25)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 1024, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 15, "Number of training epochs (default: 5)")
tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

tf.flags.DEFINE_integer("model_version", 0, "Version number (default: 0)")
tf.flags.DEFINE_integer("max_doc_length", 128, "Version number (default: 128)")

FLAGS = tf.flags.FLAGS

def preprocess():
    # Data Preparation
    # ==================================================

    vocab,embedding = data_helpers.load_pretrained_embedding(FLAGS.pretrained_embedding_file)

    # Load data
    print("Loading data...")
    train_texts, train_labels, test_tests, test_labels  = data_helpers.load_data_and_labels(FLAGS.train_file, FLAGS.test_file)

    # Build vocabulary
    all_texts = train_texts + test_tests
    train_sz = len(train_texts)
    max_document_length = min(max([len(x.split(" ")) for x in all_texts]), FLAGS.max_doc_length)


    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    pretrain = vocab_processor.fit(vocab)
    x = np.array(list(vocab_processor.transform(all_texts)))

    x_train, x_test = x[:train_sz], x[train_sz:]

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(x_train)))
    x_train_shuffled = x_train[shuffle_indices]
    y_train_shuffled = train_labels[shuffle_indices]

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    return x_train_shuffled, y_train_shuffled, vocab_processor, x_test, test_labels, embedding

def train(x_train, y_train, vocab_processor, x_dev, y_dev, embedding):
    # Training
    # ==================================================

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                embedding=embedding,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            sess.run(cnn.embedding_init, feed_dict={cnn.embedding_placeholder: embedding})
            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, loss, accuracy = sess.run(
                    [train_op, global_step, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

            def dev_step(x_batch, y_batch):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, loss, accuracy = sess.run(
                    [global_step, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                return step, loss, accuracy

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")

                    dev_batches = data_helpers.batch_iter(
                      list(zip(x_dev, y_dev)), FLAGS.batch_size, 1)
                    acc = 0
                    los = 0
                    for dev_batch in dev_batches:
                      x_dev_batch, y_dev_batch = zip(*dev_batch)
                      _, lo, ac = dev_step(x_dev_batch, y_dev_batch)
                      acc = acc + ac * len(x_dev_batch)
                      los = los + lo * len(x_dev_batch)
                    acc = acc/len(x_dev)
                    los = los/len(x_dev)
                    print("loss {:g}, acc {:g}".format(los, acc))
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))



def evaluation(x_test, y_test, vocab_processor, x_dev, y_dev, embedding):
    
    print("evaluation_test")
    #checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))




def main(argv=None):
    x_train, y_train, vocab_processor, x_dev, y_dev, embedding = preprocess()
    train(x_train, y_train, vocab_processor, x_dev, y_dev, embedding)

if __name__ == '__main__':
    tf.app.run()
