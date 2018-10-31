from __future__ import print_function

import os
import sys
import numpy as np
import tflearn
import tensorflow as tf
import time
import datetime


import timeit

from tensorflow.contrib import learn
import data_helpers


#sys.stdout = open("sentiment_log_to_check_epoch.txt", "w")


simple_tokenize = tf.load_op_library('./simple_tokenize.so')


tf.app.flags.DEFINE_string("train_file", "sentiment/clean_en_20b.csv", "Data source for the train data.")
tf.app.flags.DEFINE_string("test_file", "sentiment/clean_en_l10.csv", "Data source for the test data.")
tf.app.flags.DEFINE_string("model_path", "models/sentiment_en", "Folder to save the model")
tf.app.flags.DEFINE_string("pretrained_embedding_file", "pretrained_embedding/glove.6B.100d.txt", "Data source for the test data.")

tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory.')

# Model Hyperparameters
tf.app.flags.DEFINE_string("filter_sizes", "1,2,3", "Comma-separated filter sizes (default: '1,2,3')")
tf.app.flags.DEFINE_integer("num_filters", 25, "Number of filters per filter size (default: 25)")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.app.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.app.flags.DEFINE_integer("num_epochs", 5, "Number of training epochs (default: 5)")
tf.app.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.app.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.app.flags.DEFINE_integer("num_checkpoints", 1, "Number of checkpoints to store (default: 5)")
tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.app.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

tf.app.flags.DEFINE_integer("max_doc_length", 50, "Version number (default: 128)")
FLAGS = tf.app.flags.FLAGS

def _build_tensor_vocabulary_processor(vocab_dict, sess):
    vocabulary = list(list(zip(*vocab_dict.items())))
    keys = list(vocabulary[0])
    vals = list(vocabulary[1])
    table = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(keys, vals), 0
    )
    sess.run(tf.tables_initializer())
    return table
   
def _transform(x,  table, max_doc_len):
    def _pad_unk(u):
        u = tf.expand_dims(u, 0)
        return tf.pad(u, [[0,0],[0,max_doc_len-tf.shape(u)[1]]], 'CONSTANT',constant_values='<UNK>')[0]

    def cond(i, v, r):
        return tf.less(i, tf.shape(v)[0])

    def body(i, v, r):
        u = simple_tokenize.simple_tokenize([tf.gather(v,i)])
        u = tf.cond(tf.shape(u)[0] >= max_doc_len, lambda: tf.slice(u, [0], [max_doc_len]), lambda: _pad_unk(u))

        u = table.lookup(u)
        r = r.write(i, u)
        return tf.add(i, 1), v, r

    r = tf.TensorArray(dtype=tf.int32, size=1, dynamic_size=True)
    chop_and_pad = tf.while_loop(cond, body, [0, x, r])
    _, _, ta = chop_and_pad
    return ta.stack()


def add_training_words(vocab, train_texts, sess):
  """
    vocab: list of word (string)
    train_texts: list of sentence
    this function tokenize each sentences into words, and then 
    add merge with the vocab word list
  """
  existing_words = set(vocab)
  new_words = set()
  phrases = set()
  for text in train_texts:
    ts = text.split(" ")
    for t in ts:
      phrases.add(t)

  batches = data_helpers.batch_iter(list(phrases), 1024, 1)
  
  for batch in batches:
    
    s=" ".join(batch)
    tokens = simple_tokenize.simple_tokenize([s])
    ts = sess.run(tokens)
    for t in ts:
      t = t.decode()
      if (len(t) <= 15) and (t not in existing_words) :
        new_words.add(t)
  vocab.extend(list(new_words))



def main(_):
  start_time = timeit.default_timer()
  x_train, y_train, x_test, y_test, labels = data_helpers.load_data_and_labels(FLAGS.train_file, FLAGS.test_file)
  vocab, embedding = data_helpers.load_pretrained_embedding(
    FLAGS.pretrained_embedding_file)
  sess = tf.InteractiveSession()
  add_training_words(vocab, x_train, sess)
  print("pretrained words: "+str(len(embedding))+"    uncovered training words: "+ str(len(vocab)-len(embedding)))
  embedding = np.concatenate((embedding, np.zeros([len(vocab)-len(embedding), len(embedding[0])], dtype=np.float32)), axis=0)
  embedding_size = len(embedding[0])
  vocab_size = len(embedding)
  vocab_processor = tflearn.data_utils.VocabularyProcessor(FLAGS.max_doc_length)
  vocab_processor.fit(vocab)
  table = _build_tensor_vocabulary_processor(vocab_processor.vocabulary_._mapping, sess)


  serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
  feature_configs = {
    "x": tf.FixedLenFeature([], dtype=tf.string)
  }
  tf_example = tf.parse_example(serialized_tf_example, feature_configs)
  x = tf.identity(tf_example["x"], name="x")

  sequence_length = FLAGS.max_doc_length
  num_classes=y_train.shape[1]
  filter_sizes=list(map(int, FLAGS.filter_sizes.split(",")))
  num_filters=FLAGS.num_filters
  l2_reg_lambda=FLAGS.l2_reg_lambda


  input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
  l2_loss = tf.constant(0.0)

  labels = tf.constant(labels)
  with tf.device('/cpu:0'), tf.name_scope("embedding"):
    W = tf.Variable(embedding, name="W")
    x_ = _transform(x, table, sequence_length)
    embedded_chars = tf.nn.embedding_lookup(W, x_)
    #embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
  # Create a convolution + maxpool layer for each filter size
  pooled_outputs = []
  for i, filter_size in enumerate(filter_sizes):
      with tf.name_scope("conv-maxpool-%s" % filter_size):
          # Convolution Layer
          filter_shape = [filter_size, embedding_size, num_filters]
          W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
          b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
          conv = tf.nn.conv1d(
              embedded_chars,
              W,
              1,
              "VALID",
              name="conv")
          # Apply nonlinearity
          h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
          # Maxpooling over the outputs
          pooled = tf.nn.pool(
              h,
              window_shape=[sequence_length - filter_size + 1],
              pooling_type="MAX",
              padding='VALID',
              name="pool")
          pooled_outputs.append(pooled)

  # Combine all the pooled features
  num_filters_total = num_filters * len(filter_sizes)
  h_pool = tf.concat(pooled_outputs, 2)
  h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

  # Add dropout
  with tf.name_scope("dropout"):
      h_drop = tf.nn.dropout(h_pool_flat, FLAGS.dropout_keep_prob)

  # Final (unnormalized) scores and predictions
  with tf.name_scope("output"):
      W = tf.get_variable(
          "W",
          shape=[num_filters_total, num_classes],
          initializer=tf.contrib.layers.xavier_initializer())
      b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
      l2_loss += tf.nn.l2_loss(W)
      l2_loss += tf.nn.l2_loss(b)
      dropped_scores = tf.nn.xw_plus_b(h_drop, W, b, name="dropped_scores")
      scores = tf.nn.xw_plus_b(h_pool_flat, W, b, name="scores")
      y = tf.argmax(scores, 1, name="predictions")
      sorted_scores, sorted_indices = tf.nn.top_k(scores, num_classes)
      sorted_labels = tf.gather(labels,sorted_indices)

  # Calculate mean cross-entropy loss
  with tf.name_scope("loss"):

      #losses =  tf.nn.sigmoid_cross_entropy_with_logits(logits=dropped_scores, labels=input_y)
      losses = tf.nn.softmax_cross_entropy_with_logits(logits=dropped_scores, labels=input_y)
      loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

  # Accuracy
  with tf.name_scope("accuracy"):
      correct_predictions = tf.equal(y, tf.argmax(input_y, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

  optimizer = tf.train.AdamOptimizer()
  global_step = tf.Variable(0, name="global_step", trainable=False)
  grads_and_vars = optimizer.compute_gradients(loss)
  train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
  timestamp = str(int(time.time()))
  out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
  print("Writing to {}\n".format(out_dir))

  # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
  checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
  checkpoint_prefix = os.path.join(checkpoint_dir, "model")
  if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
  saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
  vocab_processor.save(os.path.join(out_dir, "vocab"))
  sess.run(tf.global_variables_initializer())

  batches = data_helpers.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
  for batch in batches:
    x_batch, y_batch = zip(*batch)
    _, step, lo, ac = sess.run(
      [train_op, global_step, loss, accuracy],
      feed_dict={x:x_batch, input_y:y_batch})
    time_str = datetime.datetime.now().isoformat()
    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, lo, ac))
    current_step = tf.train.global_step(sess, global_step)

    if current_step % FLAGS.evaluate_every == 0:
        print("\nEvaluation:")

        test_batches = data_helpers.batch_iter(
          list(zip(x_test, y_test)), FLAGS.batch_size, 1)
        acc = 0
        los = 0
        for test_batch in test_batches:
          x_test_batch, y_test_batch = zip(*test_batch)
          _, lo, ac = sess.run(
              [global_step, loss, accuracy],
              feed_dict={x:x_test_batch, input_y:y_test_batch})

          acc = acc + ac * len(x_test_batch)
          los = los + lo 
        acc = acc/len(x_test)
        print("loss {:g}, acc {:g}".format(los, acc))
    if current_step % FLAGS.checkpoint_every == 0:
        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
        print("Saved model checkpoint to {}\n".format(path))

  export_path = os.path.join(
      tf.compat.as_bytes(FLAGS.model_path),
      tf.compat.as_bytes(str(FLAGS.model_version)))
  print('Exporting trained model to', export_path)
  builder = tf.saved_model.builder.SavedModelBuilder(export_path)

  # Build the signature_def_map.
  classification_inputs = tf.saved_model.utils.build_tensor_info(x)
  classification_outputs_classes = tf.saved_model.utils.build_tensor_info(sorted_labels)
  classification_outputs_scores = tf.saved_model.utils.build_tensor_info(sorted_scores)

  classification_signature = (
      tf.saved_model.signature_def_utils.build_signature_def(
          inputs={
              tf.saved_model.signature_constants.CLASSIFY_INPUTS:
                  classification_inputs
          },
          outputs={
              tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
                  classification_outputs_classes,
              tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
                  classification_outputs_scores
          },
          method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))


  tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
  tensor_info_y = tf.saved_model.utils.build_tensor_info(y)

  prediction_signature = (
      tf.saved_model.signature_def_utils.build_signature_def(
          inputs={'x': tensor_info_x},
          outputs={'y': tensor_info_y},
          method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

  builder.add_meta_graph_and_variables(
      sess, [tf.saved_model.tag_constants.SERVING],
      signature_def_map={
          'predict_images':
              prediction_signature,
          tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
              classification_signature,
      },
      main_op=tf.tables_initializer(),
      strip_default_attrs=True)

  builder.save()
  print('Done exporting!')
  elapsed = timeit.default_timer() - start_time
  print(elapsed)


if __name__ == '__main__':
  tf.app.run()





