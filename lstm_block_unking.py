from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split

from tensorflow.python.client import device_lib

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", "/home/jack/ZHIYI_v2/unking",
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", "models/",
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_integer("num_gpus", 1,
                     "If larger than 1, Grappler AutoParallel optimizer "
                     "will create multiple training replicas with each GPU "
                     "running one replica.")
flags.DEFINE_string("rnn_mode", None,
                    "The low level implementation of lstm cell: one of CUDNN, "
                    "BASIC, and BLOCK, representing cudnn_lstm, basic_lstm, "
                    "and lstm_block_cell classes.")
FLAGS = flags.FLAGS
BASIC = "basic"
CUDNN = "cudnn"
BLOCK = "block"


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


class PTB(object):

  def __init__(self, config, is_training, name=None):
    self.config = config
    self._name = name
    self._is_training = is_training
    self.batch_size = batch_size = self.config.batch_size
    self.num_steps = 329
    self.input_data = tf.placeholder(tf.float32, shape=(None, 329))
    self.targets= tf.placeholder(tf.int32, shape=(None, 1))
    self._rnn_params = None
    self._cell = None
    self.size = config.hidden_size
    inputs = self.input_data
    inputs = tf.reshape(inputs, [self.batch_size * self.num_steps, 1])

    softmax_w1 = tf.get_variable("softmax_w1", [1, self.size], dtype=data_type())
    softmax_b1 = tf.get_variable("softmax_b1", [self.size], dtype=data_type())
    inputs = tf.nn.xw_plus_b(inputs, softmax_w1, softmax_b1)
    inputs = tf.reshape(inputs, [self.batch_size, self.num_steps, self.size])

    if self._is_training and self.config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, self.config.keep_prob)

    output, state = self._build_rnn_graph(inputs, self.config, self._is_training)

    softmax_w = tf.get_variable(
      "softmax_w", [self.size, 34], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [34], dtype=data_type())
    logitt = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
    logit = tf.nn.softmax(logitt)
    y_pre = onehot(self.targets, batch_size)
    correct_prediction = tf.equal(tf.argmax(logit, 1), tf.argmax(y_pre, 1))
    self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))
    logits = tf.reshape(logitt, [self.batch_size, 1, 34])
    loss = tf.contrib.seq2seq.sequence_loss(
      logits,
      self.targets,
      tf.ones([self.batch_size, 1], dtype=data_type()),
      average_across_timesteps=False,
      average_across_batch=True)
    self._cost = tf.reduce_sum(loss)
    self._final_state = state

    if not self._is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                      self.config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
      zip(grads, tvars),
      global_step=tf.train.get_or_create_global_step())

    self._new_lr = tf.placeholder(
      tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def _build_rnn_graph(self, inputs, config, is_training):
    if config.rnn_mode == CUDNN:
      return self._build_rnn_graph_cudnn(inputs, config, is_training)
    else:
      return self._build_rnn_graph_lstm(inputs, config, is_training)

  def _build_rnn_graph_cudnn(self, inputs, config, is_training):
    inputs = tf.transpose(inputs, [1, 0, 2])
    self._cell = tf.contrib.cudnn_rnn.CudnnLSTM(
      num_layers=config.num_layers,
      num_units=config.hidden_size,
      input_size=config.hidden_size,
      dropout=1 - config.keep_prob if is_training else 0)
    params_size_t = self._cell.params_size()
    self._rnn_params = tf.get_variable(
      "lstm_params",
      initializer=tf.random_uniform(
        [params_size_t], -config.init_scale, config.init_scale),
      validate_shape=False)
    c = tf.zeros([config.num_layers, self.batch_size, config.hidden_size],
                 tf.float32)
    h = tf.zeros([config.num_layers, self.batch_size, config.hidden_size],
                 tf.float32)
    self._initial_state = (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)
    outputs, h, c = self._cell(inputs, h, c, self._rnn_params, is_training)
    outputs = tf.transpose(outputs, [1, 0, 2])
    outputs = tf.reshape(outputs, [-1, config.hidden_size])
    return outputs, (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)

  def _get_lstm_cell(self, config, is_training):
    if config.rnn_mode == BASIC:
      return tf.contrib.rnn.BasicLSTMCell(
        config.hidden_size, forget_bias=0.0, state_is_tuple=True,
        reuse=not is_training)
    if config.rnn_mode == BLOCK:
      return tf.contrib.rnn.LSTMBlockCell(
        config.hidden_size, forget_bias=0.0)
    raise ValueError("rnn_mode %s not supported" % config.rnn_mode)

  def _build_rnn_graph_lstm(self, inputs, config, is_training):
    def make_cell():
      cell = self._get_lstm_cell(config, is_training)
      if is_training and config.keep_prob < 1:
        cell = tf.contrib.rnn.DropoutWrapper(
          cell, output_keep_prob=config.keep_prob)
      return cell

    cell = tf.contrib.rnn.MultiRNNCell(
      [make_cell() for _ in range(config.num_layers)], state_is_tuple=True)

    self._initial_state = cell.zero_state(config.batch_size, data_type())
    state = self._initial_state
    outputs = []
    with tf.variable_scope("RNN"):
      for time_step in range(self.num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)
    output = outputs[-1]
    output = tf.reshape(output, [-1, config.hidden_size])
    return output, state

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})


  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
      return self._cost

  @property
  def x_data(self):
    return self.input_data

  @property
  def y_data(self):
      return self.targets

  @property
  def final_state(self):
    return self._final_state

  @property
  def accuracy(self):
      return self._accuracy

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 500
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  feature_size = 26
  rnn_mode = BLOCK


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  feature_size = 26
  rnn_mode = BLOCK


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  feature_size = 26
  rnn_mode = BLOCK


class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  feature_size = 26
  rnn_mode = BLOCK


def input_targets_producer(x_data, y_data, batch_size):
    data_len = len(y_data)

    epoch_size = data_len // batch_size

    data_x = np.reshape(x_data[0: batch_size * epoch_size, :],
                        [batch_size, epoch_size, -1])
    data_y = np.reshape(y_data[0: batch_size * epoch_size], [batch_size, epoch_size])
    for i in range(epoch_size):
        x = data_x[:,i,:]
        x = np.reshape(x,[batch_size,329])
        y = data_y[:,i]
        y = np.reshape(y,[batch_size, 1])
        yield x,y

def run_epoch(session, model, xdata, ydata, epoch_size, eval_op=None, verbose=False):
  start_time = time.time()
  costs = 0.0
  iters = 0
  accuracys = 0.0
  state = session.run(model.initial_state)

  for step, (x, y) in enumerate(input_targets_producer(xdata, ydata, model.batch_size)):
    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
        "accuracy": model.accuracy
    }
    if eval_op is not None:
      fetches["eval_op"] = eval_op

    feed_dict = {}
    feed_dict[model.x_data] = x
    feed_dict[model.y_data] = y
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    state = vals["final_state"]
    accuracy = vals["accuracy"]

    costs += cost
    iters += model.num_steps
    accuracys += accuracy

    if verbose and step % (epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps, cost:%.3f, accuracy:%.3f" %
            (step * 1.0 / epoch_size, np.exp(costs / iters),
             iters * model.batch_size * max(1, FLAGS.num_gpus) /
             (time.time() - start_time), cost, accuracy))

  return np.exp(costs / iters), costs / epoch_size, accuracys / epoch_size


def onehot(targets, batch_size):
  targets = tf.reshape(targets, [batch_size])
  labels = targets
  batch_size = tf.size(labels)
  labels = tf.expand_dims(labels, 1)
  indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)  # 生成索引
  concated = tf.concat([indices, labels], 1)  # 作为拼接
  onehot_labels = tf.sparse_to_dense(concated, tf.stack([batch_size, 34]), 1.0, 0.0)
  return onehot_labels


def get_config():
  """Get model config."""
  config = None
  if FLAGS.model == "small":
    config = SmallConfig()
  elif FLAGS.model == "medium":
    config = MediumConfig()
  elif FLAGS.model == "large":
    config = LargeConfig()
  elif FLAGS.model == "test":
    config = TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)
  if FLAGS.rnn_mode:
    config.rnn_mode = FLAGS.rnn_mode
  if FLAGS.num_gpus != 1 or tf.__version__ < "1.3.0" :
    config.rnn_mode = BASIC
  return config

def file_name(file_dir):
  L = []
  for root, dirs, files in os.walk(file_dir):
    for file in files:
      L.append(os.path.join(root, file))
  return L


def run_epoch_summary(session, model, summary_op, xdata, ydata, eval_op=None):
  state = session.run(model.initial_state)
  summary = None
  for step, (x, y) in enumerate(input_targets_producer(xdata, ydata, model.batch_size)):
    fetches = {
      "summary_op": summary_op,
    }
    if eval_op is not None:
      fetches["eval_op"] = eval_op

    feed_dict = {}
    feed_dict[model.x_data] = x
    feed_dict[model.y_data] = y
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    vals = session.run(fetches, feed_dict)
    summary = vals["summary_op"]

  return summary


def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")
  gpus = [
      x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"
  ]
  if FLAGS.num_gpus > len(gpus):
    raise ValueError(
        "Your machine has only %d gpus "
        "which is less than the requested --num_gpus=%d."
        % (len(gpus), FLAGS.num_gpus))

  train_x1 = np.loadtxt('/home/jack/ZHIYI_v2/non_king_processed/X_train_non_king_1.txt')
  train_x2 = np.loadtxt('/home/jack/ZHIYI_v2/non_king_processed/X_train_non_king_2.txt')
  train_y1 = np.loadtxt('/home/jack/ZHIYI_v2/non_king_processed/Y_train_non_king_1.txt')
  train_y2 = np.loadtxt('/home/jack/ZHIYI_v2/non_king_processed/Y_train_non_king_2.txt')
  train_x = np.vstack((train_x1, train_x2))
  train_y = np.concatenate(((train_y1, train_y2)), axis=0)
  valid_x = np.loadtxt('/home/jack/ZHIYI_v2/non_king_processed/X_test_non_king_1.txt')
  test_x = np.loadtxt('/home/jack/ZHIYI_v2/non_king_processed/X_test_non_king_2.txt')
  valid_y = np.loadtxt('/home/jack/ZHIYI_v2/non_king_processed/Y_test_non_king_1.txt')
  test_y = np.loadtxt('/home/jack/ZHIYI_v2/non_king_processed/Y_test_non_king_2.txt')



  config = get_config()
  eval_config = get_config()
  eval_config.batch_size = 1

  train_epoch = len(train_y) // config.batch_size
  valid_epoch = len(valid_y)// config.batch_size
  test_epoch = len(test_y)// eval_config.batch_size

  '''
  train_x1 = tf.convert_to_tensor(train_x1, name="train_datax1", dtype=tf.float32)
  train_y1 = tf.convert_to_tensor(train_y1, name="train_datay1", dtype=tf.int32)
  valid_x1 = tf.convert_to_tensor(valid_x1, name="valid_datax1", dtype=tf.float32)
  valid_y1 = tf.convert_to_tensor(valid_y1, name="valid_datay1", dtype=tf.int32)
  test_x1 = tf.convert_to_tensor(test_x1, name="test_datax1", dtype=tf.float32)
  test_y1 = tf.convert_to_tensor(test_y1, name="test_datay1", dtype=tf.int32)
  '''
  with tf.Graph().as_default(), tf.Session() as session:

    initializer = tf.random_uniform_initializer(-config.init_scale,
                                               config.init_scale)
    with tf.name_scope("Train"):
      with tf.variable_scope("Model", reuse=tf.AUTO_REUSE, initializer=initializer):
        m = PTB(config=config, is_training=True, name="Train")
        tf.summary.scalar("Training Loss", m.cost)
        tf.summary.scalar("Learning Rate", m.lr)
        summary_op_m = tf.summary.merge_all()
    with tf.name_scope("Valid"):
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = PTB(config=config, is_training=False, name="Valid")
    with tf.name_scope("Test"):
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = PTB(config=eval_config, is_training=False, name="Test")


    summary_writer = tf.summary.FileWriter('./lstm_logs',session.graph)

    tf.initialize_all_variables().run()  # 对参数变量初始化

    for i in range(config.max_max_epoch):

      lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
      m.assign_lr(session, config.learning_rate * lr_decay)
      print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
      train_perplexity, cost, accuracy = run_epoch(session, m, train_x, train_y, train_epoch, eval_op=m.train_op,
                                         verbose=True)
      print("Epoch: %d Train Perplexity: %.3f Cost: %.3f Accuracy: %.3f" % (i + 1, train_perplexity, cost, accuracy))
      valid_perplexity, cost, accuracy = run_epoch(session, mvalid, valid_x, valid_y, valid_epoch)
      print("Epoch: %d Valid Perplexity: %.3f Cost: %.3f Accuracy: %.3f" % (i + 1, valid_perplexity, cost, accuracy))

      test_perplexity, cost, accuracy = run_epoch(session, mtest, test_x, test_y, test_epoch)
      print("Test Perplexity: %.3f accuracy: %.3f" % (test_perplexity, accuracy))
      saver = tf.train.Saver()
      saver.save(session, './model/model.ckpt', global_step=i)
      summary_str = run_epoch_summary(session, m, summary_op_m, train_x, train_y, eval_op=m.train_op)
      summary_writer.add_summary(summary_str, i)

if __name__ == "__main__":
    tf.app.run()