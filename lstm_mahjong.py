from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import time

import numpy as np
import tensorflow as tf
# from sklearn.model_selection import train_test_split
from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split

from tensorflow.models.tutorials.rnn.ptb import util

from tensorflow.python.client import device_lib

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", "/home/jack/ZHIYI_v2/king",
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


class PTBInput(object):
  """The input data."""

  def __init__(self, config, x_data, y_data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.input_data, self.targets, self.num_steps, self.epoch_size= self.input_targets_producer(x_data, y_data, batch_size, name=name)


  def input_targets_producer(self, x_data, y_data, batch_size, name):
    raw_data = tf.convert_to_tensor(x_data, name="raw_data", dtype=tf.float32)
    target_data = tf.convert_to_tensor(y_data, name="target_data", dtype=tf.int32)

    data_len = len(y_data)
    data_width = len(x_data[0, :])

    epoch_size = data_len // batch_size

    a = epoch_size
    data_x = tf.reshape(raw_data[0: batch_size * epoch_size, :],
                        [batch_size, epoch_size, -1])
    data_y = tf.reshape(target_data[0: batch_size * epoch_size], [batch_size, epoch_size])
    assertion = tf.assert_positive(
      epoch_size,
      message="epoch_size == 0, decrease batch_size")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.strided_slice(data_x, [0, i, 0],
                         [batch_size, i + 1, data_width])
    x = tf.reshape(x, [batch_size, data_width])
    y = tf.strided_slice(data_y, [0, i],
                         [batch_size, i + 1])
    y = tf.reshape(y, [batch_size, 1])
    z=data_width

    return x,y,z,a



class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config, input_):
    self._is_training = is_training
    self._input = input_
    self._rnn_params = None
    self._cell = None
    self.batch_size = input_.batch_size
    self.num_steps = input_.num_steps
    size = config.hidden_size
    # feature_size = config.feature_size

    inputs = input_.input_data
    inputs = tf.reshape(inputs, [self.batch_size * self.num_steps, 1])

    softmax_w1 = tf.get_variable(
      "softmax_w1", [1, size], dtype=data_type())
    softmax_b1 = tf.get_variable("softmax_b1", [size], dtype=data_type())
    inputs = tf.nn.xw_plus_b(inputs, softmax_w1, softmax_b1)
    inputs = tf.reshape(inputs, [self.batch_size, self.num_steps, size])

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    output, state = self._build_rnn_graph(inputs, config, is_training)

    softmax_w = tf.get_variable(
        "softmax_w", [size, 34], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [34], dtype=data_type())
    logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
     # Reshape logits to be a 3-D tensor for sequence loss
    logits = tf.reshape(logits, [self.batch_size, 1, 34])

    # Use the contrib sequence loss and average over the batches
    loss = tf.contrib.seq2seq.sequence_loss(
        logits,
        input_.targets,
        tf.ones([self.batch_size, 1], dtype=data_type()),
        average_across_timesteps=False,
        average_across_batch=True)

    # Update the cost
    self._cost = tf.reduce_sum(loss)
    self._final_state = state

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                      config.max_grad_norm)
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
    """Build the inference graph using CUDNN cell."""
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
    """Build the inference graph using canonical LSTM cells."""
    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
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
    # Simplified version of tf.nn.static_rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use tf.nn.static_rnn() or tf.nn.static_state_saving_rnn().
    #
    # The alternative version of the code below is:
    #
    # inputs = tf.unstack(inputs, num=self.num_steps, axis=1)
    # outputs, state = tf.nn.static_rnn(cell, inputs,
    #                                   initial_state=self._initial_state)
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

  def export_ops(self, name):
    """Exports ops to collections."""
    self._name = name
    ops = {util.with_prefix(self._name, "cost"): self._cost}
    if self._is_training:
      ops.update(lr=self._lr, new_lr=self._new_lr, lr_update=self._lr_update)
      if self._rnn_params:
        ops.update(rnn_params=self._rnn_params)
    for name, op in ops.items():
      tf.add_to_collection(name, op)
    self._initial_state_name = util.with_prefix(self._name, "initial")
    self._final_state_name = util.with_prefix(self._name, "final")
    util.export_state_tuples(self._initial_state, self._initial_state_name)
    util.export_state_tuples(self._final_state, self._final_state_name)

  def import_ops(self):
    """Imports ops from collections."""
    if self._is_training:
      self._train_op = tf.get_collection_ref("train_op")[0]
      self._lr = tf.get_collection_ref("lr")[0]
      self._new_lr = tf.get_collection_ref("new_lr")[0]
      self._lr_update = tf.get_collection_ref("lr_update")[0]
      rnn_params = tf.get_collection_ref("rnn_params")
      if self._cell and rnn_params:
        params_saveable = tf.contrib.cudnn_rnn.RNNParamsSaveable(
            self._cell,
            self._cell.params_to_canonical,
            self._cell.canonical_to_params,
            rnn_params,
            base_variable_scope="Model/RNN")
        tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, params_saveable)
    self._cost = tf.get_collection_ref(util.with_prefix(self._name, "cost"))[0]
    num_replicas = FLAGS.num_gpus if self._name == "Train" else 1
    self._initial_state = util.import_state_tuples(
        self._initial_state, self._initial_state_name, num_replicas)
    self._final_state = util.import_state_tuples(
        self._final_state, self._final_state_name, num_replicas)

  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

  @property
  def initial_state_name(self):
    return self._initial_state_name

  @property
  def final_state_name(self):
    return self._final_state_name


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
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


def run_epoch(session, model, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)

  fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model.input.epoch_size):
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    state = vals["final_state"]

    costs += cost
    iters += model.input.num_steps

    if verbose and step % (model.input.epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps, cost:%.3f" %
            (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
             iters * model.input.batch_size * max(1, FLAGS.num_gpus) /
             (time.time() - start_time), cost))

  return np.exp(costs / iters), costs / model.input.epoch_size


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

  i = 0
  data = []
  L = file_name(FLAGS.data_path)
  for dir in L:
    if i == 0:
      data = np.loadtxt(dir)
    else:
      tmp = np.loadtxt(dir)
      data = np.vstack((data, tmp))
    i += 1

  random.shuffle(data)
  train_x, test_x, train_y, test_y = train_test_split(data[:, :-1],
                                                      data[:, -1],
                                                      test_size=0.2,
                                                      random_state=0)

  train_x, valid_x, train_y, valid_y = train_test_split(train_x[:, :],
                                                        train_y[:],
                                                        test_size=0.2,
                                                        random_state=42)

  config = get_config()
  eval_config = get_config()
  eval_config.batch_size = 1

  # with tf.Graph().as_default():

  initializer = tf.random_uniform_initializer(-config.init_scale,
                                              config.init_scale)
  with tf.name_scope("Train"):
    train_input = PTBInput(config=config, x_data=train_x, y_data=train_y, name="TrainInput")
    with tf.variable_scope("Model", reuse=None, initializer=initializer):
      m = PTBModel(is_training=True, config=config, input_=train_input)
    tf.summary.scalar("Training Loss", m.cost)
    tf.summary.scalar("Learning Rate", m.lr)

  with tf.name_scope("Valid"):
    valid_input = PTBInput(config=config, x_data=valid_x, y_data=valid_y, name="ValidInput")
    with tf.variable_scope("Model", reuse=True, initializer=initializer):
      mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
    tf.summary.scalar("Validation Loss", mvalid.cost)

  with tf.name_scope("Test"):
    test_input = PTBInput(config=eval_config, x_data= test_x, y_data=test_y, name="TestInput")
    with tf.variable_scope("Model", reuse=True, initializer=initializer):
      mtest = PTBModel(is_training=False, config=eval_config, input_=test_input)

  # models = {"Train": m, "Valid": mvalid, "Test": mtest}
  # for name, model in models.items():
  #   model.export_ops(name)
  # metagraph = tf.train.export_meta_graph()
  # if tf.__version__ < "1.1.0" and FLAGS.num_gpus > 1:
  #   raise ValueError("num_gpus > 1 is not supported for TensorFlow versions "
  #                    "below 1.1.0")
  soft_placement = False
#   if FLAGS.num_gpus > 1:
#     soft_placement = True
#     util.auto_parallel(metagraph, m)
#
# with tf.Graph().as_default():
#   tf.train.import_meta_graph(metagraph)
#   # tf.train.import_meta_graph("models/model.ckpt-0.meta")
#   for model in models.values():
#     model.import_ops()
  sv = tf.train.Supervisor(logdir=FLAGS.save_path)
#   sv = tf.train.Supervisor()
  config_proto = tf.ConfigProto(allow_soft_placement=soft_placement)
  with sv.managed_session(config=config_proto) as session:
    for i in range(config.max_max_epoch):
      lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
      m.assign_lr(session, config.learning_rate * lr_decay)
      print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
      train_perplexity, cost = run_epoch(session, m, eval_op=m.train_op,
                                   verbose=True)
      print("Epoch: %d Train Perplexity: %.3f Cost: %.3f" % (i + 1, train_perplexity, cost))
      valid_perplexity, cost = run_epoch(session, mvalid)
      print("Epoch: %d Valid Perplexity: %.3f Cost: %.3f" % (i + 1, valid_perplexity, cost))

    test_perplexity, cost = run_epoch(session, mtest)
    print("Test Perplexity: %.3f accuracy: %.3f" % (test_perplexity, 100-cost))

    if FLAGS.save_path:
      print("Saving model to %s." % FLAGS.save_path)
      sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)


if __name__ == "__main__":
    tf.app.run()