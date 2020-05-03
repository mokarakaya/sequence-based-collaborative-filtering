
from tensorflow.python.eager import context
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell, LSTMStateTuple, _WEIGHTS_VARIABLE_NAME, \
    _BIAS_VARIABLE_NAME, LSTMCell
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest


class DualLSTMCell(LayerRNNCell):

  def __init__(self, num_units,
               cell_clip=None,
               initializer=None, num_proj=None, proj_clip=None,
               num_unit_shards=None, num_proj_shards=None,
               forget_bias=1.0, state_is_tuple=True,
               activation=None, reuse=None, name=None, dtype=None, **kwargs):

    super(DualLSTMCell, self).__init__(
        _reuse=reuse, name=name, dtype=dtype, **kwargs)
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
    if num_unit_shards is not None or num_proj_shards is not None:
      logging.warn(
          "%s: The num_unit_shards and proj_unit_shards parameters are "
          "deprecated and will be removed in Jan 2017.  "
          "Use a variable scope with a partitioner instead.", self)
    if context.executing_eagerly() and context.num_gpus() > 0:
      logging.warn("%s: Note that this cell is not optimized for performance. "
                   "Please use tf.contrib.cudnn_rnn.CudnnLSTM for better "
                   "performance on GPU.", self)

    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)

    self._num_units = num_units
    self._cell_clip = cell_clip
    self._initializer = initializers.get(initializer)
    self._num_proj = num_proj
    self._proj_clip = proj_clip
    self._num_unit_shards = num_unit_shards
    self._num_proj_shards = num_proj_shards
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    if activation:
      self._activation = activations.get(activation)
    else:
      self._activation = math_ops.tanh

    if num_proj:
      self._state_size = (
          LSTMStateTuple(num_units, num_proj)
          if state_is_tuple else num_units + num_proj)
      self._output_size = num_proj
    else:
      self._state_size = (
          LSTMStateTuple(2 * num_units, num_units)
          if state_is_tuple else 2 * num_units)
      self._output_size = num_units

    self.num_gates = 8
  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  @tf_utils.shape_type_conversion
  def build(self, inputs_shape):
    if inputs_shape[-1] is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % inputs_shape)

    input_depth = inputs_shape[-1]
    h_depth = self._num_units if self._num_proj is None else self._num_proj
    maybe_partitioner = (
        partitioned_variables.fixed_size_partitioner(self._num_unit_shards)
        if self._num_unit_shards is not None
        else None)
    self._kernel = self.add_variable(
        _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth + h_depth, self.num_gates * self._num_units],
        initializer=self._initializer,
        partitioner=maybe_partitioner)
    if self.dtype is None:
      initializer = init_ops.zeros_initializer
    else:
      initializer = init_ops.zeros_initializer(dtype=self.dtype)
    self._bias = self.add_variable(
        _BIAS_VARIABLE_NAME,
        shape=[self.num_gates * self._num_units],
        initializer=initializer)

    if self._num_proj is not None:
      maybe_proj_partitioner = (
          partitioned_variables.fixed_size_partitioner(self._num_proj_shards)
          if self._num_proj_shards is not None
          else None)
      self._proj_kernel = self.add_variable(
          "projection/%s" % _WEIGHTS_VARIABLE_NAME,
          shape=[self._num_units, self._num_proj],
          initializer=self._initializer,
          partitioner=maybe_proj_partitioner)

    self.built = True

  def call(self, inputs, state):

    num_proj = self._num_units if self._num_proj is None else self._num_proj
    sigmoid = math_ops.sigmoid

    if self._state_is_tuple:
      (c_d_prev, m_prev) = state
      c_prev = array_ops.slice(c_d_prev, [0, 0], [-1, self._num_units])
      d_prev = array_ops.slice(c_d_prev, [0, self._num_units], [-1, num_proj])
    else:
      c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
      m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])
      d_prev = []

    input_size = inputs.get_shape().with_rank(2)[1]
    if input_size.value is None:
      raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    lstm_matrix = math_ops.matmul(
        array_ops.concat([inputs, m_prev], 1), self._kernel)
    lstm_matrix = nn_ops.bias_add(lstm_matrix, self._bias)

    i, j, f, o, d_i, d_j, d_f, d_o = array_ops.split(
        value=lstm_matrix, num_or_size_splits=self.num_gates, axis=1)
    # Diagonal connections

    c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) * self._activation(j))
    # d_mem = (sigmoid(d + self._forget_bias) * d_prev + sigmoid(c - c_prev) * self._activation(j))
    d_mem = (sigmoid(d_f + self._forget_bias) * d_prev + sigmoid(d_i) * self._activation(d_j))
    # d_mem = ((sigmoid(d) - c_prev) * d_prev)
    if self._cell_clip is not None:
      # pylint: disable=invalid-unary-operand-type
      c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
      # pylint: enable=invalid-unary-operand-type

    # m = sigmoid(o) * self._activation(d_mem)
    m = sigmoid(o) * self._activation(c) * self._activation(d_mem) * sigmoid(d_o)

    if self._num_proj is not None:
      m = math_ops.matmul(m, self._proj_kernel)

      if self._proj_clip is not None:
        # pylint: disable=invalid-unary-operand-type
        m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)
        # pylint: enable=invalid-unary-operand-type

    c = array_ops.concat([c, d_mem], 1)
    new_state = (LSTMStateTuple(c, m) if self._state_is_tuple else
                 array_ops.concat([c, m], 1))
    return m, new_state

  def get_config(self):
    config = {
        "num_units": self._num_units,
        "cell_clip": self._cell_clip,
        "initializer": initializers.serialize(self._initializer),
        "num_proj": self._num_proj,
        "proj_clip": self._proj_clip,
        "num_unit_shards": self._num_unit_shards,
        "num_proj_shards": self._num_proj_shards,
        "forget_bias": self._forget_bias,
        "state_is_tuple": self._state_is_tuple,
        "activation": activations.serialize(self._activation),
        "reuse": self._reuse,
    }
    base_config = super(DualLSTMCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


def _enumerated_map_structure_up_to(shallow_structure, map_fn, *args, **kwargs):
  ix = [0]
  def enumerated_fn(*inner_args, **inner_kwargs):
    r = map_fn(ix[0], *inner_args, **inner_kwargs)
    ix[0] += 1
    return r
  return nest.map_structure_up_to(shallow_structure,
                                  enumerated_fn, *args, **kwargs)


def _default_dropout_state_filter_visitor(substate):
  if isinstance(substate, LSTMStateTuple):
    # Do not perform dropout on the memory state.
    return LSTMStateTuple(c=False, h=True)
  elif isinstance(substate, tensor_array_ops.TensorArray):
    return False
  return True


