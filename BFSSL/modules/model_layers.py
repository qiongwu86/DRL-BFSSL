from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import standard_ops

class DecomposedDense(Layer):
  def __init__(self,
               units,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               theta_sup=None,
               theta_unsup=None,
               bias=None,
               l1_thres=None,
               **kwargs):
    if 'input_shape' not in kwargs and 'input_dim' in kwargs:
      kwargs['input_shape'] = (kwargs.pop('input_dim'),)

    super(DecomposedDense, self).__init__(
        activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
    self.units = int(units)
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    self.supports_masking = True
    self.input_spec = InputSpec(min_ndim=2)

    self.theta_sup = theta_sup
    self.theta_unsup = theta_unsup
    self.bias = bias
    self.l1_thres = l1_thres

  def build(self, input_shape):
    dtype = dtypes.as_dtype(self.dtype or K.floatx())
    if not (dtype.is_floating or dtype.is_complex):
      raise TypeError('Unable to build `Dense` layer with non-floating point '
                      'dtype %s' % (dtype,))
    input_shape = tensor_shape.TensorShape(input_shape)
    if tensor_shape.dimension_value(input_shape[-1]) is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    last_dim = tensor_shape.dimension_value(input_shape[-1])
    self.input_spec = InputSpec(min_ndim=2,
                                axes={-1: last_dim})
    if self.use_bias:
      pass
    else:
      self.bias = None
    self.built = True
  
  def call(self, inputs):
    if tf.keras.backend.learning_phase() is not None:
      theta_sup = self.theta_sup
      theta_unsup = self.theta_unsup 
    else: 

      theta_sup = self.theta_sup
      hard_threshold = tf.cast(tf.greater(tf.abs(self.theta_unsup), self.l1_thres), tf.float32)
      theta_unsup = tf.multiply(self.theta_unsup, hard_threshold)

    self.my_theta = 0.5 * theta_sup + 0.5 * theta_unsup

    rank = len(inputs.shape)
    if rank > 2:
      outputs = standard_ops.tensordot(inputs, self.my_theta, [[rank - 1], [0]])

      if not context.executing_eagerly():
        shape = inputs.shape.as_list()
        output_shape = shape[:-1] + [self.units]
        outputs.set_shape(output_shape)
    else:
      inputs = math_ops.cast(inputs, self._compute_dtype)
      if K.is_sparse(inputs):
        outputs = sparse_ops.sparse_tensor_dense_matmul(inputs, self.my_theta)
      else:
        outputs = gen_math_ops.mat_mul(inputs, self.my_theta)
    if self.use_bias:
      outputs = nn.bias_add(outputs, self.bias)
    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if tensor_shape.dimension_value(input_shape[-1]) is None:
      raise ValueError(
          'The innermost dimension of input_shape must be defined, but saw: %s'
          % input_shape)
    return input_shape[:-1].concatenate(self.units)

  def get_config(self):
    config = {
        'units': self.units,
        'activation': activations.serialize(self.activation),
        'use_bias': self.use_bias,
        'kernel_initializer': initializers.serialize(self.kernel_initializer),
        'bias_initializer': initializers.serialize(self.bias_initializer),
        'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint': constraints.serialize(self.kernel_constraint),
        'bias_constraint': constraints.serialize(self.bias_constraint)
    }


    base_config = super(Dense, self).get_config()


    return dict(list(base_config.items()) + list(config.items()))


class DecomposedConv(Layer):
  def __init__(self, 
               filters,
               kernel_size,
               rank=2,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               theta_sup=None,
               theta_unsup=None,
               bias=None,
               l1_thres=None,
               **kwargs):
    super(DecomposedConv, self).__init__(
        trainable=trainable,
        name=name,
        activity_regularizer=regularizers.get(activity_regularizer),
        **kwargs)
    self.rank = rank
    self.filters = filters
    self.kernel_size = conv_utils.normalize_tuple(
        kernel_size, rank, 'kernel_size')
    self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
    self.padding = conv_utils.normalize_padding(padding)
    if (self.padding == 'causal' and not isinstance(self,
                                                    (Conv1D, SeparableConv1D))):
      raise ValueError('Causal padding is only supported for `Conv1D`'
                       'and ``SeparableConv1D`.')
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.dilation_rate = conv_utils.normalize_tuple(
        dilation_rate, rank, 'dilation_rate')
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)
    self.input_spec = InputSpec(ndim=self.rank + 2)

    self.theta_sup = theta_sup
    self.theta_unsup = theta_unsup
    self.bias = bias
    self.l1_thres = l1_thres

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_channel = self._get_input_channel(input_shape)
    kernel_shape = self.kernel_size + (input_channel, self.filters)

    if self.use_bias:
      pass
    else:
      self.bias = None
    
    channel_axis = self._get_channel_axis()
    self.input_spec = InputSpec(ndim=self.rank + 2,
                                axes={channel_axis: input_channel})

    self._build_conv_op_input_shape = input_shape
    self._build_input_channel = input_channel
    self._padding_op = self._get_padding_op()
    self._conv_op_data_format = conv_utils.convert_data_format(
        self.data_format, self.rank + 2)
   
    self.built = True
  
  def call(self, inputs):
    if tf.keras.backend.learning_phase() is not None:
      theta_sup = self.theta_sup
      theta_unsup = self.theta_unsup 
    else: 

      theta_sup = self.theta_sup
      hard_threshold = tf.cast(tf.greater(tf.abs(self.theta_unsup), self.l1_thres), tf.float32)
      theta_unsup = tf.multiply(self.theta_unsup, hard_threshold)

    self.my_theta = 0.5 * theta_sup + 0.5 * theta_unsup

    self._convolution_op = nn_ops.Convolution(
        inputs.get_shape(),
        filter_shape=self.my_theta.shape,
        dilation_rate=self.dilation_rate,
        strides=self.strides,
        padding=self._padding_op,
        data_format=self._conv_op_data_format)

    if self.padding == 'causal' and self.__class__.__name__ == 'Conv1D':
      inputs = array_ops.pad(inputs, self._compute_causal_padding())
   
    outputs = self._convolution_op(inputs, self.my_theta)

    if self.use_bias:
      if self.data_format == 'channels_first':
        if self.rank == 1:
          bias = array_ops.reshape(self.bias, (1, self.filters, 1))
          outputs += bias
        else:
          outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
      else:
        outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_last':
      space = input_shape[1:-1]
      new_space = []
      for i in range(len(space)):
        new_dim = conv_utils.conv_output_length(
            space[i],
            self.kernel_size[i],
            padding=self.padding,
            stride=self.strides[i],
            dilation=self.dilation_rate[i])
        new_space.append(new_dim)
      return tensor_shape.TensorShape([input_shape[0]] + new_space +
                                      [self.filters])
    else:
      space = input_shape[2:]
      new_space = []
      for i in range(len(space)):
        new_dim = conv_utils.conv_output_length(
            space[i],
            self.kernel_size[i],
            padding=self.padding,
            stride=self.strides[i],
            dilation=self.dilation_rate[i])
        new_space.append(new_dim)
      return tensor_shape.TensorShape([input_shape[0], self.filters] +
                                      new_space)

  def get_config(self):
    config = {
        'filters': self.filters,
        'kernel_size': self.kernel_size,
        'strides': self.strides,
        'padding': self.padding,
        'data_format': self.data_format,
        'dilation_rate': self.dilation_rate,
        'activation': activations.serialize(self.activation),
        'use_bias': self.use_bias,
        'kernel_initializer': initializers.serialize(self.kernel_initializer),
        'bias_initializer': initializers.serialize(self.bias_initializer),
        'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint': constraints.serialize(self.kernel_constraint),
        'bias_constraint': constraints.serialize(self.bias_constraint)
    }
    base_config = super(Conv, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def _compute_causal_padding(self):
    """Calculates padding for 'causal' option for 1-d conv layers."""
    left_pad = self.dilation_rate[0] * (self.kernel_size[0] - 1)
    if self.data_format == 'channels_last':
      causal_padding = [[0, 0], [left_pad, 0], [0, 0]]
    else:
      causal_padding = [[0, 0], [0, 0], [left_pad, 0]]
    return causal_padding

  def _get_channel_axis(self):
    if self.data_format == 'channels_first':
      return 1
    else:
      return -1

  def _get_input_channel(self, input_shape):
    channel_axis = self._get_channel_axis()
    if input_shape.dims[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    return int(input_shape[channel_axis])

  def _get_padding_op(self):
    if self.padding == 'causal':
      op_padding = 'valid'
    else:
      op_padding = self.padding
    if not isinstance(op_padding, (list, tuple)):
      op_padding = op_padding.upper()
    return op_padding

  def _recreate_conv_op(self, inputs):

    call_input_shape = inputs.get_shape()
    for axis in range(1, len(call_input_shape)):
      if (call_input_shape[axis] is not None
          and self._build_conv_op_input_shape[axis] is not None
          and call_input_shape[axis] != self._build_conv_op_input_shape[axis]):
        return True
    return False



