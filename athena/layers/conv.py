import tensorflow as tf
import logging
import numpy as np
from initialization import (
    init_with_lecun_normal,
    init_with_uniform,
    init_with_xavier_uniform,
    init_like_transformer_xl,
)
from tensorflow.python.keras.utils import conv_utils

logger = logging.getLogger(__name__)
layers = tf.keras.layers

class ConvEncoder(tf.keras.layers.Layer):
    """CNN encoder.
    Args:
        input_dim (int): dimension of input features (freq * channel)
        in_channel (int): number of channels of input features
        channels (list): number of channles in CNN blocks
        kernel_sizes (list): size of kernels in CNN blocks
        strides (list): strides in CNN blocks
        poolings (list): size of poolings in CNN blocks
        dropout (float): probability to drop nodes in hidden-hidden connection
        batch_norm (bool): apply batch normalization
        layer_norm (bool): apply layer normalization
        residual (bool): apply residual connections
        bottleneck_dim (int): dimension of the bridge layer after the last layer
        param_init (float): mean of uniform distribution for parameter initialization
        layer_norm_eps (float): epsilon value for layer normalization
    """

    def __init__(self, input_dim, in_channel, channels,
                 kernel_sizes, strides, poolings,
                 dropout, batch_norm, layer_norm, residual,
                 bottleneck_dim, param_init, layer_norm_eps=1e-12):

        super(ConvEncoder, self).__init__()

        (channels, kernel_sizes, strides, poolings), is_1dconv = parse_cnn_config(
            channels, kernel_sizes, strides, poolings)

        self.is_1dconv = is_1dconv
        self.in_channel = in_channel
        assert input_dim % in_channel == 0 
        self.input_freq = input_dim // in_channel
        self.residual = residual

        assert len(channels) > 0
        assert len(channels) == len(kernel_sizes) == len(strides) == len(poolings)
         
        self.layers = []
        C_i = input_dim if is_1dconv else in_channel
        in_freq = self.input_freq
        for lth, c in enumerate(channels):
            if is_1dconv:
                block = Conv1dBlock(in_channel=C_i,
                                    out_channel=channels[lth],
                                    kernel_size=kernel_sizes[lth],  # T
                                    stride=strides[lth],  # T
                                    pooling=poolings[lth],  # T
                                    dropout=dropout,
                                    batch_norm=batch_norm,
                                    layer_norm=layer_norm,
                                    layer_norm_eps=layer_norm_eps,
                                    residual=residual)
            else:
                block = Conv2dBlock(in_freq,
                                    C_i,
                                    channels[lth],
                                    kernel_sizes[lth],
                                    strides[lth],
                                    poolings[lth],
                                    dropout,
                                    batch_norm,
                                    layer_norm,
                                    layer_norm_eps,
                                    residual)
            self.layers += [block]
            in_freq = block.output_dim
            C_i = c 

        self._odim = C_i if is_1dconv else int(C_i * in_freq)

        self.bridge = None
        if bottleneck_dim > 0 and bottleneck_dim != self._odim:
            self.bridge = layers.Dense(bottleneck_dim)
            self._odim = bottleneck_dim

        # calculate subsampling factor
        self._factor = 1
        if poolings:
            for p in poolings:
                self._factor *= p if is_1dconv else p[0]

        self.calculate_context_size(kernel_sizes, strides, poolings)

    @property
    def output_dim(self):
        return self._odim

    @property
    def subsampling_factor(self):
        return self._factor

    @property
    def context_size(self):
        return self._context_size


    def calculate_context_size(self, kernel_sizes, strides, poolings):
        self._context_size = 0
        context_size_bottom = 0
        factor = 1
        for lth, k in enumerate(kernel_sizes):
            kernel_size = kernel_sizes[lth] if self.is_1dconv else kernel_sizes[lth][0]
            pooling = poolings[lth] if self.is_1dconv else poolings[lth][0]

            lookahead = (kernel_size -1 ) // 2
            lookahead *= 2 
            #NOTE: each CNN block has 2 CNN layers

            if factor == 1:
                self._context_size += lookahead
                context_size_bottom = self._context_size
            else:
                self._context_size += context_size_bottom * lookahead
                context_size_bottom *= pooling
            factor *= pooling


    def reset_parameters(self, param_init):
        """Initialize parameters with lecun style."""
        logger.info('===== Initialize %s with lecun style =====' % self.__class__.__name__) 
        for var in self.trainable_weights:
            init_with_lecun_normal(var, param_init)


    def call(self, xs, xlens, lookback=False, lookahead=False):
        """
        Args:
            xs (FloatTensor): `[B, T, F]`
            xlens (IntTenfor): `[B]` (on CPU)
        Returns:
            xs (FloatTensor): `[B, T', F']`
            xlens (IntTenfor): `[B]` (on CPU)
        """ 
        B, T, F = tf.shape(xs)
        C_i = self.in_channel
        if not self.is_1dconv:
            xs = tf.reshape(xs, (B, T, F // C_i, C_i))

        for block in self.layers:
            xs, xlens = block(xs, xlens, lookback=lookback, lookahead=lookahead)

        if not self.is_1dconv:
            B, T, F, C_o = tf.shape(xs)
            xs = tf.reshape(xs, (B, T, -1))

        if self.bridge is not None:
            xs = self.bridge(xs)

        return xs, xlens


class Conv1dBlock(tf.keras.layers.Layer):
    ''' 1d-CNN block.'''
    def __inti__(self, in_channel, out_channel,
        kernel_size, stride, pooling,
        dropout, batch_norm, layer_norm, layer_norm_eps, residual):
        super().__init__()

        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.residual = residual
        self.dropout = layers.Dropout(dropout)
        
        # 1st layer
        self.conv1 = layers.Conv1D(out_channel, 
                                   kernel_size,
                                   strides=stride,
                                   padding='same')
        self._odim = out_channel 
        self.batch_norm1 = layers.BatchNormalization() if batch_norm else lambda x: x
        self.layer_norm1 = layers.LayerNormalization(epsilon=layer_norm_eps) if layer_norm else lambda x: x


        # 2st layer
        self.conv2 = layers.Conv1D(out_channel, 
                                   kernel_size,
                                   strides=stride,
                                   padding='same')
        self._odim = out_channel 
        self.batch_norm2 = layers.BatchNormalization() if batch_norm else lambda x: x
        self.layer_norm2 = layers.LayerNormalization(epsilon=layer_norm_eps) if layer_norm else lambda x: x

        # Max Pooling
        self.pool = None
        self._factor = 1
        if pooling > 1:
            self.pool = layers.MaxPool1D(pool_size=kernel_size, strides=pooling, padding='same')

            # calculate subsampling factor
            self._factor *= pooling

    @property
    def output_dim(self):
        return self._odim

    @property
    def subsampling_factor(self):
        return self._factor

    def call(self, xs, xlens, lookback=False, lookahead=False):
        """ 
        Args:
            xs (FloatTensor): `[B, T, F]`
            xlens (IntTensor): `[B]` (on CPU)
            lookback (bool): truncate the leftmost frames
                because of lookback frames for context
            lookahead (bool): truncate the rightmost frames
                because of lookahead frames for context
        Returns:
            xs (FloatTensor): `[B, T', F']`
            xlens (IntTensor): `[B]` (on CPU)
        """
        residual = xs
        
        xs = self.conv1(xs)
        xs = self.batch_norm1(xs)
        xs = self.layer_norm1(xs)
        xs = tf.nn.relu(xs)
        xs = self.dropout(xs)
        xlens = update_lens_1d(xlens, self.conv1)

        xs = self.conv2(xs)
        xs = self.batch_norm2(xs)
        xs = self.layer_norm2(xs)
        if self.residual and xs.shape == residual.shape:
            xs += residual # NOTE: this is the sample place as in ResNet
        xs = tf.nn.relu(xs)
        xs = self.dropout(xs)
        xlens = update_len_1d(xlens, self.conv2)

        if self.pool is not None:
            xs = self.pool(xs)
            xlens = update_lens_1d(xlens, self.pool)

        return xs, xlens


class Conv2dBlock(tf.keras.layers.Layer):
    ''' 2d-CNN block.'''
    def __init__(self, input_dim, in_channel, out_channel,
            kernel_size, stride, pooling,
            dropout, batch_norm, layer_norm, layer_norm_eps, residual):

        super().__init__()

        print(out_channel, kernel_size, stride, pooling)

        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.residual = residual
        self.dropout = layers.Dropout(dropout)
        
        # 1st layer
        self.conv1 = layers.Conv2D(out_channel, 
                                   tuple(kernel_size),
                                   strides=tuple(stride),
                                   padding='same')
        self._odim = update_lens_2d(tf.convert_to_tensor([input_dim]), self.conv1, dim=3)
        self.batch_norm1 = layers.BatchNormalization() if batch_norm else lambda x: x
        self.layer_norm1 = layers.LayerNormalization(epsilon=layer_norm_eps) if layer_norm else lambda x: x


        # 2st layer
        self.conv2 = layers.Conv2D(out_channel, 
                                   tuple(kernel_size),
                                   strides=tuple(stride),
                                   padding='same')
        self._odim = update_lens_2d(tf.convert_to_tensor([self._odim]), self.conv2, dim=3)
        self.batch_norm2 = layers.BatchNormalization() if batch_norm else lambda x: x
        self.layer_norm2 = layers.LayerNormalization(epsilon=layer_norm_eps) if layer_norm else lambda x: x

        # Max Pooling
        self.pool = None
        self._factor = 1
        if len(pooling) > 0 and np.prod(pooling) > 1:
            self.pool = layers.MaxPool2D(pool_size=tuple(kernel_size), strides=tuple(pooling), padding='same')
            self._odim = update_lens_2d(tf.convert_to_tensor([self._odim]), self.pool, dim=3)
            if self._odim % 2 != 0:
                self._odim = (self._odim // 2) * 2

            # calculate subsampling factor
            self._factor *= pooling[1]

    @property
    def output_dim(self):
        return self._odim

    @property
    def subsampling_factor(self):
        return self._factor

    def call(self, xs, xlens, lookback=False, lookahead=False):
        """ 
        Args:
            xs (FloatTensor): `[B, T, F, C_i]`
            xlens (IntTensor): `[B]` (on CPU)
            lookback (bool): truncate leftmost frames for lookback in CNN context
            lookahead (bool): truncate rightmost frames for lookahead in CNN context
        Returns:
            xs (FloatTensor): `[B, T', F', C_o]`
            xlens (IntTensor): `[B]` (on CPU)
        """
        residual = xs
        
        xs = self.conv1(xs)
        xs = self.batch_norm1(xs)
        xs = self.layer_norm1(xs)
        xs = tf.nn.relu(xs)
        xs = self.dropout(xs)
        xlens = update_lens_2d(xlens, self.conv1, dim=0)
        if lookback and tf.shape(xs)[1] > self.conv1.stride[0]:
            xmax = tf.shape(xs)[1]
            xs = xs[:, self.conv1.stride[0]:]
            xlens = xlens - (xmax - tf.shape(xs)[1])
        if lookahead and tf.shape(xs)[1] > self.conv1.stride[0]:
            xmax = tf.shape(xs)[1]
            xs = xs[:, : tf.shape(xs)[1] - self.conv1.stride[0]]
            xlens = xlens - (xmax - tf.shape(xs)[1]) 

        xs = self.conv2(xs)
        xs = self.batch_norm2(xs)
        xs = self.layer_norm2(xs)
        if self.residual and xs.shape == residual.shape:
            xs += residual # NOTE: this is the sample place as in ResNet
        xs = tf.nn.relu(xs)
        xs = self.dropout(xs)
        xlens = update_lens_2d(xlens, self.conv2, dim=0)
        if lookback and tf.shape(xs)[1] > self.conv2.stride[0]:
            xmax = tf.shape(xs)[1]
            xs = xs[:, self.conv2.stride[0]:]
            xlens = xlens - (xmax - tf.shape(xs)[1])
        if lookahead and tf.shape(xs)[1] > self.conv2.stride[0]:
            xmax = tf.shape(xs)[1]
            xs = xs[:, : tf.shape(xs)[1] - self.conv2.stride[0]]
            xlens = xlens - (xmax - tf.shape(xs)[1]) 

        if self.pool is not None:
            xs = self.pool(xs)
            xlens = update_lens_2d(xlens, self.pool, dim=0)

        return xs, xlens


def update_lens_1d(seq_lens, layer):
    """ update lengths (frequencey or time).
    Args:
        seq_lens (IntTensor): `[B]`
        layer (layers.Conv1d or layers.MaxPool1d):
    Returns:
        seq_lens (IntTensor): `[B]`
    """
    if seq_lens is None:
        return seq_lens
    assert isinstance(seq_lens, tf.Tensor)
    assert type(layer) in [layers.Conv1D, layers.MaxPool1D]
    seq_lens = [_update_1d(seq_len, layer) for seq_len in seq_lens]
    seq_lens = tf.stack(seq_lens, axis=0)
    return seq_lens


def _update_1d(seq_len, layer):
    if type(layer) == layers.MaxPool1D: 
        return conv_utils.conv_output_length(seq_len, layer.pool_size[0], layer.padding, layer.strides[0])
    else:
        return conv_utils.conv_output_length(seq_len, layer.kernel_size[0], layer.padding, layer.strides[0])


def update_lens_2d(seq_lens, layer, dim=0):
    """ update lengths (frequencey or time).
    Args:
        seq_lens (IntTensor): `[B]`
        layer (layers.Conv1d or layers.MaxPool1d):
    Returns:
        seq_lens (IntTensor): `[B]`
    """
    if seq_lens is None:
        return seq_lens
    assert isinstance(seq_lens, tf.Tensor)
    assert type(layer) in [layers.Conv2D, layers.MaxPool2D]
    seq_lens = [_update_2d(seq_len, layer) for seq_len in seq_lens]
    seq_lens = tf.stack(seq_lens, axis=0)
    return seq_lens


def _update_2d(seq_len, layer):
    if type(layer) == layers.MaxPool2D: 
        return conv_utils.conv_output_length(seq_len, layer.pool_size[0], layer.padding, layer.strides[0])
    else:
        return conv_utils.conv_output_length(seq_len, layer.kernel_size[0], layer.padding, layer.strides[0])


def parse_cnn_config(channels, kernel_sizes, strides, poolings):
    _channels, _kernel_sizes, _strides, _poolings = [], [], [], []
    is_1dconv = '(' not in kernel_sizes
    if len(channels) > 0:
        _channels = [int(c) for c in channels.split('_')]
    if len(kernel_sizes) > 0:
        if is_1dconv:
            _kernel_sizes = [int(c) for c in kernel_sizes.split('_')]
        else:
            _kernel_sizes = [[int(c.split(',')[0].replace('(', '')),
                              int(c.split(',')[1].replace(')', ''))] for c in kernel_sizes.split('_')]
    if len(strides) > 0:
        if is_1dconv:
            assert '(' not in _strides and ')' not in _strides
            _strides = [int(s) for s in strides.split('_')]
        else:
            _strides = [[int(s.split(',')[0].replace('(', '')),
                         int(s.split(',')[1].replace(')', ''))] for s in strides.split('_')]
    if len(poolings) > 0:
        if is_1dconv:
            assert '(' not in poolings and ')' not in poolings
            _poolings = [int(p) for p in poolings.split('_')]
        else:
            _poolings = [[int(p.split(',')[0].replace('(', '')),
                          int(p.split(',')[1].replace(')', ''))] for p in poolings.split('_')]
    return (_channels, _kernel_sizes, _strides, _poolings), is_1dconv
