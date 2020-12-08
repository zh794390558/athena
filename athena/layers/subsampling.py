import tensorflow as tf
from tensorflow.python.keras.utils import conv_utils

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
    layers = tf.keras.layers
    assert type(layer) in [layers.Conv1D, layers.MaxPool1D]
    seq_lens = [_update_1d(seq_len, layer) for seq_len in seq_lens]
    seq_lens = tf.stack(seq_lens, axis=0)
    return seq_lens


def _update_1d(seq_len, layer):
    layers = tf.keras.layers
    if type(layer) == layers.MaxPool1D: 
        return conv_utils.conv_output_length(seq_len, layer.pool_size[0], layer.padding, layer.strides[0])
    else:
        return conv_utils.conv_output_length(seq_len, layer.pool_size[0], layer.padding, layer.strides[0])


class ConcatSubsampler(tf.keras.layers.Layer):
    """Subsample by concatenating successive input frames."""
    def __init__(self, subsampling_factor, n_units):
        super().__init__()
        layers = tf.keras.layers

        self.factor = subsampling_factor
        if self.factor > 1:
            # input_dim : n_units * self.factor
            self.proj = layers.Dense(n_units)

    def call(self, xs, xlens, batch_first=True):
        '''
        Args:
            xs (FloatTensor): `[B, T, F]` or `[T, B, F]`
            xlens (IntTensor): `[B]` (on CPU)
            batch_first (bool): operate batch-first tensor
        Returns:
            xs (FloatTensor): `[B, T', F']` or `[T', B, F']`
            xlens (IntTensor): `[B]` (on CPU)
        '''
        if self.factor == 1:
            return xs, xlens

        if batch_first:
            xs = tf.transpose(xs, perm=(1, 0, 2)) 

        xs = [tf.concat( [ xs[t - r:t - r + 1] for r in tf.range(self.factor -1, -1, -1)], axis=-1) 
              for t in tf.range(tf.shape(xs)[0]) if (t + 1) % self.factor == 0]
        xs = tf.concat(xs, axis=0)
        # NOTE: Exclude the last frames if the length is not divisible
        xs = tf.nn.relu(self.proj(xs))


        if batch_first:
            xs = tf.transpose(xs, perm=(1, 0, 2))

        xlens = [tf.math.maximum(1, i // self.factor) for i in xlens ]
        xlens = tf.stack(xlens, axis=0)
        xlens = tf.cast(xlens, tf.int32)
        return xs, xlens


class DropSubsampler(tf.keras.layers.Layer):
    """Subsample by dropping input frames."""
    def __init__(self, subsampling_factor):
        super().__init__()
        self.factor = subsampling_factor

    def call(self, xs, xlens, batch_first=True):
        '''
        Args:
            xs (FloatTensor): `[B, T, F]` or `[T, B, F]`
            xlens (IntTensor): `[B]` (on CPU)
            batch_first (bool): operate batch-first tensor
        Returns:
            xs (FloatTensor): `[B, T', F']` or `[T', B, F']`
            xlens (IntTensor): `[B]` (on CPU)
        '''
        if self.factor == 1:
            return xs, xlens

        if batch_first:
            xs = xs[:, ::self.factor]
        else:
            xs = xs[::self.factor]

        xlens = [tf.math.maximum(1, tf.math.ceil(i / self.factor)) for i in xlens ]
        xlens = tf.stack(xlens, axis=0)
        xlens = tf.cast(xlens, tf.int32)
        return xs, xlens


class AddSubsampler(tf.keras.layers.Layer):
    """Subsample by summing input frames."""
    def __init__(self, subsampling_factor):
        super().__init__()
        self.factor = subsampling_factor
        assert subsampling_factor <= 2

    def call(self, xs, xlens, batch_first=True):
        '''
        Args:
            xs (FloatTensor): `[B, T, F]` or `[T, B, F]`
            xlens (IntTensor): `[B]` (on CPU)
            batch_first (bool): operate batch-first tensor
        Returns:
            xs (FloatTensor): `[B, T', F']` or `[T', B, F']`
            xlens (IntTensor): `[B]` (on CPU)
        '''
        if self.factor == 1:
            return xs, xlens

        if batch_first:
            bs, xmax, idim = tf.shape(xs)
            xs_even = xs[:, ::self.factor]
            if xmax % 2 == 0:
                xs_odd = xs[:, 1::self.factor]
            else:
                xs_odd = tf.concat([xs, tf.zeros([bs, 1, idim])], axis=1)[:, 1::self.factor]
        else:
            xmax, bs, idim = tf.shape(xs)
            xs_even = xs[::self.factor]
            if xmax % 2 == 0:
                xs_odd = xs[1::self.factor]
            else:
                xs_odd = tf.concat([xs, tf.zeros([1, bs, idim])], axis=0)[1::self.factor]

        xs = xs_odd + xs_even

        xlens = [ tf.math.maximum(1, tf.math.ceil(i / self.factor)) for i in xlens ]
        xlens = tf.stack(xlens)
        xlens = tf.cast(xlens, tf.int32)
        return xs, xlens


class MaxpoolSubsampler(tf.keras.layers.Layer):
    """Subsample by max-pooling input frames."""
    def __init__(self, subsampling_factor):
        super().__init__()
        layers = tf.keras.layers

        self.factor = subsampling_factor
        if subsampling_factor > 1:
            self.pool = layers.MaxPool1D(pool_size=subsampling_factor,
                                         strides=subsampling_factor,
                                         padding='same')

    def call(self, xs, xlens, batch_first=True):
        '''
        Args:
            xs (FloatTensor): `[B, T, F]` or `[T, B, F]`
            xlens (IntTensor): `[B]` (on CPU)
            batch_first (bool): operate batch-first tensor
        Returns:
            xs (FloatTensor): `[B, T', F']` or `[T', B, F']`
            xlens (IntTensor): `[B]` (on CPU)
        '''
        if self.factor == 1:
            return xs, xlens

        if batch_first:
            xs = self.pool(xs)
        else:
            xs = tf.transpose(xs, perm=(1,0, 2))
            xs = self.pool(xs)
            xs = tf.transpose(xs, perm=(1,0, 2))
        
        xlens = update_lens_1d(xlens, self.pool)
        xlens = tf.cast(xlens, tf.int32)
        return xs, xlens


class Conv1dSubsampler(tf.keras.layers.Layer):
    """Subsample by 1d convolution and max-pooling."""
    def __init__(self, subsampling_factor, n_units, conv_kernel_size=5):
        super().__init__()
        layers = tf.keras.layers

        assert conv_kernel_size % 2 == 1, "Kernel size should be odd for 'same' conv."
        self.factor = subsampling_factor
        if subsampling_factor > 1:
            self.conv1d = layers.Conv1D(n_units, conv_kernel_size, strides=1, padding='same')
            self.pool = layers.MaxPool1D(pool_size=subsampling_factor,
                                         strides=subsampling_factor,
                                         padding='same')

    def call(self, xs, xlens, batch_first=True):
        '''
        Args:
            xs (FloatTensor): `[B, T, F]` or `[T, B, F]`
            xlens (IntTensor): `[B]` (on CPU)
            batch_first (bool): operate batch-first tensor
        Returns:
            xs (FloatTensor): `[B, T', F']` or `[T', B, F']`
            xlens (IntTensor): `[B]` (on CPU)
        '''
        if self.factor == 1:
            return xs, xlens

        if batch_first:
            xs = tf.nn.relu(self.conv1d(xs))
            xs = self.pool(xs)
        else:
            xs = tf.transpose(xs, perm=(1,0, 2))
            xs = tf.nn.relu(self.conv1d(xs))
            xs = self.pool(xs)
            xs = tf.transpose(xs, perm=(1,0, 2))
        
        xlens = update_lens_1d(xlens, self.pool)
        xlens = tf.cast(xlens, tf.int32)
        return xs, xlens

