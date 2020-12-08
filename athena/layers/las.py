# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Xiangang Li
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# changed from the pytorch transformer implementation
# pylint: disable=invalid-name, too-many-instance-attributes
# pylint: disable=too-few-public-methods, too-many-arguments

""" the transformer model """
import tensorflow as tf
import logging

import numpy as np
#from .conv_module import ConvModule
#from .commons import ACTIVATIONS
from initialization import (
    init_with_lecun_normal,
    init_with_uniform,
    init_with_xavier_uniform,
    init_like_transformer_xl,
)
from subsampling import (
        AddSubsampler,
        ConcatSubsampler,
        Conv1dSubsampler,
        DropSubsampler,
        MaxpoolSubsampler
)

logger = logging.getLogger(__name__)

class Transformer(tf.keras.layers.Layer):
    """A transformer model. User is able to modify the attributes as needed.

    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder/decoder intermediate layer, relu or gelu
            (default=relu).
        custom_encoder: custom encoder (default=None).
        custom_decoder: custom decoder (default=None).

    Examples:
        >>> transformer_model = Transformer(nhead=16, num_encoder_layers=12)
        >>> src = tf.random.normal((10, 32, 512))
        >>> tgt = tf.random.normal((20, 32, 512))
        >>> out = transformer_model(src, tgt)
    """

    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="gelu",
        unidirectional=False,
        look_ahead=0,
        custom_encoder=None,
        custom_decoder=None,
        conv_module_kernel_size=0
    ):
        super().__init__()
        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layers = [
                TransformerEncoderLayer(
                    d_model, nhead, dim_feedforward,
                    dropout, activation, unidirectional, look_ahead,
                    conv_module_kernel_size=conv_module_kernel_size
                )
                for _ in range(num_encoder_layers)
            ]
            self.encoder = TransformerEncoder(encoder_layers)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layers = [
                TransformerDecoderLayer(
                    d_model, nhead, dim_feedforward, dropout, activation
                )
                for _ in range(num_decoder_layers)
            ]
            self.decoder = TransformerDecoder(decoder_layers)

        self.d_model = d_model
        self.nhead = nhead

    def call(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None,
             return_encoder_output=False, return_attention_weights=False, training=None):
        """Take in and process masked source/target sequences.

        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
            src_mask: the additive mask for the src sequence (optional).
            tgt_mask: the additive mask for the tgt sequence (optional).
            memory_mask: the additive mask for the encoder output (optional).
            src_key_padding_mask: the ByteTensor mask for src keys per batch (optional).
            tgt_key_padding_mask: the ByteTensor mask for tgt keys per batch (optional).
            memory_key_padding_mask: the ByteTensor mask for memory keys per batch (optional).

        Shape:
            - src: :math:`(N, S, E)`.
            - tgt: :math:`(N, T, E)`.
            - src_mask: :math:`(N, S)`.
            - tgt_mask: :math:`(N, T)`.
            - memory_mask: :math:`(N, S)`.

            Note: [src/tgt/memory]_mask should be a ByteTensor where True values are positions
            that should be masked with float('-inf') and False values will be unchanged.
            This mask ensures that no information will be taken from position i if
            it is masked, and has a separate mask for each sequence in a batch.

            - output: :math:`(N, T, E)`.

            Note: Due to the multi-head attention architecture in the transformer model,
            the output sequence length of a transformer is same as the input sequence
            (i.e. target) length of the decode.

            where S is the source sequence length, T is the target sequence length, N is the
            batch size, E is the feature number

        Examples:
            >>> output = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        """

        if src.shape[0] != tgt.shape[0]:
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.shape[2] != self.d_model or tgt.shape[2] != self.d_model:
            raise RuntimeError(
                "the feature number of src and tgt must be equal to d_model"
            )

        memory = self.encoder(src, src_mask=src_mask, training=training)
        output = self.decoder(
            tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
            return_attention_weights=return_attention_weights, training=training
        )
        if return_encoder_output:
            return output, memory
        return output


layers = tf.keras.layers

class RNNEncoder(tf.keras.layers.Layer):
    """RNNEncoder is a stack of N encoder layers

    Args:
        input_dim (int): dimension of input features (freq * channel)
        enc_type (str): type of encoder (including pure CNN layers)
        n_units (int): number of units in each layer
        n_projs (int): number of units in each projection layer
        last_proj_dim (int): dimension of the last projection layer
        n_layers (int): number of layers
        dropout_in (float): dropout probability for input-hidden connection
        dropout (float): dropout probability for hidden-hidden connection
        subsample (list): subsample in the corresponding RNN layers
            ex.) [1, 2, 2, 1] means that subsample is conducted in the 2nd and 3rd layers.
        subsample_type (str): drop/concat/max_pool/1dconv
        n_stacks (int): number of frames to stack
        n_splices (int): number of frames to splice
        conv_in_channel (int): number of channels of input features
        conv_channels (int): number of channels in CNN blocks
        conv_kernel_sizes (list): size of kernels in CNN blocks
        conv_strides (list): number of strides in CNN blocks
        conv_poolings (list): size of poolings in CNN blocks
        conv_batch_norm (bool): apply batch normalization only in CNN blocks
        conv_layer_norm (bool): apply layer normalization only in CNN blocks
        conv_bottleneck_dim (int): dimension of bottleneck layer between CNN and RNN layers
        bidir_sum_fwd_bwd (bool): sum up forward and backward outputs for dimension reduction
        param_init (float): model initialization parameter
        chunk_size_left (int): left chunk size for latency-controlled bidirectional encoder
        chunk_size_right (int): right chunk size for latency-controlled bidirectional encoder
    """

    def __init__(self, input_dim, enc_type, n_units, n_projs, last_proj_dim, n_layers,
            dropout_in, dropout,
            subsample, subsample_type, n_stacks, n_splices,
            conv_in_channel, conv_channels, conv_kernel_sizes, conv_strides, conv_poolings,
            conv_batch_norm, conv_layer_norm, conv_bottleneck_dim,
            bidir_sum_fwd_bwd, param_init,
            chunk_size_left, chunk_size_right):
        super().__init__()

        subsamples = [1] * n_layers
        for l, s in enumerate(list(map(int, subsample.split('_')[:n_layers]))):
            subsamples[l] = s

        if len(subsamples) > 0 and len(subsamples) != n_layers:
            raise ValueError(f"subsample must be the same size as n_layers. {n_layers} : {subsamples}")

        self.enc_type = enc_type
        self.bidirectional = True if ('blstm' in enc_type or 'bgru' in enc_type) else False
        self.n_units = n_units
        self.n_dirs = 2 if self.bidirectional else 1
        self.n_layers = n_layers
        self.bidir_sum = bidir_sum_fwd_bwd

        # for latency-controlled
        self.chunk_size_left = int(chunk_size_left.split('_')[0]) // n_stacks
        self.chunk_size_right = int(chunk_size_right.split('_')[0]) // n_stacks
        self.lc_bidir = self.chunk_size_left > 0 or self.chunk_size_right > 0
        if self.lc_bidir:
            assert enc_type not in ('lstm', 'gru', 'conv_lstm', 'conv_gru')

        # for bridge layers
        self.bridge = None

        # Dropout for input-hidden connection
        self.dropout_in = layers.Dropout(dropout_in)

        if 'conv' in enc_type:
            assert n_stacks == 1 and n_splices == 1
            self.conv = ConvEncoder(input_dim,
                        in_channel=conv_in_channel,
                        channels=conv_channels,
                        kernel_sizes=conv_kernel_sizes,
                        strides=conv_strides,
                        poolings=conv_poolings,
                        dropout=0.,
                        batch_norm=conv_batch_norm,
                        layer_norm=conv_layer_norm,
                        residual=False,
                        bottleneck_dim=conv_bottleneck_dim,
                        param_init=param_init)
            self._odim = self.conv.output_dim
        else:
            self.conv = None
            self._odim = input_dim * n_splices * n_stacks

        if enc_type != 'conv':
            self.rnn = []
            if self.lc_bidir:
                self.rnn_bwd = []
            self.dropout = layers.Dropout(dropout)
            self.proj = [] if n_projs > 0 else None
            self.subsample = [] if np.prod(subsamples) > 1 else None
            self.padding = Padding(enc_type=self.enc_type, bidirectional=self.bidirectional)

            for l in range(n_layers):
                if 'lstm' in enc_type:
                    rnn_i = layers.LSTM
                elif 'gru' in enc_type:
                    rnn_i = layers.GRU
                else:
                    raise ValueError('enc_type must be "(conv)(b)lstm" or "(conv)(b)gru".')

                if self.lc_bidir:
                    self.rnn += [rnn_i(n_units, time_major=False, return_sequences=True, return_state=True)]
                    self.rnn_bwd += [rnn_i(n_units, time_major=False,  return_sequences=True, return_state=True)]
                else:
                    self.rnn += [
                         layers.Bidirectional(rnn_i(n_units, time_major=False,  return_sequences=True, return_state=True), merge_mode='sum' if self.bidir_sum else "concat")
                         if self.bidirectional else rnn_i(n_units, time_major=False,  return_sequences=True, return_state=True)]
                self._odim = n_units if bidir_sum_fwd_bwd else n_units * self.n_dirs

                # Projection layer
                if self.proj is not None:
                    if l != n_layers -1:
                        self.proj += [layers.Dense(n_projs)]
                        self._odim = n_projs

                # subsample
                if np.prod(subsamples) > 1:
                    if subsample_type == 'max_pool':
                        self.subsample += [MaxpoolSubsampler(subsamples[l]) ]
                    elif subsample_type == 'concat':
                        self.subsample += [ConcatSubsampler(subsamples[l], self._odim)]
                    elif subsample_type == 'drop':
                        self.subsample += [DropSubsampler(subsamples[l])]
                    elif subsample_type == '1dconv':
                        self.subsample += [Conv1dSubsampler(subsamples[l], self._odim)]
                    elif subsample_type == 'add':
                        self.subsample += [AddSubsampler(subsamples[l])]

            if last_proj_dim > 0 and last_proj_dim != self.output_dim:
                self.bridge = layers.Dense(last_proj_dim)
                self._odim = last_proj_dim

        # calculate subsampling factor
        self._factor = 1
        if self.conv is not None:
            self._factor *= self.conv.subsampling_factor
        elif np.prod(subsamples) > 1:
            self._factor *= np.prod(subsamples)
        # Note: subsampling factor for frame stacking should not be included here
        if self.chunk_size_left > 0:
            assert self.chunk_size_left % self._factor == 0
        if self.chunk_size_right > 0:
            assert self.chunk_size_right % self._factor == 0

        self.reset_parameters(param_init)

    @property
    def output_dim(self):
        return self._odim

    @property
    def subsampling_factor(self):
        return self._factor

    def reset_parameters(self, param_init):
        logger.info('===== Initialize %s with uniform distribution =====' % self.__class__.__name__)
        #for layer in self.layers:
        #    for var in layer.trainable_weights:
        #        print("l:", var.name, " " , var.shape)

        for var in self.trainable_weights:
            if 'conv' in var.name:
                continue
            init_with_uniform(var, param_init)

    def reset_cache(self):
        self.hx_fwd = [None] * self.n_layers
        logger.debug('Reset cache.')

    def call(self, xs, xlens, training=None, streaming=False, lookback=False, lookahead=False):
        """Pass the input through the endocder layers in turn.

        Args:
            src: the sequnce to the encoder (required).
            mask: the mask for the src sequence (optional).
        """
        eouts = {'ys': {'xs': None, 'xlens': None}}

        # dropout for inputs-hidden connection
        xs = self.dropout_in(xs)

        bs, xmax, idim = tf.shape(xs)
        N_l, N_r = self.chunk_size_left, self.chunk_size_right

        # pass through CNN blocks before RNN layers
        if self.conv: 
            xs, xlens = self.conv(xs, xlens, lookback=lookback, lookahead=lookahead)
            if self.enc_type == 'conv':
                eouts['ys']['xs'] = xs
                eouts['ys']['xlen'] = xlens
                return eouts
            if self.lc_bidir:
                N_l = N_l // self.conv.subsampling_factor
                N_r = N_r // self.conv.subsampling_factor

        if not streaming:
            self.reset_cache()

        if self.lc_bidir:
            # Flip the layer and time loop
            if self.chunk_size_left <= 0:
                xs, xlens = self._forward_full_context(xs, xlens)
            else:
                xs, xlens = self._forwad_latency_controlled(xs, xlen, N_l, N_r, streaming)
        else:
            for l in range(self.n_layers):
                xs, state = self.padding(xs, xlens, self.rnn[l], prev_state=self.hx_fwd[l], streaming=streaming)
                self.hx_fwd[l] = state
                xs = self.dropout(xs)

                # projection layer
                if self.proj and l != self.n_layers-1:
                    xs = tf.nn.relu(self.proj[l](xs))
                # subsampling layer
                if self.subsample:
                    xs, xlens = self.subsample[l](xs, xlens)

        # bridge layer
        if self.bridge:
            xs = self.bridge(xs)
            eouts['ys']['xs'], eouts['ys']['xlens'] = xs, xlens
        return eouts


    def _forward_full_context(self, xs, xlens, task='all'):
        """Full context BPTT encoding. This is used for pre-training latency-controlled bidirectional encoder.
            Args:
                xs (FloatTensor): `[B, T, n_units]`
            Returns:
                xs (FloatTensor): `[B, T, n_units]`
        """
        return xs, xlens

    def _forward_latency_controlled(self, xs, xlens, N_l, N_r, streaming, task='all'):
        """Streaming encoding for the conventional latency-controlled bidirectional encoder.
           Args:
               xs (FloatTensor): `[B, T, n_units]`
           Returns:
               xs (FloatTensor): `[B, T, n_units]`
        """
        return xs, xlens

class Padding(tf.keras.layers.Layer):
    '''Padding variable length of sequences.'''
    def __init__(self, enc_type, bidirectional):
        super().__init__()
        self.enc_type = enc_type 
        self.bidirectional = bidirectional

    def call(self, xs, xlens, rnn, prev_state=None, streaming=False):
        layers = tf.keras.layers
        if not streaming and xlens is not None:
            maxstep = tf.shape(xs)[1]
            mask = tf.sequence_mask(xlens, maxlen=maxstep) 
            outs = rnn(xs, mask=mask, initial_state=prev_state)
        else:
            outs = rnn(xs, initial_state=prev_state)

        if self.bidirectional:
            if 'blstm' in self.enc_type:
                assert len(outs) == 5
                xs, h_fwd, c_fwd, h_bwd, c_bwd = outs
                state = (h_fwd, c_fwd, h_bwd, c_bwd)
            elif 'bgru' in self.enc_type:
                assert len(outs) == 3
                xs, h_fwd, h_bwd = outs
                state = (h_fwd, h_bwd)
        else:
            if 'lstm' in self.enc_type:
                assert len(outs) == 3
                xs, h, c = outs
                state = (h, c)
            elif 'gru' in self.enc_type:
                assert len(outs) == 2
                xs, h = outs
                state = h

        return xs, state

if __name__ == '__main__':
   logging.basicConfig(level=logging.INFO)
   xs = tf.random.normal([2, 50, 80])
   xlens = tf.constant([5, 13])

   enc = RNNEncoder(input_dim=80, enc_type='lstm', n_units=100, n_projs=5, last_proj_dim=1024, n_layers=6,
            dropout_in=0.2, dropout=0.2,
            subsample='1_1_1_1_1_1', subsample_type='add', n_stacks=1, n_splices=1,
            conv_in_channel=1, conv_channels='32_32', conv_kernel_sizes='(3,3)_(3,3)', conv_strides='(1,1)_(1,1)', conv_poolings='(2,2)_(2,2)',
            conv_batch_norm=False, conv_layer_norm=False, conv_bottleneck_dim=100,
            bidir_sum_fwd_bwd=True, param_init=0.01,
            chunk_size_left='-1', chunk_size_right='0')
   print(enc.output_dim)
   print(enc.subsampling_factor)
   eouts = enc(xs, xlens)
   print(eouts)


   enc = RNNEncoder(input_dim=80, enc_type='blstm', n_units=100, n_projs=5, last_proj_dim=1024, n_layers=6,
            dropout_in=0.2, dropout=0.2,
            subsample='1_1_1_1_1_1', subsample_type='add', n_stacks=1, n_splices=1,
            conv_in_channel=1, conv_channels='32_32', conv_kernel_sizes='(3,3)_(3,3)', conv_strides='(1,1)_(1,1)', conv_poolings='(2,2)_(2,2)',
            conv_batch_norm=False, conv_layer_norm=False, conv_bottleneck_dim=100,
            bidir_sum_fwd_bwd=True, param_init=0.01,
            chunk_size_left='-1', chunk_size_right='0')
   print(enc.output_dim)
   print(enc.subsampling_factor)
   eouts = enc(xs, xlens)
   print(eouts)



   enc = RNNEncoder(input_dim=80, enc_type='gru', n_units=100, n_projs=5, last_proj_dim=1024, n_layers=6,
            dropout_in=0.2, dropout=0.2,
            subsample='1_1_1_1_1_1', subsample_type='add', n_stacks=1, n_splices=1,
            conv_in_channel=1, conv_channels='32_32', conv_kernel_sizes='(3,3)_(3,3)', conv_strides='(1,1)_(1,1)', conv_poolings='(2,2)_(2,2)',
            conv_batch_norm=False, conv_layer_norm=False, conv_bottleneck_dim=100,
            bidir_sum_fwd_bwd=True, param_init=0.01,
            chunk_size_left='-1', chunk_size_right='0')
   print(enc.output_dim)
   print(enc.subsampling_factor)
   eouts = enc(xs, xlens)
   print(eouts)

   enc = RNNEncoder(input_dim=80, enc_type='bgru', n_units=100, n_projs=5, last_proj_dim=1024, n_layers=6,
            dropout_in=0.2, dropout=0.2,
            subsample='1_1_1_1_1_1', subsample_type='add', n_stacks=1, n_splices=1,
            conv_in_channel=1, conv_channels='32_32', conv_kernel_sizes='(3,3)_(3,3)', conv_strides='(1,1)_(1,1)', conv_poolings='(2,2)_(2,2)',
            conv_batch_norm=False, conv_layer_norm=False, conv_bottleneck_dim=100,
            bidir_sum_fwd_bwd=True, param_init=0.01,
            chunk_size_left='-1', chunk_size_right='0')
   print(enc.output_dim)
   print(enc.subsampling_factor)
   eouts = enc(xs, xlens)
   print(eouts)

   enc = RNNEncoder(input_dim=80, enc_type='bgru', n_units=100, n_projs=5, last_proj_dim=1024, n_layers=6,
            dropout_in=0.2, dropout=0.2,
            subsample='1_2_1_2_1_1', subsample_type='add', n_stacks=1, n_splices=1,
            conv_in_channel=1, conv_channels='32_32', conv_kernel_sizes='(3,3)_(3,3)', conv_strides='(1,1)_(1,1)', conv_poolings='(2,2)_(2,2)',
            conv_batch_norm=False, conv_layer_norm=False, conv_bottleneck_dim=100,
            bidir_sum_fwd_bwd=True, param_init=0.01,
            chunk_size_left='-1', chunk_size_right='0')
   print(enc.output_dim)
   print(enc.subsampling_factor)
   eouts = enc(xs, xlens)
   print(eouts)
   tf.debugging.assert_equal(eouts['ys']['xlens'], tf.constant([2, 4], dtype=tf.int32))

   enc = RNNEncoder(input_dim=80, enc_type='bgru', n_units=100, n_projs=5, last_proj_dim=1024, n_layers=6,
            dropout_in=0.2, dropout=0.2,
            subsample='1_2_1_2_1_1', subsample_type='drop', n_stacks=1, n_splices=1,
            conv_in_channel=1, conv_channels='32_32', conv_kernel_sizes='(3,3)_(3,3)', conv_strides='(1,1)_(1,1)', conv_poolings='(2,2)_(2,2)',
            conv_batch_norm=False, conv_layer_norm=False, conv_bottleneck_dim=100,
            bidir_sum_fwd_bwd=True, param_init=0.01,
            chunk_size_left='-1', chunk_size_right='0')
   print(enc.output_dim)
   print(enc.subsampling_factor)
   eouts = enc(xs, xlens)
   print(eouts)
   tf.debugging.assert_equal(eouts['ys']['xlens'], tf.constant([2, 4], dtype=tf.int32))

   enc = RNNEncoder(input_dim=80, enc_type='bgru', n_units=100, n_projs=5, last_proj_dim=1024, n_layers=6,
            dropout_in=0.2, dropout=0.2,
            subsample='1_2_1_2_1_1', subsample_type='concat', n_stacks=1, n_splices=1,
            conv_in_channel=1, conv_channels='32_32', conv_kernel_sizes='(3,3)_(3,3)', conv_strides='(1,1)_(1,1)', conv_poolings='(2,2)_(2,2)',
            conv_batch_norm=False, conv_layer_norm=False, conv_bottleneck_dim=100,
            bidir_sum_fwd_bwd=True, param_init=0.01,
            chunk_size_left='-1', chunk_size_right='0')
   print(enc.output_dim)
   print(enc.subsampling_factor)
   eouts = enc(xs, xlens)
   print(eouts)
   tf.debugging.assert_equal(eouts['ys']['xlens'], tf.constant([1, 3], dtype=tf.int32))


   enc = RNNEncoder(input_dim=80, enc_type='bgru', n_units=100, n_projs=5, last_proj_dim=1024, n_layers=6,
            dropout_in=0.2, dropout=0.2,
            subsample='1_2_1_2_1_1', subsample_type='max_pool', n_stacks=1, n_splices=1,
            conv_in_channel=1, conv_channels='32_32', conv_kernel_sizes='(3,3)_(3,3)', conv_strides='(1,1)_(1,1)', conv_poolings='(2,2)_(2,2)',
            conv_batch_norm=False, conv_layer_norm=False, conv_bottleneck_dim=100,
            bidir_sum_fwd_bwd=True, param_init=0.01,
            chunk_size_left='-1', chunk_size_right='0')
   print(enc.output_dim)
   print(enc.subsampling_factor)
   eouts = enc(xs, xlens)
   print(eouts)
   tf.debugging.assert_equal(eouts['ys']['xlens'], tf.constant([2, 4], dtype=tf.int32))

   enc = RNNEncoder(input_dim=80, enc_type='bgru', n_units=100, n_projs=5, last_proj_dim=1024, n_layers=6,
            dropout_in=0.2, dropout=0.2,
            subsample='1_2_1_2_1_1', subsample_type='1dconv', n_stacks=1, n_splices=1,
            conv_in_channel=1, conv_channels='32_32', conv_kernel_sizes='(3,3)_(3,3)', conv_strides='(1,1)_(1,1)', conv_poolings='(2,2)_(2,2)',
            conv_batch_norm=False, conv_layer_norm=False, conv_bottleneck_dim=100,
            bidir_sum_fwd_bwd=True, param_init=0.01,
            chunk_size_left='-1', chunk_size_right='0')
   print(enc.output_dim)
   print(enc.subsampling_factor)
   eouts = enc(xs, xlens)
   print(eouts)
   tf.debugging.assert_equal(eouts['ys']['xlens'], tf.constant([2, 4], dtype=tf.int32))
   #print(enc.summary())
   enc.reset_parameters(0.1)
