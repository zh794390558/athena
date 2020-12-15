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
from conv import ConvEncoder
from attention import AttentionMechanism
from criterion import cross_entropy_lsm

logger = logging.getLogger(__name__)

class LAS(tf.keras.layers.Layer):
    """A LAS model. User is able to modify the attributes as needed.

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
    ):
        super().__init__()
        self.encoder = RNNEncoder(
            input_dim=80, enc_type='conv_blstm', n_units=512, n_projs=0, last_proj_dim=0, n_layers=5,
            dropout_in=0.4, dropout=0.4,
            subsample='1_1_1_1_1', subsample_type='drop', n_stacks=1, n_splices=1,
            conv_in_channel=1, conv_channels='32_32', conv_kernel_sizes='(3,3)_(3,3)', conv_strides='(1,1)_(1,1)', conv_poolings='(2,2)_(2,2)',
            conv_batch_norm=False, conv_layer_norm=False, conv_bottleneck_dim=0,
            bidir_sum_fwd_bwd=True, param_init=0.01,
            chunk_size_left='-1', chunk_size_right='0')

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
        if np.prod(subsamples) > 1:
            self._factor *= np.prod(subsamples)
        # Note: subsampling factor for frame stacking should not be included here
        if self.chunk_size_left > 0:
            assert self.chunk_size_left % self._factor == 0
        if self.chunk_size_right > 0:
            assert self.chunk_size_right % self._factor == 0

    @property
    def output_dim(self):
        return self._odim

    @property
    def subsampling_factor(self):
        return self._factor

    def reset_parameters(self, param_init):
        logger.info('===== Initialize %s with uniform distribution =====' % self.__class__.__name__)
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


def np2tensor(array, device=None):
    """Convert form np.ndarray to torch.Tensor.
    Args:
        array (np.ndarray): A tensor of any sizes
    Returns:
        tensor (torch.Tensor):
    """
    tensor = tf.convert_to_tensor(array)
    return tensor

def tensor2np(x):
    """Convert torch.Tensor to np.ndarray.
    Args:
        x (torch.Tensor):
    Returns:
        np.ndarray
    """
    if x is None:
        return x
    return tf.stop_gradient(x).numpy()


def tensor2scalar(x):
    """Convert torch.Tensor to a scalar value.
    Args:
        x (torch.Tensor):
    Returns:
        scaler
    """
    if isinstance(x, float):
        return x
    return tf.stop_gradient(x).numpy()


def pad_list(xs, pad_value=0., pad_left=False):
    """Convert list of Tensors to a single Tensor with padding.
    Args:
        xs (list): A list of length `[B]`, which contains Tensors of size `[T, input_size]`
        pad_value (float):
        pad_left (bool):
    Returns:
        xs_pad (FloatTensor): `[B, T, input_size]`
    """
    bs = len(xs)
    max_time = tf.math.reduce_max([tf.shape(x)[0] for x in xs])
    xs_pad = []
    for b in tf.range(bs):
        if len(xs[b]) == 0:
            continue

        padlen = max_time - tf.shape(xs[b])[0]
        p = tf.ones(padlen) * pad_value
        p = tf.cast(p, xs[b].dtype)
        if pad_left:
            xs_pad.append(
                 tf.concat([p, xs[b]], axis=0)
            )
        else:
            xs_pad.append(
                 tf.concat([xs[b], p], axis=0)
            )
    xs_pad = tf.stack(xs_pad)
    return xs_pad


def make_pad_mask(seq_lens):
    """Make mask for padding.
    Args:
        seq_lens (IntTensor): `[B]`
    Returns:
        mask (IntTensor): `[B, T]`
    """
    bs = tf.shape(seq_lens)[0]
    max_time = tf.math.reduce_max(seq_lens)
    seq_range = tf.range(0, max_time)
    seq_range = tf.tile(tf.expand_dims(seq_range, 0), (bs, 1))
    mask = seq_range < tf.expand_dims(seq_lens, -1)
    return mask


def append_sos_eos(ys, sos, eos, pad, device, bwd=False, replace_sos=False):
    """Append <sos> and <eos> and return padded sequences.
    Args:
        ys (list): A list of length `[B]`, which contains a list of size `[L]`
        sos (int): index for <sos>
        eos (int): index for <eos>
        pad (int): index for <pad>
        bwd (bool): reverse ys for backward reference
        replace_sos (bool): replace <sos> with the special token
    Returns:
        ys_in (LongTensor): `[B, L]`
        ys_out (LongTensor): `[B, L]`
        ylens (IntTensor): `[B]`
    """
    _eos = tf.ones(1, dtype=tf.int64) * eos
    ys = [np2tensor(np.fromiter(y[::-1] if bwd else y, dtype=np.int64),
                    device) for y in ys]
    if replace_sos:
        ylens = np2tensor(np.fromiter([y[1:].shape[0] + 1 for y in ys], dtype=np.int32))  # +1 for <eos>
        ys_in = pad_list([y for y in ys], pad)
        ys_out = pad_list([tf.concat([y[1:], _eos], axis=0) for y in ys], pad)
    else:
        _sos = tf.ones(1, dtype=tf.int64) * sos
        ylens = np2tensor(np.fromiter([y.shape[0] + 1 for y in ys], dtype=np.int32))  # +1 for <eos>
        ys_in = pad_list([tf.concat([_sos, y], axis=0) for y in ys], pad)
        ys_out = pad_list([tf.concat([y, _eos], axis=0) for y in ys], pad)
    return ys_in, ys_out, ylens


def compute_accuracy(logits, ys_ref, pad):
    """Compute teacher-forcing accuracy.
    Args:
        logits (FloatTensor): `[B, T, vocab]`
        ys_ref (LongTensor): `[B, T]`
        pad (int): index for padding
    Returns:
        acc (float): teacher-forcing accuracy
    """
    _, _, vocab = tf.shape(logits)
    bs, time = tf.shape(ys_ref)
    logits = tf.reshape(logits, (bs, time, vocab))
    pad_pred = tf.math.argmax(logits, 2)
    mask = tf.cast(ys_ref != pad, pad_pred.dtype)
    pad_pred *= mask
    ys_ref *= mask
    numerator = tf.reduce_sum(tf.cast(pad_pred == ys_ref, tf.float32))
    denominator = tf.reduce_sum(tf.cast(mask, tf.float32))
    acc = numerator * 100 / denominator
    return acc


class RNNDecoder(tf.keras.layers.Layer):
    """RNN decoder.
    Args:
        special_symbols (dict):
            eos (int): index for <eos> (shared with <sos>)
            unk (int): index for <unk>
            pad (int): index for <pad>
            blank (int): index for <blank>
        enc_n_units (int): number of units of encoder outputs
        attn_type (str): type of attention mechanism
        rnn_type (str): lstm/gru
        n_units (int): number of units in each RNN layer
        n_projs (int): number of units in each projection layer
        n_layers (int): number of RNN layers
        bottleneck_dim (int): dimension of bottleneck layer before the softmax layer for label generation
        emb_dim (int): dimension of embedding in target spaces.
        vocab (int): number of nodes in softmax layer
        tie_embedding (bool): tie parameters of embedding and output layers
        attn_dim (int): dimension of attention space
        attn_sharpening_factor (float): sharpening factor in softmax for attention
        attn_sigmoid_smoothing (bool): replace softmax with sigmoid for attention calculation
        attn_conv_out_channels (int): channel size of convolution in location-aware attention
        attn_conv_kernel_size (int): kernel size of convolution in location-aware attention
        attn_n_heads (int): number of attention heads
        dropout (float): dropout probability for RNN layer
        dropout_emb (float): dropout probability for embedding layer
        dropout_att (float): dropout probability for attention distributions
        lsm_prob (float): label smoothing probability
        ss_prob (float): scheduled sampling probability
        ctc_weight (float): CTC loss weight
        ctc_lsm_prob (float): label smoothing probability for CTC
        ctc_fc_list (list): fully-connected layer configuration before the CTC softmax
        mbr_training (bool): MBR training
        mbr_ce_weight (float): CE weight for regularization during MBR training
        external_lm (RNNLM): external RNNLM for LM fusion/initialization
        lm_fusion (str): type of LM fusion
        lm_init (bool): initialize decoder with pre-trained LM
        backward (bool): decode in the backward order
        global_weight (float): global loss weight for multi-task learning, ctc+att
        mtl_per_batch (bool): change mini-batch per task for multi-task training
        param_init (float): parameter initialization
        gmm_attn_n_mixtures (int): number of mixtures for GMM attention
        replace_sos (bool): replace <sos> with special tokens
        distil_weight (float): soft label weight for knowledge distillation
        discourse_aware (str): state_carry_over
    """

    def __init__(self, special_symbols,
                 enc_n_units, attn_type, rnn_type, n_units, n_projs, n_layers,
                 bottleneck_dim, emb_dim, vocab, tie_embedding,
                 attn_dim, attn_sharpening_factor, attn_sigmoid_smoothing,
                 attn_conv_out_channels, attn_conv_kernel_size, attn_n_heads,
                 dropout, dropout_emb, dropout_att,
                 lsm_prob, ss_prob,
                 ctc_weight, ctc_lsm_prob, ctc_fc_list,
                 mbr_training, mbr_ce_weight,
                 external_lm, lm_fusion, lm_init,
                 backward, global_weight, mtl_per_batch, param_init,
                 gmm_attn_n_mixtures, replace_sos, distillation_weight, discourse_aware):

        super(RNNDecoder, self).__init__()

        self.eos = special_symbols['eos']
        self.unk = special_symbols['unk']
        self.pad = special_symbols['pad']
        self.blank = special_symbols['blank']
        self.vocab = vocab
        self.attn_type = attn_type
        self.rnn_type = rnn_type
        assert rnn_type in ['lstm', 'gru']
        self.enc_n_units = enc_n_units
        self.dec_n_units = n_units
        self.n_projs = n_projs
        self.n_layers = n_layers
        self.lsm_prob = lsm_prob
        self.ss_prob = ss_prob
        self._ss_prob = 0  # for curriculum
        self.att_weight = global_weight - ctc_weight
        self.ctc_weight = ctc_weight
        self.lm_fusion = lm_fusion
        self.mbr = None 
        self.bwd = backward
        self.mtl_per_batch = mtl_per_batch
        self.replace_sos = replace_sos
        self.distil_weight = distillation_weight

        # for contextualization
        self.discourse_aware = discourse_aware
        self.dstate_prev = None
        self._new_session = False

        self.prev_spk = ''
        self.dstates_final = None
        self.lmstate_final = None
        self.lmmemory = None

        # for attention plot
        self.aws_dict = {}
        self.data_dict = {}

        if ctc_weight > 0:
            self.ctc = CTC(eos=self.eos,
                           blank=self.blank,
                           enc_n_units=enc_n_units,
                           vocab=vocab,
                           dropout=dropout,
                           lsm_prob=ctc_lsm_prob,
                           fc_list=ctc_fc_list,
                           param_init=param_init)


        if self.att_weight > 0:
            # Attention layer
            qdim = n_units if n_projs == 0 else n_projs
            if attn_type == 'mocha':
                assert attn_n_heads == 1

            elif attn_type == 'gmm':
                self.score = GMMAttention(enc_n_units, qdim, attn_dim,
                                          n_mixtures=gmm_attn_n_mixtures)
            else:
                if attn_n_heads > 1:
                    assert attn_type == 'add'
                    self.score = MultiheadAttentionMechanism(
                        enc_n_units, qdim, attn_dim, enc_n_units,
                        n_heads=attn_n_heads,
                        dropout=dropout_att,
                        atype='add')
                else:
                    self.score = AttentionMechanism(
                        enc_n_units, qdim, attn_dim, attn_type,
                        sharpening_factor=attn_sharpening_factor,
                        sigmoid_smoothing=attn_sigmoid_smoothing,
                        conv_out_channels=attn_conv_out_channels,
                        conv_kernel_size=attn_conv_kernel_size,
                        dropout=dropout_att,
                        lookahead=2)


        # Decoder
        self.rnn = []
        cell = layers.LSTMCell if rnn_type == 'lstm' else layers.GRUCell
        dec_odim = enc_n_units + emb_dim
        self.proj = [nn.Linear(n_units, n_projs) for i in range(n_layers)] if n_projs > 0 else None
        self.dropout = layers.Dropout(dropout)
        for _ in range(n_layers):
            #self.rnn += [cell(dec_odim, n_units)]
            self.rnn += [cell(n_units)]
            dec_odim = n_projs if n_projs > 0 else n_units

        # RNNLM fusion
        if external_lm is not None and lm_fusion:
            #self.linear_dec_feat = nn.Linear(dec_odim + enc_n_units, n_units)
            self.linear_dec_feat = layers.Dense(n_units)
            if lm_fusion in ['cold', 'deep']:
                #self.linear_lm_feat = nn.Linear(external_lm.output_dim, n_units)
                self.linear_lm_feat = layers.Dense(n_units)
                #self.linear_lm_gate = nn.Linear(n_units * 2, n_units)
                self.linear_lm_gate = layers.Dense(n_units)
            elif lm_fusion == 'cold_prob':
                #self.linear_lm_feat = nn.Linear(external_lm.vocab, n_units)
                self.linear_lm_feat = layers.Dense(n_units)
                #self.linear_lm_gate = nn.Linear(n_units * 2, n_units)
                self.linear_lm_gate = layers.Dense(n_units)
            else:
                raise ValueError(lm_fusion)
            #self.output_bn = nn.Linear(n_units * 2, bottleneck_dim)
            self.output_bn = layers.Dense(bottleneck_dim)
        else:
            #self.output_bn = nn.Linear(dec_odim + enc_n_units, bottleneck_dim)
            self.output_bn = layers.Dense(bottleneck_dim)


        #self.embed = nn.Embedding(vocab, emb_dim, padding_idx=self.pad)
        self.embed = layers.Embedding(vocab, emb_dim)
        self.dropout_emb = layers.Dropout(dropout_emb)
        assert bottleneck_dim > 0, 'bottleneck_dim must be larger than zero.'
        # (bottleneck_dim, vocab)
        self.output_layer = layers.Dense(vocab)
        if tie_embedding:
            if emb_dim != bottleneck_dim:
                raise ValueError('When using the tied flag, n_units must be equal to emb_dim.')
            self.output_layer.kernel.assing(self.embed.kernel)

        #self.reset_parameters(param_init)

        # resister the external RNNLM
        self.lm = external_lm if lm_fusion else None

        # decoder initialization with pre-trained RNNLM
        if lm_init:
            pass

    def reset_parameters(self, param_init):
        """Initialize parameters with uniform distribution."""
        logger.info('===== Initialize %s with uniform distribution =====' % self.__class__.__name__)
        for var in self.trainable_weights:
            n = var.name
            if 'score.monotonic_energy.v.weight_g' in n or 'score.monotonic_energy.r' in n:
                logger.info('Skip initialization of %s' % n)
                continue
            if 'score.chunk_energy.v.weight_g' in n or 'score.chunk_energy.r' in n:
                logger.info('Skip initialization of %s' % n)
                continue
            if 'linear_lm_gate.fc.bias' in n and var.shape.rank == 1:
                # Initialize bias in gating with -1 for cold fusion
                nn.init.constant_(p, -1.)  # bias
                var.assign(tf.constant(-1))
                logger.info('Initialize %s with %s / %.3f' % (n, 'constant', -1.))
                continue

            init_with_uniform(var, param_init)


    @property
    def training(self):
        return tf.keras.backend.learning_phase()

    def call(self, eouts, elens, ys, task='all',
        teacher_logits=None, recog_params={}, idx2token=None, trigger_points=None):
        """Forward pass.
        Args:
            eouts (FloatTensor): `[B, T, enc_n_units]`
            elens (IntTensor): `[B]`
            ys (list): length `B`, each of which contains a list of size `[L]`
            task (str): all/ys*/ys_sub*
            teacher_logits (FloatTensor): `[B, L, vocab]`
            recog_params (dict): parameters for MBR training
            idx2token ():
            trigger_points (np.ndarray): `[B, L]`
        Returns:
            loss (FloatTensor): `[1]`
            observation (dict):
        """
        observation = {'loss': None, 'loss_att': None, 'loss_ctc': None, 'loss_mbr': None,
                       'acc_att': None, 'ppl_att': None}
        loss = tf.zeros(1) 

        # CTC loss
        if self.ctc_weight > 0 and (task == 'all' or 'ctc' in task):
            ctc_forced_align = (
                'ctc_sync' in self.latency_metric and self.training) or self.attn_type == 'triggered_attention'
            loss_ctc, ctc_trigger_points = self.ctc(eouts, elens, ys, forced_align=ctc_forced_align)
            observation['loss_ctc'] = tensor2scalar(loss_ctc)
            if self.mtl_per_batch:
                loss += loss_ctc
            else:
                loss += loss_ctc * self.ctc_weight
        else:
            ctc_trigger_points = None

        # XE loss
        if self.att_weight > 0 and (task == 'all' or 'ctc' not in task) and self.mbr is None:
            loss_att, acc_att, ppl_att, loss_quantity, loss_latency = self.forward_att(
                eouts, elens, ys, teacher_logits=teacher_logits,
                ctc_trigger_points=ctc_trigger_points, forced_trigger_points=trigger_points)
            observation['loss_att'] = tensor2scalar(loss_att)
            observation['acc_att'] = acc_att
            observation['ppl_att'] = ppl_att
            if self.mtl_per_batch:
                loss += loss_att
            else:
                loss += loss_att * self.att_weight

        # MBR loss
        if self.mbr is not None and (task == 'all' or 'mbr' not in task):
            pass

        observation['loss'] = tensor2scalar(loss)
        return loss, observation

    def zero_state(self, bs):
        """Initialize decoder state.
        Args:
            bs (int): batch size
        Returns:
            dstates (dict):
                dout (FloatTensor): `[B, 1, dec_n_units]`
                dstate (tuple): A tuple of (hxs, cxs)
                    hxs (FloatTensor): `[n_layers, B, dec_n_units]`
                    cxs (FloatTensor): `[n_layers, B, dec_n_units]`
        """
        dstates = {'dstate': None}
        # hxs, cxs
        hxs = tf.zeros([self.n_layers, bs, self.dec_n_units])
        cxs = tf.zeros([self.n_layers, bs, self.dec_n_units]) if self.rnn_type == 'lstm' else None
        dstates['dstate'] = (hxs, cxs)
        return dstates


    def forward_att(self, eouts, elens, ys,
                    return_logits=False, teacher_logits=None,
                    ctc_trigger_points=None, forced_trigger_points=None):
        """Compute XE loss for the attention-based decoder.
        Args:
            eouts (FloatTensor): `[B, T, enc_n_units]`
            elens (IntTensor): `[B]`
            ys (list): length `B`, each of which contains a list of size `[L]`
            return_logits (bool): return logits for knowledge distillation
            teacher_logits (FloatTensor): `[B, L, vocab]`
            ctc_trigger_points (IntTensor): `[B, L]`
            forced_trigger_points (IntTensor): `[B, L]`
        Returns:
            loss (FloatTensor): `[1]`
            acc (float): accuracy for token prediction
            ppl (float): perplexity
            loss_quantity (FloatTensor): `[1]`
            loss_latency (FloatTensor): `[1]`
        """
        bs, xmax = tf.shape(eouts)[:2]

        # Append <sos> and <eos>
        print('ys:', ys)
        ys_in, ys_out, ylens = append_sos_eos(ys, self.eos, self.eos, self.pad, eouts.device, self.bwd)
        print('ysin:', ys_in)
        print('ysout:', ys_out)
        print('yslen:', ylens)
        ymax = tf.shape(ys_in)[1]

        # Initialization
        dstates = self.zero_state(bs)
        if self.training:
            if self.discourse_aware and not self._new_session:
                dstates = {'dstate': (self.dstate_prev['hxs'], self.dstate_prev['cxs'])}
            self.dstate_prev = {'hxs': [None] * bs, 'cxs': [None] * bs}
            self._new_session = False
        cv = tf.zeros([bs, 1, self.enc_n_units])
        self.score.reset()
        aw, aws = None, []
        betas = []
        p_chooses = []
        lmout, lmstate = None, None

        ys_emb = self.dropout_emb(self.embed(ys_in))
        src_mask = tf.expand_dims(make_pad_mask(elens), axis=1)  # `[B, 1, T]`
        tgt_mask = tf.expand_dims((ys_out != self.pad), 2)  # `[B, L, 1]`
        logits = []
        for i in tf.range(ymax):
            is_sample = i > 0 and self._ss_prob > 0 and random.random() < self._ss_prob

            # Update LM states for LM fusion
            if self.lm is not None:
                pass
                #self.lm.eval()
                #with torch.no_grad():
                #    y_lm = self.output_layer(logits[-1]).detach().argmax(-1) if is_sample else ys_in[:, i:i + 1]
                #    lmout, lmstate, _ = self.lm.predict(y_lm, lmstate)

            # Recurrency -> Score -> Generate
            y_emb = self.dropout_emb(self.embed(
                tf.math.argmax(tf.stop_gradient(self.output_layer(logits[-1])), -1))) if is_sample else ys_emb[:, i:i + 1]
            if is_sample:
                print('ii', tf.math.argmax(tf.stop_gradient(self.output_layer(logits[-1])), -1))
            print('xx', y_emb.shape)
            dstates, cv, aw, attn_v, beta, p_choose = self.decode_step(
                eouts, dstates, cv, y_emb, src_mask, aw, lmout, mode='parallel',
                trigger_points=forced_trigger_points[:, i:i + 1] if forced_trigger_points is not None else None)
            aws.append(aw)  # `[B, H, 1, T]`
            if beta is not None:
                betas.append(beta)  # `[B, H, 1, T]`
            if p_choose is not None:
                p_chooses.append(p_choose)  # `[B, H, 1, T]`
            logits.append(attn_v)

            if self.training and self.discourse_aware:
                for b in [b for b, ylen in enumerate(ylens.tolist()) if i == ylen - 1]:
                    self.dstate_prev['hxs'][b] = dstates['dstate'][0][:, b:b + 1].detach()
                    if self.rnn_type == 'lstm':
                        self.dstate_prev['cxs'][b] = dstates['dstate'][1][:, b:b + 1].detach()

        if self.training and self.discourse_aware:
            if bs > 1:
                self.dstate_prev['hxs'] = tf.concat(self.dstate_prev['hxs'], axis=1)
                if self.rnn_type == 'lstm':
                    self.dstate_prev['cxs'] = tf.concat(self.dstate_prev['cxs'], axis=1)
            else:
                self.dstate_prev['hxs'] = self.dstate_prev['hxs'][0]
                if self.rnn_type == 'lstm':
                    self.dstate_prev['cxs'] = self.dstate_prev['cxs'][0]

        logits = self.output_layer(tf.concat(logits, axis=1))

        # for knowledge distillation
        if return_logits:
            return logits

        # for attention plot
        aws = tf.concat(aws, axis=2)  # `[B, H(head), L(dec), T(enc)]`
        if not self.training:
            self.data_dict['elens'] = tensor2np(elens)
            self.data_dict['ylens'] = tensor2np(ylens)
            self.data_dict['ys'] = tensor2np(ys_out)
            self.aws_dict['xy_aws'] = tensor2np(aws)
            if len(betas) > 0:
                betas = torch.cat(betas, dim=2)  # `[B, H, L, T]`
                self.aws_dict['xy_aws_beta'] = tensor2np(betas)
            if len(p_chooses) > 0:
                p_chooses = torch.cat(p_chooses, dim=2)  # `[B, H, L, T]`
                self.aws_dict['xy_p_choose'] = tensor2np(p_chooses)

        n_heads = tf.shape(aws)[1]  # mono

        # Compute XE sequence loss (+ label smoothing)
        loss, ppl = cross_entropy_lsm(logits, ys_out, self.lsm_prob, self.pad, self.training)

        # Attention padding
        if self.attn_type == 'mocha' or (ctc_trigger_points is not None or forced_trigger_points is not None):
            aws = aws.masked_fill_(tgt_mask.unsqueeze(1).repeat([1, n_heads, 1, 1]) == 0, 0)
            # NOTE: attention padding is quite effective for quantity loss

        # Quantity loss
        loss_quantity = 0.

        # Latency loss
        loss_latency = 0.

        # Knowledge distillation
        if teacher_logits is not None:
            kl_loss = distillation(logits, teacher_logits, ylens, temperature=5.0)
            loss = loss * (1 - self.distil_weight) + kl_loss * self.distil_weight

        # Compute token-level accuracy in teacher-forcing
        acc = compute_accuracy(logits, ys_out, self.pad)

        return loss, acc, ppl, loss_quantity, loss_latency

    def decode_step(self, eouts, dstates, cv, y_emb, mask, aw, lmout,
                    mode='hard', trigger_points=None, cache=True):
        print('step:')
        print('yemb', y_emb.shape)
        print('cv', cv.shape)
        dstates = self.recurrency(tf.concat([y_emb, cv], axis=-1), dstates['dstate'])
        cv, aw, beta, p_choose = self.score(eouts, eouts, dstates['dout_score'], mask, aw,
                                            cache=cache, mode=mode, trigger_points=trigger_points)
        print('cv', cv.shape)
        attn_v = self.generate(cv, dstates['dout_gen'], lmout)
        print('attn_v', attn_v.shape)
        return dstates, cv, aw, attn_v, beta, p_choose

    def recurrency(self, inputs, dstate):
        """Recurrency function.
        Args:
            inputs (FloatTensor): `[B, 1, emb_dim + enc_n_units]`
            dstate (tuple): A tuple of (hxs, cxs)
        Returns:
            new_dstates (dict):
                dout_score (FloatTensor): `[B, 1, dec_n_units]`
                dout_gen (FloatTensor): `[B, 1, dec_n_units]`
                dstate (tuple): A tuple of (hxs, cxs)
                    hxs (FloatTensor): `[n_layers, B, dec_n_units]`
                    cxs (FloatTensor): `[n_layers, B, dec_n_units]`
        """
        hxs, cxs = dstate
        dout = tf.squeeze(inputs, 1)

        new_dstates = {'dout_score': None,  # for attention scoring
                       'dout_gen': None,  # for token generation
                       'dstate': None}

        new_hxs, new_cxs = [], []
        for lth in range(self.n_layers):
            if self.rnn_type == 'lstm':
                out, (h, c) = self.rnn[lth](dout, (hxs[lth], cxs[lth]))
                new_cxs.append(c)
            elif self.rnn_type == 'gru':
                h = self.rnn[lth](dout, hxs[lth])
            new_hxs.append(h)
            dout = self.dropout(h)
            if self.proj is not None:
                dout = tf.math.tanh(self.proj[lth](dout))
            # use output in the first layer for attention scoring
            if lth == 0:
                new_dstates['dout_score'] = tf.expand_dims(dout, 1)
        new_hxs = tf.stack(new_hxs, axis=0)
        if self.rnn_type == 'lstm':
            new_cxs = tf.stack(new_cxs, axis=0)

        # use oupput in the the last layer for label generation
        new_dstates['dout_gen'] = tf.expand_dims(dout, 1)
        new_dstates['dstate'] = (new_hxs, new_cxs)
        return new_dstates


    def generate(self, cv, dout, lmout):
        """Generate function.
        Args:
            cv (FloatTensor): `[B, 1, enc_n_units]`
            dout (FloatTensor): `[B, 1, dec_n_units]`
            lmout (FloatTensor): `[B, 1, lm_n_units]`
        Returns:
            attn_v (FloatTensor): `[B, 1, vocab]`
        """
        gated_lmout = None
        if self.lm is not None:
            # LM fusion
            dec_feat = self.linear_dec_feat(tf.concat([dout, cv], axis=-1))

            if self.lm_fusion in ['cold', 'deep']:
                lmout = self.linear_lm_feat(lmout)
                gate = tf.math.sigmoid(self.linear_lm_gate(tf.concat([dec_feat, lmout], axis=-1)))
                gated_lmout = gate * lmout
            elif self.lm_fusion == 'cold_prob':
                lmout = self.linear_lm_feat(self.lm.output(lmout))
                gate = tf.math.sigmoid(self.linear_lm_gate(tf.concat([dec_feat, lmout], axis=-1)))
                gated_lmout = gate * lmout

            out = self.output_bn(tf.concat([dec_feat, gated_lmout], axis=-1))
        else:
            out = self.output_bn(tf.concat([dout, cv], axis=-1))
        attn_v = tf.math.tanh(out)
        return attn_v

    def greedy(self, eouts, elens, max_len_ratio, idx2token,
            exclude_eos=False, refs_id=None, utt_ids=None, speakers=None,
            trigger_points=None):
        """Greedy decoding.
        Args:
            eouts (FloatTensor): `[B, T, enc_units]`
            elens (IntTensor): `[B]`
            max_len_ratio (int): maximum sequence length of tokens
            idx2token (): converter from index to token
            exclude_eos (bool): exclude <eos> from hypothesis
            refs_id (list): reference list
            utt_ids (list): utterance id list
            speakers (list): speaker list
            trigger_points (IntTensor): `[B, T]`
        Returns:
            hyps (list): length `B`, each of which contains arrays of size `[L]`
            aws (list): length `B`, each of which contains arrays of size `[H, L, T]`
        """
        bs, xmax = tf.shape(eouts)[:2]
        
        #initialization
        dstates = self.zero_state(bs) 
        if self.discourse_aware and not self._new_session:
            dstates = {'dstate': (self.dstate_prev['hxs'], self.dstate_prev['cxs'])}
        #self.dstate_prev = {'hxs': [None] * bs, 'cxs': [None] * bs}
        self.dstate_prev = {'hxs': [tf.zeros(0)] * bs, 'cxs': [tf.zeros(0)] * bs}
        self._new_session = False
        cv = tf.zeros((bs, 1, self.enc_n_units))
        self.score.reset()
        aw = None
        lmout, lmstate = None, None
        y = tf.ones((bs, 1), dtype=tf.int64) 
        y = y * refs_id[0][0] if self.replace_sos else y * self.eos

        # Create the attention mask
        src_mask = tf.expand_dims(make_pad_mask(elens), axis=1)  # `[B, 1, T]`

        if self.attn_type == 'triggered_attention':
            assert trigger_points is not None

        hyps_batch, aws_batch = [], []
        ylens = [tf.zeros(1, dtype=tf.int64) for b in tf.range(bs)]
        eos_flags = [False for b in tf.range(bs)]
        ymax = tf.math.ceil(tf.cast(xmax, tf.float32) * max_len_ratio)
        for i in tf.range(ymax):
            # Update LM states for LM fusion
            if self.lm is not None:
                lmout, lmstate, _ = self.lm.predict(y, lmstate)

            # Recurrency -> Score -> Generate
            y_emb = self.dropout_emb(self.embed(y))
            dstates, cv, aw, attn_v, _, _ = self.decode_step(
                eouts, dstates, cv, y_emb, src_mask, aw, lmout,
                trigger_points=trigger_points[:, i:i + 1] if trigger_points is not None else None)
            aws_batch += [aw]  # `[B, H, 1, T]`

            # Pick up 1-best
            y = self.output_layer(attn_v)
            y = tf.math.argmax(y, -1)
            hyps_batch += [y]

            # Count lengths of hypotheses
            for b in tf.range(bs):
                if not eos_flags[b]:
                    if y[b] == self.eos:
                        eos_flags[b] = True
                        if self.discourse_aware:
                            self.dstate_prev['hxs'][b] = dstates['dstate'][0][:, b:b + 1]
                            if self.rnn_type == 'lstm':
                                self.dstate_prev['cxs'][b] = dstates['dstate'][1][:, b:b + 1]
                    ylens[b] += 1  # include <eos>

            # Break if <eos> is outputed in all mini-batch
            if sum(eos_flags) == bs:
                break
            if i == ymax - 1:
                break

        # ASR state carry over
        if self.discourse_aware:
            if bs > 1:
                self.dstate_prev['hxs'] = tf.concat(self.dstate_prev['hxs'], dim=1)
                if self.rnn_type == 'lstm':
                    self.dstate_prev['cxs'] = tf.concat(self.dstate_prev['cxs'], dim=1)
            else:
                self.dstate_prev['hxs'] = self.dstate_prev['hxs']
                if self.rnn_type == 'lstm':
                    self.dstate_prev['cxs'] = self.dstate_prev['cxs']

        # LM state carry over
        self.lmstate_final = lmstate

        # Concatenate in L dimension
        ylens = tensor2np(tf.concat(ylens, axis=0))
        hyps_batch = tensor2np(tf.concat(hyps_batch, axis=1))
        aws_batch = tensor2np(tf.concat(aws_batch, axis=2))  # `[B, H, L, T]`

        # Truncate by the first <eos> (<sos> in case of the backward decoder)
        if self.bwd:
            # Reverse the order
            hyps = [hyps_batch[b, :ylens[b]][::-1] for b in range(bs)]
            aws = [aws_batch[b, :, :ylens[b]][::-1] for b in range(bs)]
        else:
            hyps = [hyps_batch[b, :ylens[b]] for b in range(bs)]
            aws = [aws_batch[b, :, :ylens[b]] for b in range(bs)]

        # Exclude <eos> (<sos> in case of the backward decoder)
        if exclude_eos:
            if self.bwd:
                hyps = [hyps[b][1:] if eos_flags[b] else hyps[b] for b in range(bs)]
                aws = [aws[b][:, 1:] if eos_flags[b] else aws[b] for b in range(bs)]
            else:
                hyps = [hyps[b][:-1] if eos_flags[b] else hyps[b] for b in range(bs)]
                aws = [aws[b][:, :-1] if eos_flags[b] else aws[b] for b in range(bs)]


        if idx2token is not None:
            for b in range(bs):
                if utt_ids is not None:
                    logger.debug('Utt-id: %s' % utt_ids[b])
                if refs_id is not None and self.vocab == idx2token.vocab:
                    logger.debug('Ref: %s' % idx2token(refs_id[b]))
                if self.bwd:
                    logger.debug('Hyp: %s' % idx2token(hyps[b][::-1]))
                else:
                    logger.debug('Hyp: %s' % idx2token(hyps[b]))
                logger.info('=' * 200)
                # NOTE: do not show with logger.info here

        return hyps, aws


if __name__ == '__main__':
   logging.basicConfig(level=logging.INFO)

   logging.info("encoder:")
   xs = tf.random.normal([2, 13, 80])
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


   xs = tf.random.normal([2, 23, 80])
   xlens = tf.constant([14, 23])
   enc = RNNEncoder(input_dim=80, enc_type='conv_blstm', n_units=100, n_projs=5, last_proj_dim=1024, n_layers=6,
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
   tf.debugging.assert_equal(eouts['ys']['xlens'], tf.constant([1, 2], dtype=tf.int32))
   #print(enc.summary())


   xs = tf.random.normal([2, 23, 80])
   xlens = tf.constant([14, 23])
   enc = RNNEncoder(input_dim=80, enc_type='conv_blstm', n_units=16, n_projs=0, last_proj_dim=0, n_layers=4,
            dropout_in=0.2, dropout=0.2,
            subsample='1_1_1_1', subsample_type='1dconv', n_stacks=1, n_splices=1,
            conv_in_channel=1, conv_channels='32_32', conv_kernel_sizes='(3,3)_(3,3)', conv_strides='(1,1)_(1,1)', conv_poolings='(2,2)_(2,2)',
            conv_batch_norm=False, conv_layer_norm=False, conv_bottleneck_dim=0,
            bidir_sum_fwd_bwd=True, param_init=0.01,
            chunk_size_left='-1', chunk_size_right='0')
   print(enc.output_dim)
   print(enc.subsampling_factor)
   eouts = enc(xs, xlens)
   print(eouts)
   tf.debugging.assert_equal(eouts['ys']['xlens'], tf.constant([4, 6], dtype=tf.int32))
   #print(enc.summary())

   logging.info("decoder:")
   ENC_N_UNITS = 16 
   VOCAB = 10
   assert(ENC_N_UNITS == enc.output_dim)

   #ylens = [4, 5, 3, 7]
   ylens = [4, 7]
   mylen = max(ylens)
   ys = [np.random.randint(0, high=VOCAB, size=ylen).astype(np.int32) for ylen in ylens]
   #ys = [tf.constant(np.random.randint(0, high=VOCAB, size=ylen).astype(np.int32)) for ylen in ylens]
   #ys = pad_list(ys, -1)

   print(ys)
   print(ylens)
   #ylens = tf.constant(ylens)
   #ys = tf.constant(ys)

   dec = RNNDecoder(
        special_symbols={'blank': 0, 'unk': 1, 'eos': 2, 'pad': 3},
        enc_n_units=ENC_N_UNITS,
        attn_type='location',
        rnn_type='lstm',
        n_units=16,
        n_projs=0,
        n_layers=2,
        bottleneck_dim=8,
        emb_dim=8,
        vocab=VOCAB,
        tie_embedding=False,
        attn_dim=16,
        attn_sharpening_factor=1.0,
        attn_sigmoid_smoothing=False,
        attn_conv_out_channels=10,
        attn_conv_kernel_size=201,
        attn_n_heads=1,
        dropout=0.1,
        dropout_emb=0.1,
        dropout_att=0.1,
        lsm_prob=0.0,
        ss_prob=0.0,
        ctc_weight=0.0,
        ctc_lsm_prob=0.1,
        ctc_fc_list='16_16',
        mbr_training=False,
        mbr_ce_weight=0.01,
        external_lm=None,
        lm_fusion='',
        lm_init=False,
        backward=False,
        global_weight=1.0,
        mtl_per_batch=False,
        param_init=0.1,
        gmm_attn_n_mixtures=1,
        replace_sos=False,
        distillation_weight=0.0,
        discourse_aware=False)

   outs = dec.call(eouts['ys']['xs'], eouts['ys']['xlens'], ys, task='all', teacher_logits=None, recog_params={}, idx2token=None, trigger_points=None)
   print(outs)
   hyps, aws = dec.greedy(eouts['ys']['xs'], eouts['ys']['xlens'], 0.8, idx2token=None,
                exclude_eos=False, refs_id=None, utt_ids=None, speakers=None,
                trigger_points=None)
   print(hyps)
   print(aws)
