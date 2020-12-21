# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Xiangang Li; Dongwei Jiang; Xiaoning Lei
# All rights reserved.
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
# Only support eager mode
# pylint: disable=no-member, invalid-name, relative-beyond-top-level
# pylint: disable=too-many-locals, too-many-statements, too-many-arguments, too-many-instance-attributes

""" speech transformer implementation"""

from absl import logging
import tensorflow as tf
from .base import BaseModel
from ..loss import Seq2SeqSparseCategoricalCrossentropy
from ..metrics import Seq2SeqSparseCategoricalAccuracy
from ..utils.misc import generate_square_subsequent_mask, insert_sos_in_labels, create_multihead_mask
from ..layers.commons import PositionalEncoding
from ..layers.las import LAS 
from ..utils.hparam import register_and_parse_hparams


class SpeechLAS(BaseModel):
    """ Standard implementation of a SpeechTransformer. Model mainly consists of three parts:
    the x_net for input preparation, the y_net for output preparation and the transformer itself
    """
    default_config = {
        "return_encoder_output": False,
        "num_filters": 512,
        "d_model": 512,
        "num_heads": 8,
        "num_encoder_layers": 12,
        "num_decoder_layers": 6,
        "dff": 1280,
        "rate": 0.1,
        "schedual_sampling_rate": 0.9,
        "label_smoothing_rate": 0.0,
        "unidirectional": False,
        "look_ahead": 0,
        "conv_module_kernel_size": 0
    }

    def __init__(self, data_descriptions, config=None):
        super().__init__()
        self.hparams = register_and_parse_hparams(self.default_config, config['model_config'], cls=self.__class__)

        self.unk = data_descriptions.text_featurizer.unk_index
        self.num_class = data_descriptions.num_class + 1
        self.sos = self.num_class - 1
        self.eos = self.num_class - 1
        self.num_class = self.num_class + 1
        self.blank = self.num_class -1
        self.pad = 0

        ls_rate = self.hparams.label_smoothing_rate

        self.loss_function = Seq2SeqSparseCategoricalCrossentropy(
            num_classes=self.num_class, eos=self.eos, label_smoothing=ls_rate
        )
        self.metric = Seq2SeqSparseCategoricalAccuracy(eos=self.eos, name="acc_att")

        # for deployment
        self.data_descriptions = data_descriptions
        self.deploy_encoder = None
        self.deploy_decoder = None

        # for the x_net
        layers = tf.keras.layers
        xs = layers.Input(shape=data_descriptions.sample_shape["input"], dtype=tf.float32)
        xlens = layers.Input(shape=data_descriptions.sample_shape["input_length"], dtype=tf.int32)
        ys = layers.Input(shape=data_descriptions.sample_shape["output"], dtype=tf.int32)
        ylens = layers.Input(shape=data_descriptions.sample_shape["output_length"], dtype=tf.int32)


        # transformer layer
        self.las = LAS(self.hparams, self.num_class, self.blank, self.sos, self.eos, self.unk, self.pad)
        #self.las = LAS(self.hparams, self.num_class, self.blank, self.sos, self.eos, self.unk, self.pad, dynamic=True)

        #output = self.las(xs, xlens, ys, ylens)
        #self.model = tf.keras.Model(inputs=(xs, xlens, ys, ylens), outputs=output, name="net", dynamic=True)
        #print(self.model.summary())

        # last layer for output
        self.final_layer = layers.Dense(self.num_class)

        # some temp function
        self.random_num = tf.random_uniform_initializer(0, 1)

    def call(self, samples, training: bool = None):
        xs = samples["input"] # [B, T, F, C=1]
        xlens = samples['input_length'] # [B]
        ys = samples['output'] # [B, T]
        ylens = samples['output_length'] # [B]
        #print('model call xs', xs.shape)
        #print('model call xlens', xlens)
        #print('model call ys', ys)
        #print('model call ylens', ylens)

        loss, observation = self.las(
            tf.squeeze(xs, axis=-1), #[B, T, F]
            xlens,
            ys,
            ylens,
            training=training,
        )
        return loss, observation

    def get_loss(self, outputs, *args, **kwargs):
        ''' must be called after `call` '''
        loss, observation = outputs
        return loss, observation 

    def time_propagate(self, history_logits, history_predictions, step, enc_outputs):
        """
        Args:
            history_logits: the logits of history from 0 to time_step, type: TensorArray
            history_predictions: the predictions of history from 0 to time_step,
                type: TensorArray
            step: current step
            enc_outputs: encoder outputs and its corresponding memory mask, type: tuple
        Returns::

            logits: new logits
            history_logits: the logits array with new logits
            step: next step
        """
        # merge
        (encoder_output, memory_mask) = enc_outputs
        step = step + 1
        output_mask = generate_square_subsequent_mask(step)
        # propagate 1 step
        logits = self.y_net(tf.transpose(history_predictions.stack()), training=False)
        logits = self.las.decoder(
            logits,
            encoder_output,
            tgt_mask=output_mask,
            memory_mask=memory_mask,
            training=False,
        )
        logits = self.final_layer(logits)
        logits = logits[:, -1, :]
        history_logits = history_logits.write(step - 1, logits)
        return logits, history_logits, step

    def decode(self, samples, hparams, decoder, return_encoder=False):
        """ beam search decoding

        Args:
            samples: the data source to be decoded
            hparams: decoding configs are included here
            decoder: it contains the main decoding operations
            return_encoder: if it is True,
                encoder_output and input_mask will be returned
        Returns::

            predictions: the corresponding decoding results
                shape: [batch_size, seq_length]
                it will be returned only if return_encoder is False
            encoder_output: the encoder output computed in decode mode
                shape: [batch_size, seq_length, hsize]
            input_mask: it is masked by input length
                shape: [batch_size, 1, 1, seq_length]
                encoder_output and input_mask will be returned
                only if return_encoder is True
        """
        x0 = samples["input"]
        batch = tf.shape(x0)[0]
        x = self.x_net(x0, training=False)
        input_length = self.compute_logit_length(samples)
        input_mask, _ = create_multihead_mask(x, input_length, None)
        encoder_output = self.las.encoder(x, input_mask, training=False)
        if return_encoder:
            return encoder_output, input_mask
        # init op
        last_predictions = tf.ones([batch], dtype=tf.int32) * self.sos
        history_predictions = tf.TensorArray(
            tf.int32, size=1, dynamic_size=True, clear_after_read=False
        )
        step = 0
        history_predictions.write(0, last_predictions)
        history_predictions = history_predictions.stack()
        init_cand_states = [history_predictions]

        predictions = decoder(
            history_predictions, init_cand_states, step, (encoder_output, input_mask)
        )
        return predictions

    def restore_from_pretrained_model(self, pretrained_model, model_type=""):
        if model_type == "":
            return
        if model_type == "mpc":
            logging.info("loading from pretrained mpc model")
            self.x_net = pretrained_model.x_net
            self.las.encoder = pretrained_model.encoder
        elif model_type == "SpeechTransformer":
            logging.info("loading from pretrained SpeechTransformer model")
            self.x_net = pretrained_model.x_net
            self.y_net = pretrained_model.y_net
            self.las = pretrained_model.las
            self.final_layer = pretrained_model.final_layer
        else:
            raise ValueError("NOT SUPPORTED")


    def deploy(self):
        """ deployment function """
        layers = tf.keras.layers
        input_samples = {
            "input": layers.Input(shape=self.data_descriptions.sample_shape["input"],
                                    dtype=tf.float32, name="deploy_encoder_input_seq"),
            "input_length": layers.Input(shape=self.data_descriptions.sample_shape["input_length"],
                                    dtype=tf.int32, name="deploy_encoder_input_length")
        }
        x = self.x_net(input_samples["input"], training=False)
        input_length = self.compute_logit_length(input_samples)
        input_mask, _ = create_multihead_mask(x, input_length, None)
        encoder_output = self.las.encoder(x, input_mask, training=False)
        self.deploy_encoder = tf.keras.Model(inputs=[input_samples["input"],
                                                     input_samples["input_length"]],
                                             outputs=[encoder_output, input_mask],
                                             name="deploy_encoder_model")
        print(self.deploy_encoder.summary())


        decoder_encoder_output = layers.Input(shape=tf.TensorShape([None, self.hparams.d_model]),
                                        dtype=tf.float32, name="deploy_decoder_encoder_output")
        memory_mask = layers.Input(shape=tf.TensorShape([None, None, None]),
                                        dtype=tf.float32, name="deploy_decoder_memory_mask")
        step = layers.Input(shape=tf.TensorShape([]), dtype=tf.int32, name="deploy_decoder_step")
        history_predictions = layers.Input(shape=tf.TensorShape([None]),
                                        dtype=tf.float32, name="deploy_decoder_history_predictions")

        # propagate one step
        output_mask = generate_square_subsequent_mask(step[0])
        logits = self.y_net(history_predictions, training=False)
        logits = self.las.decoder(
            logits,
            decoder_encoder_output,
            tgt_mask=output_mask,
            memory_mask=memory_mask,
            training=False,
        )
        logits = self.final_layer(logits)
        logits = logits[:, -1, :]
        self.deploy_decoder = tf.keras.Model(inputs=[decoder_encoder_output,
                                                     memory_mask,
                                                     step,
                                                     history_predictions],
                                             outputs=[logits],
                                             name="deploy_decoder_model")
        print(self.deploy_decoder.summary())

    def inference_one_step(self, enc_outputs, cur_input, inner_packed_states_array):
        """call back function for WFST decoder

        Args:
          enc_outputs: outputs and mask of encoder
          cur_input: input sequence for transformer, type: list
          inner_packed_states_array: inner states need to be record, type: tuple
        Returns::

          scores: log scores for all labels
          inner_packed_states_array: inner states for next iterator
        """
        (encoder_output, memory_mask) = enc_outputs
        batch_size = len(cur_input)
        encoder_output = tf.tile(encoder_output, [batch_size, 1, 1])
        memory_mask = tf.tile(memory_mask, [batch_size, 1, 1, 1])
        assert batch_size == len(inner_packed_states_array)
        (step,) = inner_packed_states_array[0]
        step += 1
        output_mask = generate_square_subsequent_mask(step)
        cur_input = tf.constant(cur_input, dtype=tf.float32)
        cur_input = self.y_net(cur_input, training=False)
        logits = self.las.decoder(cur_input, encoder_output, tgt_mask=output_mask,
                                          memory_mask=memory_mask, training=False)
        logits = self.final_layer(logits)
        logits = logits[:, -1, :]
        Z = tf.reduce_logsumexp(logits, axis=(1,), keepdims=True)
        logprobs = logits - Z
        return logprobs.numpy(), [(step,) for _ in range(batch_size)]

