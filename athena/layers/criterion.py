import math
import numpy as np
import tensorflow as tf

def cross_entropy_lsm(logits, ys, lsm_prob, ignore_index, training, normalize_length=False):
    """Compute cross entropy loss for label smoothing of sequence-to-sequence models.
    Args:
        logits (FloatTensor): `[B, T, vocab]`
        ys (LongTensor): Indices of labels. `[B, L]`
        lsm_prob (float): label smoothing probability
        ignore_index (int): index for padding
        normalize_length (bool): normalize XE loss by target sequence length
    Returns:
        loss_mean (FloatTensor): `[1]`
        ppl (float): perplexity
    """ 
    bs, _, vocab = tf.shape(logits)
    ys = tf.reshape(ys, [-1])
    mask = tf.cast((ys != ignore_index), tf.float32)
    logits = tf.reshape(logits, [-1, vocab])

    if lsm_prob == 0 or not training:
        loss = tf.keras.backend.sparse_categorical_crossentropy(ys, logits, from_logits=True)
        loss *= mask
        loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
        ppl = tf.math.exp(loss)
        if not normalize_length:
            loss *= tf.reduce_sum(mask) / tf.cast(bs, tf.float32)
    else:
        target_dist = tf.one_hot(ys, vocab, on_value=(1-lsm_prob), off_value=(lsm_prob/(tf.cast(vocab, tf.float32)-1)))
        target_dist = target_dist * tf.expand_dims(mask, -1)
        target_dist = tf.stop_gradient(target_dist)
        
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        log_probs *= tf.expand_dims(mask, -1)
        loss_sum = - tf.reduce_sum(target_dist * log_probs, -1)
        n_tokens = tf.reduce_sum(mask)
        denom = n_tokens if normalize_length else bs
        loss = tf.reduce_sum(loss_sum) / tf.cast(denom, loss_sum.dtype)

        ppl = tf.math.exp(loss) if normalize_length else tf.math.exp(loss * tf.cast(bs, loss.dtype) / n_tokens)
        
    return loss, ppl


if __name__ == '__main__':
    tf.random.set_seed(10)
    ys = np.array([[1,2,3,4,0,0], [2,3,4,5,5,6]])
    ys = tf.convert_to_tensor(ys)
    logits = tf.random.normal((2, 6, 100))
    loss, ppl = cross_entropy_lsm(logits, ys, 0, 0, True, normalize_length=False)
    print('loss', loss, 'ppl', ppl)
    loss, ppl = cross_entropy_lsm(logits, ys, 0, 0, False, normalize_length=False)
    print('loss', loss, 'ppl', ppl)
    loss, ppl = cross_entropy_lsm(logits, ys, 0, 0, True, normalize_length=True)
    print('loss', loss, 'ppl', ppl)
    loss, ppl = cross_entropy_lsm(logits, ys, 0, 0, False, normalize_length=True)
    print('loss', loss, 'ppl', ppl)

    loss, ppl = cross_entropy_lsm(logits, ys, 0.2, 0, True, normalize_length=False)
    print('loss', loss, 'ppl', ppl)
    loss, ppl = cross_entropy_lsm(logits, ys, 0.2, 0, False, normalize_length=False)
    print('loss', loss, 'ppl', ppl)
    loss, ppl = cross_entropy_lsm(logits, ys, 0.2, 0, True, normalize_length=True)
    print('loss', loss, 'ppl', ppl)
    loss, ppl = cross_entropy_lsm(logits, ys, 0.2, 0, False, normalize_length=True)
    print('loss', loss, 'ppl', ppl)
