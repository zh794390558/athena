import math
import logging
import tensorflow as tf

logger = logging.getLogger(__name__)

def init_with_uniform(var, param_init):
    ''' initialize with uniform distribution.
    Args:
        var (tf.Variable) : variable
        param_init (float) : 
    '''
    if var is None:
        return

    if var.shape.rank == 1:
        var.assign(tf.zeros(var.shape)) # bias
        logger.info(f"Initialize {var.name} {var.shape} {var.dtype} with constant / 0.0")
    elif var.shape.rank in [2,3,4]:
        var.assign(tf.random.uniform(var.shape, -param_init, param_init))
        logger.info(f"Initialize {var.name} {var.shape} {var.dtype} with uniform / {param_init:.3f}")
    else:
        raise ValueError(var)


def init_with_lecun_normal(var, param_init):
    ''' initialize with Lecun style.
    Args:
        var (tf.Variable) : variable
        param_init (float) : 
    '''
    if var is None:
        return

    if var.shape.rank == 1:
        var.assign(tf.zeros(var.shape)) # bias
        logger.info(f"Initialize {var.name} {var.shape} {var.dtype} with constant / 0.0")
    elif var.shape.rank == 2:
        fan_in = var.shape[1]
        var.assign(tf.random.normal(var.shape, mean=0., stddev=1./math.sqrt(fan_in))) # linear weight 
        logger.info(f"Initialize {var.name} {var.shape} {var.dtype} with lecun / {param_init:.3f}")
    elif var.shape.rank == 3:
        s = var.shape
        fan_in = s[1] * s[2]
        var.assign(tf.random.normal(var.shape, mean=0., stddev=1./math.sqrt(fan_in))) # 1d conv weight 
        logger.info(f"Initialize {var.name} {var.shape} {var.dtype} with lecun / {param_init:.3f}")
    elif var.shape.rank == 4:
        s = var.shape
        fan_in = s[1] * s[2]
        var.assign(tf.random.normal(var.shape, mean=0., stddev=1./math.sqrt(fan_in))) # 2d conv weight 
        logger.info(f"Initialize {var.name} {var.shape} {var.dtype} with lecun / {param_init:.3f}")
    else:
        raise ValueError(var)


def init_with_xavier_uniform(var, param_init):
    ''' initialize with Xavier unifrom distribution.
    Args:
        var (tf.Variable) : variable
        param_init (float) : 
    '''
    if var is None:
        return

    if var.shape.rank == 1:
        var.assign(tf.zeros(var.shape)) # bias
        logger.info(f"Initialize {var.name} {var.shape} {var.dtype} with constant / 0.0")
    elif var.shape.rank in [2, 3, 4]:
        val = tf.keras.initializers.glorot_uniform()(var.shape)
        var.assign(val)
        logger.info(f"Initialize {var.name} {var.shape} {var.dtype} with xavier_uniform / {param_init:.3f}")
    else:
        raise ValueError(var)



def init_like_transformer_xl(var, param_init):
    ''' initialize with TransformerXL.
    Args:
        var (tf.Variable) : variable
        param_init (float) : 
    '''
    if var is None:
        return

    if 'norm' in var.name and 'weight' in var.name:
        assert var.shape.rank == 1
        var.assign(tf.random.normal(var.shape, mean=1., stddev=param_init)) # 2d conv weight 
        logger.info(f"Initialize {var.name} {var.shape} {var.dtype} with normal / {param_init:.3f}")
    elif var.shape.rank == 1:
        var.assign(tf.zeros(var.shape)) # bias
        logger.info(f"Initialize {var.name} {var.shape} {var.dtype} with constant / 0.0")
    elif var.shape.rank == 2:
        val = tf.keras.initializers.glorot_uniform()(var.shape)
        var.assign(val)
        logger.info(f"Initialize {var.name} {var.shape} {var.dtype} with xavier_uniform / {param_init:.3f}")
    else:
        raise ValueError(var)
