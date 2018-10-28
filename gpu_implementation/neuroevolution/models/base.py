__copyright__ = """
Copyright (c) 2018 Uber Technologies, Inc.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import tensorflow as tf
import torch.nn as nn
import torch
import numpy as np
import math
import tabular_logger as tlogger
from gym_tensorflow.ops import indexed_matmul
import logging
from tensorflow.python.ops import random_ops


logger = logging.getLogger(__name__)
fh = logging.FileHandler('./logger.out')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger.setLevel(level=logging.INFO)


class Net(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(Net, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, padding=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)


def normal(shape, scale=0.05, name=None):
    return np.random.normal(loc=0.0, scale=scale, size=shape)


def get_fans(shape):
    # if len shape == 2 mean fc connection
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[:-1])
    fan_out = shape[1] if len(shape) == 2 else shape[-1]
    return fan_in, fan_out


def he_normal(shape, name=None):
    ''' Reference:  He et al., http://arxiv.org/abs/1502.01852
    '''
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(2. / fan_in)
    return normal(shape, s, name=name)


def calculate_gain(nonlinearity, param=None):
    r"""Return the recommended gain value for the given nonlinearity function.
    The values are as follows:

    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Conv{1,2,3}D      :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`\frac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    Leaky Relu        :math:`\sqrt{\frac{2}{1 + \text{negative_slope}^2}}`
    ================= ====================================================

    Args:
        nonlinearity: the non-linear function (`nn.functional` name)
        param: optional parameter for the non-linear function

    Examples:
        >>> gain = nn.init.calculate_gain('leaky_relu')
    """
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))



def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.ndimension()
    print("dimension:",dimensions)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with less than 2 dimensions")

    if dimensions == 2:  # Linear
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:
        num_input_fmaps = tensor.size(1)
        print("num_input_fmaps:",num_input_fmaps)
        num_output_fmaps = tensor.size(0)
        print("num_output_fmaps:",num_output_fmaps)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    r"""Fills the input `Tensor` with values according to the method
    described in "Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification" - He, K. et al. (2015), using a
    normal distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std})` where

    .. math::
        \text{std} = \sqrt{\frac{2}{(1 + a^2) \times \text{fan_in}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the negative slope of the rectifier used after this layer (0 for ReLU
            by default)
        mode: either 'fan_in' (default) or 'fan_out'. Choosing `fan_in`
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing `fan_out` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with 'relu' or 'leaky_relu' (default).

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
    """
    fan = _calculate_correct_fan(tensor, mode)
    print()
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    print("std in kaiming", std)
    with torch.no_grad():
        return tensor.normal_(0, std)


class BaseModel(object):
    def __init__(self):
        self.nonlin = tf.nn.relu
        self.scope = None
        self.load_op = None
        self.indices = None
        self.variables = []
        self.description = ""

    @property
    def requires_ref_batch(self):
        return False

    def create_variable(self, name, shape, scale_by):
        # logger.debug("come create_variable````````````````")
        # logger.debug("in create_variable,shape:{0},scale_by:{1},batch_size:{2}".
        #              format(shape, scale_by, self.batch_size))

        var = tf.get_variable(name, (self.batch_size, ) + shape, trainable=False)
        if not hasattr(var, 'scale_by'):
            var.scale_by = scale_by
            self.variables.append(var)
            # logger.debug("in create_variable,var:{0},self.variables:{1}".format(var, self.variables))
        return var

    def create_weight_variable(self, name, shape, std):
        # logger.debug("in create_weight_variable==========,shape:{0},name:{1},std:{2}".
        #              format(shape, name, std))
        factor = (shape[-2] + shape[-1]) * np.prod(shape[:-2]) / 2
        scale_by = std * np.sqrt(factor)
        # logger.debug("in create_weight_variable,factor:{0},scale_by:{1}".
        #              format(factor, scale_by))
        return self.create_variable(name, shape, scale_by)

    def create_bias_variable(self, name, shape):
        # logger.debug("come create_bias_variable~~~~~~~~~~~~~~~~~")
        return self.create_variable(name, shape, 0.0)

    def conv(self, x, kernel_size, num_outputs, name, stride=1, padding="SAME", bias=True, std=1.0):
        assert len(x.get_shape()) == 5 # Policies x Batch x Height x Width x Feature
        logger.debug("in conv, x.shape".format(x.get_shape))
        with tf.variable_scope(name):
            # logger.debug("in base conv88888888888")
            w = self.create_weight_variable('w', std=std,
                                            shape=(kernel_size, kernel_size, int(x.get_shape()[-1].value), num_outputs))
            logger.debug("in conv,w:{}".format(w))
            w = tf.reshape(w, [-1, kernel_size *kernel_size * int(x.get_shape()[-1].value), num_outputs])

            x_reshape = tf.reshape(x, (-1, x.get_shape()[2], x.get_shape()[3], x.get_shape()[4]))
            patches = tf.extract_image_patches(x_reshape, [1, kernel_size, kernel_size, 1], [1, stride, stride, 1], rates=[1, 1, 1, 1], padding=padding)
            final_shape = (tf.shape(x)[0], tf.shape(x)[1], patches.get_shape()[1].value, patches.get_shape()[2].value, num_outputs)
            patches = tf.reshape(patches, [tf.shape(x)[0],
                                           -1,
                                           kernel_size * kernel_size * x.get_shape()[-1].value])

            if self.indices is None:
                ret = tf.matmul(patches, w)
            else:
                ret = indexed_matmul(patches, w, self.indices)
            ret = tf.reshape(ret, final_shape)
            self.description += "Convolution layer {} with input shape {} and output shape {}\n".format(name, x.get_shape(), ret.get_shape())

            if bias:
                b = self.create_bias_variable('b', (1, 1, 1, num_outputs))
                if self.indices is not None:
                    b = tf.gather(b, self.indices)
                ret = ret + b
            return ret

    def dense(self, x, size, name, bias=True, std=1.0):
        with tf.variable_scope(name):
            logger.debug("in dense function------========")
            w = self.create_weight_variable('w', std=std, shape=(x.get_shape()[-1].value, size))
            if self.indices is None:
                ret = tf.matmul(x, w)
                logger.debug("in dense,indices is None,indices:{0},ret:{1}".format(self.indices, ret))
            else:
                ret = indexed_matmul(x, w, self.indices)
                logger.debug("in dense,indices is not None,indices:{0},ret:{1}".format(self.indices, ret))
            self.description += "Dense layer {} with input shape {} and output shape {}\n".format(name, x.get_shape(), ret.get_shape())
            if bias:
                b = self.create_bias_variable('b', (1, size, ))
                if self.indices is not None:
                    b = tf.gather(b, self.indices)

                return ret + b
            else:
                return ret

    def flattenallbut0(self, x):
        logger.debug("in flattenallbut0, tf.shape(x)[1]:{0},np.prod(x.get_shape()[2:]:{1}，x.get_shape():{2}".
                     format(tf.shape(x)[1], np.prod(x.get_shape()[2:]), x.get_shape()))
        logger.debug("in flattenallbut0, x:{}".format(x))

        return tf.reshape(x, [-1, tf.shape(x)[1], np.prod(x.get_shape()[2:])])

    def make_net(self, x, num_actions, indices=None, batch_size=1, ref_batch=None):
        with tf.variable_scope('Model') as scope:
            logger.debug("in make_net, why++++++++")
            self.description = "Input shape: {}. Number of actions: {}\n".format(x.get_shape(), num_actions)
            self.scope = scope
            self.num_actions = num_actions
            self.ref_batch = ref_batch
            assert self.requires_ref_batch == False or self.ref_batch is not None
            self.batch_size = batch_size
            self.indices = indices
            self.graph = tf.get_default_graph()
            a = self._make_net(x, num_actions)
            logger.debug("in make_net, a:{}".format(a))
            return tf.reshape(a, (-1, num_actions))

    def _make_net(self, x, num_actions):
        logger.debug("why come here")
        raise NotImplementedError()

    def initialize(self):
        self.make_weights()

    def randomize(self, rs, noise):
        logger.debug("randomize:rs:{0},noise:{1}".format(rs, noise))
        seeds = (noise.sample_index(rs, self.num_params), )
        logger.debug("randomnize:seeds:{0}".format(seeds))
        return self.compute_weights_from_seeds(noise, seeds), seeds

    def compute_weights_from_seeds(self, noise, seeds, cache=None):
        if cache:
            # logger.debug("in compute_weights_from_seeds,debug come cache12121212")
            cache_seeds = [o[1] for o in cache]
            if seeds in cache_seeds:
                return cache[cache_seeds.index(seeds)][0]
            elif seeds[:-1] in cache_seeds:
                theta = cache[cache_seeds.index(seeds[:-1])][0]
                return self.compute_mutation(noise, theta, *seeds[-1])
            elif len(seeds) == 1:
                return self.compute_weights_from_seeds(noise, seeds)
            else:
                raise NotImplementedError()
        else:
            idx = seeds[0]
            logger.debug("in compute_weights_from_seeds idx:{0},self.scale_by:{1},len of scale_by:{2}".
                         format(idx, self.scale_by, len(self.scale_by)))
            theta = noise.get(idx, self.num_params).copy() * self.scale_by
            logger.debug("in compute_weights_from_seeds,theta:{}".format(theta))

            for mutation in seeds[1:]:
                idx, power = mutation
                logger.debug("come mutation")
                theta = self.compute_mutation(noise, theta, idx, power)
            return theta

    def mutate(self, parent, rs, noise, mutation_power):
        parent_theta, parent_seeds = parent
        idx = noise.sample_index(rs, self.num_params)
        seeds = parent_seeds + ((idx, mutation_power), )
        theta = self.compute_mutation(noise, parent_theta, idx, mutation_power)
        return theta, seeds

    def compute_mutation(self, noise, parent_theta, idx, mutation_power):
        return parent_theta + mutation_power * noise.get(idx, self.num_params)

    def load(self, sess, i, theta, seeds):
        # logger.debug("come in load,theta:{}".format(theta))
        if self.seeds[i] == seeds:
            # logger.debug("in load, return false!!!!!!!!!!!!!~~~~~~~~~~~~")
            return False
        sess.run(self.load_op, {self.theta: theta, self.theta_idx: i})
        # logger.debug("in load,theta:{0},self.theta_idx:{1},i:{2}".
        #              format(theta, self.theta_idx, i))
        self.seeds[i] = seeds
        return True

    def make_weights(self):
        self.num_params = 0
        self.num_params_test = 0
        self.batch_size = 0
        self.batch_size_test = 0
        self.scale_by = []
        self.scale_by_test = []
        logger.debug("in make_weight````````````````````````")
        shapes = []
        ran_num = np.random.randint(1, 2)
        shape_out = [v.value for v in self.variables[-1].get_shape()][-1]

        # logger.debug("in make_weight87~~~~~~~~~~_____, shape_out:{}".format(shape_out))
        # if ran_num == 0:
        torch.manual_seed(123)
        net = Net((4, 84, 84), shape_out)
        for p in net.parameters():
            # logger
            logger.debug("in make_weights:.data.size{}".format(np.prod(p.data.size())))
            if len(torch.tensor(p.data.size()).numpy()) == 10:
                logger.debug("p in make_weights:{}".format(p))
            self.num_params += np.prod(p.data.size())
            self.scale_by.append(p.data.numpy().flatten().copy())
        self.batch_size = [v.value for v in self.variables[-1].get_shape()][0]
        self.scale_by = np.concatenate(self.scale_by)
        logger.debug("in make weight, heming init batch_size_test:{0},shape_out:{1}".format(self.batch_size_test, shape_out))
        # else:
        # for var in self.variables:
        #     shape = [v.value for v in var.get_shape()]
        #     shapes.append(shape)
        #     logger.debug("in make_weights,shape:{0},np.prod shape[1:]:{1}".format(shape, np.prod(shape[1:])))
        #     self.num_params += np.prod(shape[1:])
        #     parameters = var.scale_by * np.ones(np.prod(shape[1:]), dtype=np.float32)
        #     # logger.debug("in make_weights, not he init parameters:{}".format(parameters))
        #     # logger.debug("in make_weights, parameters:{}".format(parameters))
        #     logger.debug("in make_weights, var.scale:{0},var:{1}".format(var.scale_by, var))
        #     # self.scale_by.append(var.scale_by * np.ones(np.prod(shape[1:]), dtype=np.float32))
        #     self.scale_by.append(parameters)
        #     self.batch_size = shape[0]
        self.seeds = [None] * self.batch_size
        # self.scale_by = np.concatenate(self.scale_by)
        logger.debug("in make_weight, self.num_params:{0},len of self.scale_by:{1}, self.scale_by:{2}".
                     format(self.num_params, len(self.scale_by), self.scale_by[-200:]))
        # logger.debug("in make_weight, self.num_params_test:{0},len of self.scale_by_test:{1}, self.scale_by_test:{2}".
        #              format(self.num_params_test, len(self.scale_by_test), self.scale_by_test[-200:]))
        assert self.scale_by.size == self.num_params
        # Make commit op
        # assigns = []
        self.theta = tf.placeholder(tf.float32, [self.num_params])
        self.theta_idx = tf.placeholder(tf.int32, ())
        offset = 0
        assigns = []
        # reshape
        logger.debug("in make_weight, self.theta:{0},self.theta_idx:{1}".format(self.theta, self.theta_idx))
        for (shape, v) in zip(shapes, self.variables):
            size = np.prod(shape[1:])
            logger.debug("in make_weight, before reshape shape:{0},v:{1}".format(shape, v))
            assigns.append(tf.scatter_update(v, self.theta_idx, tf.reshape(self.theta[offset:offset+size], shape[1:])))
            offset += size
        self.load_op = tf.group(* assigns)
        self.description += "Number of parameteres: {}".format(self.num_params)
