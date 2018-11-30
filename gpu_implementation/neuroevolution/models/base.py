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
from tensorflow.contrib.layers.python.layers import initializers
import random,time

MAX_SEED = 2**32 - 1

# logger = logging.getLogger(__name__)
# # fh = logging.FileHandler('./logger.out')
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# # fh.setFormatter(formatter)
# # logger.addHandler(fh)
#
# console_handler = logging.StreamHandler()
# console_handler.setFormatter(formatter)
# logger.addHandler(console_handler)
#
# logger.setLevel(level=logging.INFO)


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
            nn.Linear(512, n_actions),
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)


def init_weight(shape, type):
    w = torch.empty(shape)
    if type == "xavier_normal":
        nn.init.xavier_normal_(w)
    if type == "xavier_uniform":
        nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
    if type == "kaiming_uniform":
        nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
    if type == "kaiming_normal":
        nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
    if type == "orthogonal":
        nn.init.orthogonal_(w)

    return w.numpy().flatten()


class BaseModel(object):
    def __init__(self):
        self.nonlin = tf.nn.relu
        self.scope = None
        self.load_op = None
        self.indices = None
        self.variables = []
        self.description = ""
        # self.logger = logger
        # self.count = 0

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
        # logger.debug("in create_weight_variable~~~~~``````,shape:{0},name:{1},std:{2}".
        #              format(shape, name, std))
        factor = (shape[-2] + shape[-1]) * np.prod(shape[:-2]) / 2
        scale_by = std * np.sqrt(factor)
        # logger.debug("in create_weight_variable,factor:{0},scale_by:{1}".
        #              format(factor, scale_by))
        return self.create_variable(name, shape, scale_by)

    def create_bias_variable(self, name, shape):
        return self.create_variable(name, shape, 0.0)

    def conv(self, x, kernel_size, num_outputs, name, stride=1, padding="SAME", bias=True, std=1.0):
        assert len(x.get_shape()) == 5 # Policies x Batch x Height x Width x Feature
        # logger.debug("in conv, x.shape".format(x.get_shape))
        with tf.variable_scope(name):
            # logger.debug("in base conv88888888888")
            w = self.create_weight_variable('w', std=std,
                                            shape=(kernel_size, kernel_size, int(x.get_shape()[-1].value), num_outputs))
            # logger.debug("in conv,w:{}".format(w))
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
            w = self.create_weight_variable('w', std=std, shape=(x.get_shape()[-1].value, size))
            if self.indices is None:
                ret = tf.matmul(x, w)
                # logger.debug("in dense,indices is None,indices:{0},ret:{1}".format(self.indices, ret))
            else:
                ret = indexed_matmul(x, w, self.indices)
                # logger.debug("in dense,indices is not None,indices:{0},ret:{1}".format(self.indices, ret))
            self.description += "Dense layer {} with input shape {} and output shape {}\n".format(name, x.get_shape(), ret.get_shape())
            if bias:
                b = self.create_bias_variable('b', (1, size, ))
                if self.indices is not None:
                    b = tf.gather(b, self.indices)
                return ret + b
            else:
                return ret

    def flattenallbut0(self, x):
        return tf.reshape(x, [-1, tf.shape(x)[1], np.prod(x.get_shape()[2:])])

    def make_net(self, x, num_actions, indices=None, batch_size=1, ref_batch=None):
        with tf.variable_scope('Model') as scope:
            # logger.debug("in make_net, why++++++++")
            self.description = "Input shape: {}. Number of actions: {}\n".format(x.get_shape(), num_actions)
            self.scope = scope
            self.num_actions = num_actions
            self.ref_batch = ref_batch
            assert self.requires_ref_batch == False or self.ref_batch is not None
            self.batch_size = batch_size
            self.indices = indices
            self.graph = tf.get_default_graph()
            a = self._make_net(x, num_actions)
            # logger.debug("in make_net, a:{}".format(a))
            return tf.reshape(a, (-1, num_actions))

    def _make_net(self, x, num_actions):
        # logger.debug("why come here")
        raise NotImplementedError()

    def initialize(self):
        self.make_weights()

    def randomize(self, rs, noise):
        # logger.debug("randomize:rs:{0},noise:{1}".format(rs, noise))
        seeds = (noise.sample_index(rs, self.num_params), )
        return self.compute_weights_from_seeds(noise, seeds), seeds

    def compute_weights_from_seeds(self, noise, seeds, cache=None):
        # self.count = self.count+1
        # logger.error("in compute_weight_from_seeds:len of cache:{0},seeds:{1}".format(len(cache), seeds))
        if cache:
            # logger.error("in compute_weights_from_seeds,cache:{0},seeds:{1}".format(len(cache), seeds))
            cache_seeds = [o[1] for o in cache]
            if seeds in cache_seeds:
                # logger.debug("in compute_weights_from_seeds,cache,fisrt!!!!")
                return cache[cache_seeds.index(seeds)][0]
            elif seeds[:-1] in cache_seeds:
                # logger.debug("in compute_weights_from_seeds,cache,second!!!!")
                theta = cache[cache_seeds.index(seeds[:-1])][0]
                return self.compute_mutation(noise, theta, *seeds[-1])
            elif len(seeds) == 1:
                # logger.debug("in compute_weights_from_seeds,cache,third!!!!")
                print("why come here")
                print("cache:", cache)
                return self.compute_weights_from_seeds(noise, seeds)
            else:
                raise NotImplementedError()
        else:
            idx = seeds[0]
            theta = noise.get(idx, self.num_params).copy() * self.scale_by  # self.scale_by
            # logger.debug("in compute_weights_from_seeds,ran_num:{0},theta[-100:]:{1}".format(ran_num, theta[-100:]))
            for mutation in seeds[1:]:
                idx, power = mutation
                theta = self.compute_mutation(noise, theta, idx, power)
            return theta

    def mutate(self, parent, rs, noise, mutation_power):
        # assert False
        parent_theta, parent_seeds = parent
        idx = noise.sample_index(rs, self.num_params)
        seeds = parent_seeds + ((idx, mutation_power), )
        theta = self.compute_mutation(noise, parent_theta, idx, mutation_power)
        return theta, seeds

    def compute_mutation(self, noise, parent_theta, idx, mutation_power):
        return parent_theta + mutation_power * noise.get(idx, self.num_params)

    def load(self, sess, i, theta, seeds):
        if self.seeds[i] == seeds:
            return False
        sess.run(self.load_op, {self.theta: theta, self.theta_idx: i})
        self.seeds[i] = seeds
        return True

    def make_weights(self):
        self.num_params = 0
        self.num_params_test = 0
        self.batch_size = 0
        self.batch_size_test = 0
        self.scale_by = []
        self.scale_by_test = []
        shapes = []
        # shape_out = [v.value for v in self.variables[-1].get_shape()][-1]

        for var in self.variables:
            shape = [v.value for v in var.get_shape()]
            shapes.append(shape)
            # logger.debug("in make_weights,shape:{0},np.prod shape[1:]:{1}".format(shape, np.prod(shape[1:])))
            self.num_params += np.prod(shape[1:])
            parameters = var.scale_by * np.ones(np.prod(shape[1:]), dtype=np.float32)
            # logger.debug("in make_weights, not he init parameters:{}".format(parameters))
            # logger.debug("in make_weights, parameters:{}".format(parameters))
            # logger.debug("in make_weights, var.scale:{0},var:{1}".format(var.scale_by, var))
            # self.scale_by.append(var.scale_by * np.ones(np.prod(shape[1:]), dtype=np.float32))
            self.scale_by.append(parameters)
            self.batch_size = shape[0]
        self.seeds = [None] * self.batch_size
        self.scale_by = np.concatenate(self.scale_by)
        # logger.debug("in make_weight, self.num_params:{0},len of self.scale_by:{1}, self.scale_by:{2}".
        #              format(self.num_params, len(self.scale_by), self.scale_by[-100:]))
        assert self.scale_by.size == self.num_params
        # Make commit op
        # assigns = []
        self.theta = tf.placeholder(tf.float32, [self.num_params])
        self.theta_idx = tf.placeholder(tf.int32, ())
        offset = 0
        assigns = []
        # reshape
        # logger.debug("in make_weight, self.theta:{0},self.theta_idx:{1}".format(self.theta, self.theta_idx))
        for (shape, v) in zip(shapes, self.variables):
            size = np.prod(shape[1:])
            # logger.debug("in make_weight, before reshape shape:{0},v:{1}".format(shape, v))
            assigns.append(tf.scatter_update(v, self.theta_idx, tf.reshape(self.theta[offset:offset+size], shape[1:])))
            offset += size
        self.load_op = tf.group(* assigns)
        self.description += "Number of parameteres: {}".format(self.num_params)
