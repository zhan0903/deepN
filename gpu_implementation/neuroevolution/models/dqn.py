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

import numpy as np
import tensorflow as tf
from .base import BaseModel
import logging

logger = logging.getLogger(__name__)
fh = logging.FileHandler('./logger.out')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger.setLevel(level=logging.DEBUG)




class Model(BaseModel):
    def create_weight_variable(self, name, shape, std):
        logger.debug("in create_weight_variable, 99999999999")
        scale_by = std / np.sqrt(np.prod(shape[:-1]))
        logger.debug("in Model, create_weight_variable:{0},shape[:-1]:{1}".format(scale_by, shape[:-1]))
        return self.create_variable(name, shape, scale_by)

    def _make_net(self, x, num_actions):
        x = self.nonlin(self.conv(x, name='conv1', num_outputs=16, kernel_size=8, stride=4))
        x = self.nonlin(self.conv(x, name='conv2', num_outputs=32, kernel_size=4, stride=2))
        x = self.flattenallbut0(x)
        x = self.nonlin(self.dense(x, 256, 'fc'))

        return self.dense(x, num_actions, 'out', std=0.1)


class LargeModel(Model):
    def _make_net(self, x, num_actions):
        logger.debug("in _make_net, come here is right=========")
        x = self.nonlin(self.conv(x, name='conv1', num_outputs=32, kernel_size=8, stride=4, std=1.0))
        x = self.nonlin(self.conv(x, name='conv2', num_outputs=64, kernel_size=4, stride=2, std=1.0))
        x = self.nonlin(self.conv(x, name='conv3', num_outputs=64, kernel_size=3, stride=1, std=1.0))
        x = self.flattenallbut0(x)
        logger.debug("in _make_net, after flatten:x:{}".format(x))
        x = self.nonlin(self.dense(x, 512, 'fc'))

        return self.dense(x, num_actions, 'out', std=0.1)
