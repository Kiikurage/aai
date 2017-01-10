import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import reporter
import numpy as np

import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reversi import board


class ValueNet(chainer.Chain):
    """Supervised learning policy network"""

    def __init__(self, density=1, channel=6, use_bn=True):
        """
        黒[0,1] | 白[0,1] | valid[0, 1] | 置いた場合にひっくり返る個数 | ターン数 | 手番

        :param density: 1
        :param channel: 6
        """

        self.use_bn = use_bn
        self.train = True
        layers = dict()

        layers['conv1'] = L.Convolution2D(channel, 8 * density, 1, stride=1, pad=0)
        layers['conv2'] = L.Convolution2D(8 * density, 16 * density, 3, stride=1, pad=1)
        layers['conv3'] = L.Convolution2D(16 * density, 32 * density, 3, stride=1, pad=1)
        layers['conv4'] = L.Convolution2D(32 * density, 32 * density, 3, stride=1, pad=1)

        if self.use_bn:
            layers['norm2'] = L.BatchNormalization(16 * density)
            layers['norm3'] = L.BatchNormalization(32 * density)
            layers['norm4'] = L.BatchNormalization(32 * density)
            layers['norm5'] = L.BatchNormalization(1024 * density)

        layers['linear1'] = L.Linear(2048 * density, 1024 * density)
        layers['linear2'] = L.Linear(1024 * density, 1024 * density)
        layers['linear3'] = L.Linear(1024 * density, 41)

        super(ValueNet, self).__init__(**layers)

    # noinspection PyUnresolvedReferences,PyCallingNonCallable
    def predict(self, x, train=True):
        if self.use_bn:
            h1 = F.relu(self.conv1(x))
            h2 = F.relu(self.norm2(self.conv2(h1), test=not train))
            h3 = F.relu(self.norm3(self.conv3(h2), test=not train))
            h4 = F.relu(self.norm4(self.conv4(h3), test=not train))
            h5 = F.relu(self.norm5(self.linear1(h4), test=not train))
            h6 = F.relu(self.linear2(h5))
            score = self.linear3(h6)

            return score
        else:
            h1 = F.relu(self.conv1(x))
            h2 = F.relu(self.conv2(h1))
            h3 = F.relu(self.conv3(h2))
            h4 = F.relu(self.conv4(h3))
            h5 = F.relu(self.linear1(h4))
            h6 = F.relu(self.linear2(h5))
            score = self.linear3(h6)

            return score

    # noinspection PyCallingNonCallable
    def __call__(self, x, t, train=True):
        pred_score = self.predict(x, train)
        self.loss = F.softmax_cross_entropy(pred_score, t)
        self.accuracy = F.accuracy(pred_score, t)

        return self.loss