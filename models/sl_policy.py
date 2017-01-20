import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import reporter
import numpy as np

import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reversi import board


def batch_softmax(x, T):
    y = x - x.max(axis=1, keepdims=True)
    y = np.exp(y / T)
    y /= y.sum(axis=1, keepdims=True)
    return y


def softmax(x, mask, T=1.):
    y = x - x.max()
    y = np.exp(y / T) * mask
    y /= y.sum()
    return y


class SLPolicy(chainer.Chain):
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
        layers['linear3'] = L.Linear(1024 * density, 64)

        super(SLPolicy, self).__init__(**layers)

    # noinspection PyUnresolvedReferences,PyCallingNonCallable
    def predict(self, x, train=True):
        if self.use_bn:
            h1 = F.relu(self.conv1(x))
            h2 = F.relu(self.norm2(self.conv2(h1), test=not train))
            h3 = F.relu(self.norm3(self.conv3(h2), test=not train))
            h4 = F.relu(self.norm4(self.conv4(h3), test=not train))
            h5 = F.relu(self.norm5(self.linear1(h4), test=not train))
            h6 = F.relu(self.linear2(h5))
            scores = self.linear3(h6)

            return scores
        else:
            h1 = F.relu(self.conv1(x))
            h2 = F.relu(self.conv2(h1))
            h3 = F.relu(self.conv3(h2))
            h4 = F.relu(self.conv4(h3))
            h5 = F.relu(self.linear1(h4))
            h6 = F.relu(self.linear2(h5))
            scores = self.linear3(h6)

            return scores

    # noinspection PyCallingNonCallable
    def __call__(self, x, ply, train=True):
        scores = self.predict(x, train)
        self.loss = F.softmax_cross_entropy(scores, ply)
        # reporter.report({'loss': self.loss}, self)

        self.accuracy = F.accuracy(scores, ply)
        # reporter.report({'accuracy': self.accuracy}, self)
        if train:
            return self.loss
        else:
            return np.argmax(cuda.to_cpu(scores.data), axis=1)

    # noinspection PyCallingNonCallable
    def act(self, b, color, turn, temperature=1.):
        state = board.to_state(b, color, turn)
        x = chainer.Variable(self.xp.array([state], 'float32'), volatile=True)
        scores = self.predict(x, False)
        pred = softmax(cuda.to_cpu(scores.data[0]), mask=state[2].ravel(), T=temperature)
        action = np.random.choice(64, p=pred)

        return action
