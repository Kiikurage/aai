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


class RolloutPolicy(chainer.Chain):
    """Supervised learning policy network"""

    def __init__(self, density=1, channel=6, use_bn=True):
        """
        黒[0,1] | 白[0,1] | valid[0, 1] | 置いた場合にひっくり返る個数 | ターン数 | 手番

        :param density: 1
        :param channel: 6
        """

        layers = dict()

        layers['conv1'] = L.Convolution2D(channel, 8 * density, 1, stride=1, pad=0)
        layers['linear1'] = L.Linear(512 * density, 64)

        super(RolloutPolicy, self).__init__(**layers)

    # noinspection PyUnresolvedReferences,PyCallingNonCallable
    def predict(self, x, train=True):

        h1 = F.relu(self.conv1(x))
        scores = self.linear1(h1)
        return scores

    # noinspection PyCallingNonCallable
    def __call__(self, x, ply, train=True):
        scores = self.predict(x, train)
        self.loss = F.softmax_cross_entropy(scores, ply)
        self.accuracy = F.accuracy(scores, ply)
        if train:
            return self.loss
        else:
            return np.argmax(cuda.to_cpu(scores.data), axis=1)

    # noinspection PyCallingNonCallable
    def act(self, b, color, turn, temperature=1.):
        state = board.to_state(b, color, turn)
        x = chainer.Variable(self.xp.array([state], 'float32'), volatile=True)
        scores = self.predict(x, False)
        pred = softmax(cuda.to_cpu(scores.data[0]), mask=state[2].ravel(), T=0.5)
        action = np.random.choice(64, p=pred)

        return action
