import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import reporter
import numpy as np


def softmax(x, T):
    y = x - x.max(axis=1, keepdims=True)
    y = np.exp(y / T)
    y /= y.sum(axis=1, keepdims=True)
    return y


class SLPolicy(chainer.Chain):
    """Supervised learning policy network"""

    def __init__(self, density=1, channel=6, use_bn=False):
        """
        黒[0,1] | 白[0,1] | valid[0, 1] | 置いた場合にひっくり返る個数 | ターン数 | 手番

        :param density: 1
        :param channel: 6
        """

        self.use_bn = use_bn
        layers = dict()

        layers['conv1'] = L.Convolution2D(channel, 8 * density, 1, stride=1, pad=0)
        layers['conv2'] = L.Convolution2D(8 * density, 16 * density, 3, stride=1, pad=1)
        layers['conv3'] = L.Convolution2D(16 * density, 32 * density, 3, stride=1, pad=1)
        layers['conv4'] = L.Convolution2D(32 * density, 32 * density, 3, stride=1, pad=1)

        if self.use_bn:
            layers['norm1'] = L.BatchNormalization(8 * density)
            layers['norm2'] = L.BatchNormalization(16 * density)
            layers['norm3'] = L.BatchNormalization(32 * density)
            layers['norm4'] = L.BatchNormalization(32 * density)

        layers['linear1'] = L.Linear(2048 * density, 1024 * density)
        layers['linear2'] = L.Linear(1024 * density, 1024 * density)
        layers['linear3'] = L.Linear(1024 * density, 64)

        super(SLPolicy, self).__init__(**layers)

    # noinspection PyUnresolvedReferences,PyCallingNonCallable
    def predict(self, x, train=True):
        if self.use_bn:
            h1 = F.relu(self.norm1(self.conv1(x), test=not train))
            h2 = F.relu(self.norm2(self.conv2(h1), test=not train))
            h3 = F.relu(self.norm3(self.conv3(h2), test=not train))
            h4 = F.relu(self.norm4(self.conv4(h3), test=not train))
            h5 = F.relu(self.linear1(h4))
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
    def __call__(self, x, ply):
        scores = self.predict(x)
        self.loss = F.softmax_cross_entropy(scores, ply)
        reporter.report({'loss': self.loss}, self)

        self.accuracy = F.accuracy(scores, ply)
        reporter.report({'accuracy': self.accuracy}, self)

        return self.loss

    # noinspection PyCallingNonCallable
    def act(self, x, temperature=1):
        scores = self.predict(x)
        pred = softmax(cuda.to_cpu(scores.data), T=temperature)
        action = [np.random.choice(64, p=p) for p in pred]

        return action
