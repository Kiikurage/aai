import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
import numpy as np


def softmax(x, T):
    y = x - x.max(axis=1, keepdims=True)
    y = np.exp(y / T)
    y /= y.sum(axis=1, keepdims=True)
    return y


class RLPolicy(chainer.Chain):
    """Supervised learning policy network"""

    def __init__(self, density=1, size=8, channel=6):
        """
        黒[0,1] | 白[0,1] | empty[0, 1] | 置いた場合にひっくり返る個数 | 手番が黒か否か | ターン数 |

        :param density: 1
        :param size: 8
        :param channel: 5?
        """

        super(RLPolicy, self).__init__(
            conv1=L.Convolution2D(channel, 8 * density, 4, stride=1, pad=1),
            norm1=L.BatchNormalization(8 * density),
            conv2=L.Convolution2D(8 * density, 16 * density, 4, stride=1, pad=1),
            norm2=L.BatchNormalization(16 * density),
            conv3=L.Convolution2D(16 * density, 16 * density, 4, stride=1, pad=1),
            norm3=L.BatchNormalization(16 * density),
            linear1=L.Linear(400 * density, 400 * density),
            linear2=L.Linear(400 * density, 64)
        )

    # noinspection PyUnresolvedReferences,PyCallingNonCallable
    def predict(self, x, train=True):
        h1 = F.relu(self.norm1(self.conv1(x), test=not train))
        h2 = F.relu(self.norm2(self.conv2(h1), test=not train))
        h3 = F.relu(self.norm3(self.conv3(h2), test=not train))
        h4 = F.relu(self.linear1(h3))
        scores = self.linear2(h4)
        return scores

    # noinspection PyCallingNonCallable
    def __call__(self, x, ply):
        scores = self.predict(x)
        self.loss = F.softmax_cross_entropy(scores, ply)

        return self.loss

    # noinspection PyCallingNonCallable
    def act(self, x, temperature=1):
        scores = self.predict(x)
        pred = softmax(cuda.to_cpu(scores.data), T=temperature)
        action = [np.random.choice(64, p=p) for p in pred]

        return action
