import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import reporter
import numpy as np

import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reversi import board


def batch_softmax(x, T=1.0):
    y = x - x.max(axis=1, keepdims=True)
    y = np.exp(y / T)
    y /= y.sum(axis=1, keepdims=True)
    return y


class RolloutValueNet(chainer.Chain):
    """Supervised learning policy network"""

    def __init__(self, density=1, channel=6, use_bn=True, output=41):
        """
        黒[0,1] | 白[0,1] | valid[0, 1] | 置いた場合にひっくり返る個数 | ターン数 | 手番

        :param density: 1
        :param channel: 6
        """

        self.use_bn = use_bn
        self.train = True
        layers = dict()

        layers['conv1'] = L.Convolution2D(channel, 8, 1, stride=1, pad=0)
        layers['conv2'] = L.Convolution2D(8, 16, 3, stride=1, pad=1)
        layers['norm2'] = L.BatchNormalization(16)
        layers['linear3'] = L.Linear(1024, output)

        super(RolloutValueNet, self).__init__(**layers)

    # noinspection PyUnresolvedReferences,PyCallingNonCallable
    def predict(self, x, train=False):
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.norm2(self.conv2(h1), test=not train))
        score = self.linear3(h2)

        return score

    # noinspection PyCallingNonCallable
    def __call__(self, x, t, train=True):
        pred_score = self.predict(x, train)
        self.loss = F.softmax_cross_entropy(pred_score, t)
        self.accuracy = F.accuracy(pred_score, t)

        return self.loss

    def act(self, b, c, t, temperature=1):
        board_batch = []
        valid_ply = []
        for x in range(8):
            for y in range(8):
                if board.is_valid(b, c, x, y):
                    valid_ply.append((x, y))
                    b_ = board.put(b, c, x, y)
                    board_batch.append(board.to_state(b_, 1 - c, t + 1))
        x_batch = chainer.Variable(self.xp.array(np.stack(board_batch), 'float32'), volatile=True)
        scores = self.predict(x_batch, train=False)

        min_score = 64
        best_ply = None
        for ply, score in zip(valid_ply, batch_softmax(cuda.to_cpu(scores.data))):
            print(ply, np.argmax(score) - 20)
            print()
            if np.argmax(score) < min_score:
                min_score = np.argmax(score)
                best_ply = ply
        action = best_ply[0] * 8 + best_ply[1]

        return action
