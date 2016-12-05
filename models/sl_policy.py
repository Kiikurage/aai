import chainer
import chainer.functions as F
from chainer.functions.loss.vae import gaussian_kl_divergence
import chainer.links as L
from chainer import cuda
import numpy as np


class SL_POLICY(chainer.Chain):
    """Supervised learning policy network"""

    def __init__(self,density=1, size=8, channel=5):
        """
        黒[0,1] | 白[0,1] | empty[0, 1] |置いた場合にひっくり返る個数 | ターン数

        :param density: 1
        :param size: 8
        :param channel: 5?
        """

        super(SL_POLICY, self).__init__(
            conv1=L.Convolution2D(channel, 8 * density, 4, stride=1, pad=1),
            norm1=L.BatchNormalization(8 * density),
            conv2=L.Convolution2D(8 * density, 16 * density, 4, stride=1, pad=1),
            norm2=L.BatchNormalization(16 * density),
            conv3=L.Convolution2D(16 * density, 16 * density, 4, stride=1, pad=1),
            norm3=L.BatchNormalization(16 * density),
            linear1=L.Linear(400*density, 400*density),
            linear2=L.Linear(300*density, 64)
        )

    def predict(self, x, train=True):

        h1 = F.relu(self.norm1(self.conv1(x), test=not train))
        h2 = F.relu(self.norm2(self.conv2(h1), test=not train))
        h3 = F.relu(self.norm3(self.conv3(h2), test=not train))
        h4 = F.relu(self.linear1(h3))
        scores = self.linear(h4)
        return scores

    def __call__(self, x, ply):
        scores = self.predict(x)
        self.loss = F.softmax_cross_entropy(scores, ply)

        return self.loss

    def act(self, x):
        scores = self.predict(x)
        pred = cuda.to_cpu(F.softmax(scores).data)
        action = [np.random.choice(64, p=p) for p in pred]

        return action

