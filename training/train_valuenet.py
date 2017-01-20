import random
import argparse
import os, sys
import numpy as np
import datetime
import time
import pickle as pkl
import copy
import chainer
from chainer import cuda
from chainer import dataset
from chainer import training, iterators
from chainer.training import extensions
from chainer import serializers

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reversi import board
from models import ValueNet, RolloutValueNet

DATA_DIR = '/mnt/share/aai_fukuta/'
train_path = DATA_DIR + 'train_withscore.pkl'
train_small_path = DATA_DIR + 'train_small_withscore.pkl'
test_path = DATA_DIR + 'test_withscore.pkl'


class PreprocessedDataset(dataset.DatasetMixin):
    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            self.data = pkl.load(f)

    def __len__(self):
        return len(self.data)

    def data_augmentation(self, b, ply):
        """
        :param b: list or np.ndarray (board, shape = [2, 8, 8])
        :param ply: int (next play index)
        :return: board, play (applied augmentation)
        """
        ply_array = np.zeros(b.shape[1:], dtype=int)
        ply_array[ply // 8, ply % 8] = 1

        converted_b = np.array(b)
        converted_ply_array = np.array(ply_array)

        operation = [random.choice([0, 1]) for _ in range(3)]

        if operation[0]:  # flip x
            converted_b = converted_b[:, ::-1, :]
            converted_ply_array = converted_ply_array[::-1, :]
        if operation[1]:  # flip y
            converted_b = converted_b[:, :, ::-1]
            converted_ply_array = converted_ply_array[:, ::-1]
        if operation[2]:  # rotation
            converted_b[0] = np.rot90(converted_b[0])
            converted_b[1] = np.rot90(converted_b[1])
            converted_ply_array = np.rot90(converted_ply_array)

        if ply == -1:
            ret_ply = -1
        else:
            ply_indeces = np.where(converted_ply_array == 1)
            ret_ply = int(ply_indeces[0] * 8 + ply_indeces[1])

        ret_b = converted_b

        return ret_b, ret_ply

    # noinspection PyUnresolvedReferences
    def get_example(self, i, with_aug=True):
        plies, score = self.data[i]

        # 盤面を作る
        # 各チャンネルの詳細は https://github.com/Kiikurage/aai/issues/13 を参照

        # plies = game['plies']
        b = board.init()
        n = random.randint(0, len(plies) - 1)

        for color, ply in plies[:n]:
            if ply == -1:
                continue

            x = ply // 8
            y = ply % 8
            b = board.put(b, color, x, y)

        color, ply = plies[n]

        if with_aug:
            b, ply = self.data_augmentation(b, ply)

        res = board.to_state(b, color, n)
        score = np.clip(score, -40, 40)
        score = (abs(score)+1) // 2 if score >= 0 else -((abs(score)+1)//2)
        score = score if color == 0 else -score
        return res, np.int32(score + 20)


def progress_report(count, start_time, batchsize):
    duration = time.time() - start_time
    throughput = count * batchsize / duration
    sys.stderr.write(
        '\r{} updates ({} samples) time: {} ({:.2f} samples/sec)'.format(
            count, count * batchsize, str(datetime.timedelta(seconds=duration)).split('.')[0], throughput
        )
    )


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU device ID')
    parser.add_argument('--epoch', '-e', type=int, default=50, help='# of epoch')
    parser.add_argument('--batch_size', type=int, default=128, help='size of mini-batch')
    parser.add_argument('--density', type=int, default=1, help='density of cnn kernel')
    parser.add_argument('--small', dest='small', action='store_true', default=False)
    parser.add_argument('--no_bn', dest='use_bn', action='store_false', default=True)
    parser.add_argument('--out', default='')
    parser.set_defaults(test=False)
    args = parser.parse_args()

    model = ValueNet(use_bn=args.use_bn)
    # model = RolloutValueNet(use_bn=args.use_bn, output=41)
    # log directory
    out = datetime.datetime.now().strftime('%m%d')
    if args.out:
        out = out + '_' + args.out
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs_value", out))
    os.makedirs(os.path.join(out_dir, 'models'), exist_ok=True)

    # gpu
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    # setting
    with open(os.path.join(out_dir, 'setting.txt'), 'w') as f:
        for k, v in args._get_kwargs():
            print('{} = {}'.format(k, v))
            f.write('{} = {}\n'.format(k, v))

    # prepare for dataset
    if args.small:
        train = PreprocessedDataset(train_small_path)
    else:
        train = PreprocessedDataset(train_path)
    test = PreprocessedDataset(test_path)
    train_iter = iterators.SerialIterator(train, args.batch_size)
    val_iter = iterators.SerialIterator(test, args.batch_size, repeat=False)

    # optimizer
    optimizer = chainer.optimizers.Adam(eps=1e-2)
    optimizer.setup(model)

    # start training
    start = time.time()
    train_count = 0
    for epoch in range(args.epoch):

        # train
        train_loss = []
        train_accuracy = []

        for i in range(len(train) // args.batch_size):
            batch = train_iter.next()
            x = chainer.Variable(model.xp.array([b[0] for b in batch], 'float32'))
            y = chainer.Variable(model.xp.array([b[1] for b in batch], 'int32'))
            optimizer.update(model, x, y)
            train_count += 1

            progress_report(train_count, start, args.batch_size)

            train_loss.append(cuda.to_cpu(model.loss.data))
            train_accuracy.append(cuda.to_cpu(model.accuracy.data))

        # test
        test_loss = []
        test_accuracy = []

        it = copy.copy(val_iter)
        for batch in it:
            x = chainer.Variable(model.xp.array([b[0] for b in batch], 'float32'), volatile=True)
            y = chainer.Variable(model.xp.array([b[1] for b in batch], 'int32'), volatile=True)
            model(x, y, train=False)

            test_loss.append(cuda.to_cpu(model.loss.data))
            test_accuracy.append(cuda.to_cpu(model.accuracy.data))

        print('\nepoch {}  train_loss {:.5f}  train_accuracy {:.3f} \n'
              '          test_loss {:.5f}  test_accuracy {:.3f}'.
              format(epoch, np.mean(train_loss), np.mean(train_accuracy), np.mean(test_loss), np.mean(test_accuracy)))
        with open(os.path.join(out_dir, "log"), 'a+') as f:
            f.write('epoch {}  train_loss {:.5f}  train_accuracy {:.3f} \n'
                    '          test_loss {:.5f}   test_accuracy {:.3f} \n'.
                    format(epoch, np.mean(train_loss), np.mean(train_accuracy), np.mean(test_loss),
                           np.mean(test_accuracy)))

        if epoch % 5 == 0:
            serializers.save_hdf5(os.path.join(out_dir, "models", "value_net_{}.model".format(epoch)), model)


if __name__ == '__main__':
    main()
