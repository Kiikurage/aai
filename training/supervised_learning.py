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
from models import SLPolicy, RolloutPolicy, RLPolicy

# DATA_DIR = '/mnt/share/aai_fukuta/'
DATA_DIR = '/home/mil/fukuta/work_space/iizuka_aai/train_data/'
train_path = DATA_DIR + 'train.pkl'
train_small_path = DATA_DIR + 'train_small.pkl'
test_path = DATA_DIR + 'test.pkl'


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
        plies = self.data[i]

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

        return res, np.int32(ply)


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

    # model = SLPolicy(use_bn=args.use_bn)
    # model = RolloutPolicy()
    model = RLPolicy()

    # log directory
    out = datetime.datetime.now().strftime('%m%d')
    if args.out:
        out = out + '_' + args.out
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", out))
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
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.001))

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
        valid_ply_rate = []

        it = copy.copy(val_iter)
        for batch in it:
            x = chainer.Variable(model.xp.array([b[0] for b in batch], 'float32'), volatile=True)
            y = chainer.Variable(model.xp.array([b[1] for b in batch], 'int32'), volatile=True)
            plies = model(x, y, train=False)
            for b, ply in zip(batch, plies):
                if b[1] >= 0:
                    valid_ply_rate.append(board.is_valid(b[0][:2], b[0][4][0][0], ply // 8, ply % 8))

            test_loss.append(cuda.to_cpu(model.loss.data))
            test_accuracy.append(cuda.to_cpu(model.accuracy.data))

        print('\nepoch {}  train_loss {:.5f}  train_accuracy {:.3f} \n'
              '          test_loss {:.5f}  test_accuracy {:.3f} valid_ply_rate {:.3f}'.
              format(epoch, np.mean(train_loss), np.mean(train_accuracy), np.mean(test_loss), np.mean(test_accuracy),
                     np.mean(valid_ply_rate)))
        with open(os.path.join(out_dir, "log"), 'a+') as f:
            f.write('epoch {}  train_loss {:.5f}  train_accuracy {:.3f} \n'
                    '          test_loss {:.5f}   test_accuracy {:.3f}  valid_ply_rate {:.3f}\n'.
                    format(epoch, np.mean(train_loss), np.mean(train_accuracy), np.mean(test_loss),
                           np.mean(test_accuracy), np.mean(valid_ply_rate)))

        if epoch % 3 == 0:
            serializers.save_hdf5(os.path.join(out_dir, "models", "sl_policy_{}.model".format(epoch)), model)


if __name__ == '__main__':
    main()
