import random
import argparse
import os, sys
import numpy as np
import datetime
import pickle as pkl
import chainer
from chainer import cuda
from chainer import dataset
from chainer import training, iterators
from chainer.training import extensions

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reversi import board
from models import SLPolicy

DATA_DIR = '/mnt/share/aai_fukuta/'
train_path = DATA_DIR + 'train.pkl'
train_small_path = DATA_DIR + 'train_small.pkl'
test_path = DATA_DIR + 'test.pkl'


class PreprocessedDataset(dataset.DatasetMixin):
    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            self.data = pkl.load(f)

    def __len__(self):
        return len(self.data)

    # noinspection PyUnresolvedReferences
    def get_example(self, i):
        plies = self.data[i]

        # 盤面を作る
        # 各チャンネルの詳細は https://github.com/Kiikurage/aai/issues/13 を参照

        # plies = game['plies']
        b = board.init()
        n = random.randint(0, len(plies) - 1)

        for color, ply in plies[:n]:
            if ply == -1:
                continue

            y = ply // 8
            x = ply % 8
            b = board.put(b, color, x, y)

        color, ply = plies[n]

        res = board.to_state(b, color, n)

        return res, np.int32(ply)


class TestModeEvaluator(extensions.Evaluator):
    def evaluate(self):
        model = self.get_target('main')
        model.train = False
        ret = super(TestModeEvaluator, self).evaluate()
        model.train = True
        return ret


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU device ID')
    parser.add_argument('--epoch', '-e', type=int, default=50, help='# of epoch')
    parser.add_argument('--batch_size', type=int, default=256, help='size of mini-batch')
    parser.add_argument('--density', type=int, default=1, help='density of cnn kernel')
    parser.add_argument('--small', dest='small', action='store_true', default=False)
    parser.add_argument('--use_bn', dest='use_bn', action='store_true', default=False)
    parser.add_argument('--out', default='')
    parser.set_defaults(test=False)
    args = parser.parse_args()

    model = SLPolicy(use_bn=args.use_bn)

    out = datetime.datetime.now().strftime('%m%d')
    if args.out:
        out = out + '_' + args.out
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", out))

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    with open(os.path.join(out_dir, 'setting.txt'), 'w') as f:
        for k, v in args._get_kwargs():
            print('{} = {}'.format(k, v))
            f.write('{} = {}\n'.format(k, v))

    if args.small:
        train = PreprocessedDataset(train_small_path)
    else:
        train = PreprocessedDataset(train_path)
    test = PreprocessedDataset(test_path)

    train_iter = iterators.SerialIterator(train, args.batch_size)
    val_iter = iterators.SerialIterator(test, args.batch_size, repeat=False)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out_dir)

    val_interval = (1000, 'iteration')
    log_interval = (100, 'iteration')
    snapshot_interval = (10000, 'iteration')

    trainer.extend(TestModeEvaluator(val_iter, model, device=args.gpu), trigger=val_interval)
    trainer.extend(extensions.snapshot(), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(model, 'model_iter_{.updater.iteration}'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration',
        'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy',
        'lr'
    ]), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=log_interval[0]))

    print('start training')
    trainer.run()


if __name__ == '__main__':
    main()
