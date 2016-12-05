import random
import argparse
import numpy as np
from game import board
import pickle as pkl
import chainer
from chainer import cuda
from chainer import dataset
from chainer import training, iterators
from chainer.training import extensions
from models.sl_policy import SLPolicy

DATA_PATH = '/mnt/share/aai_kikura/data.pkl'


class PreprocessedDataset(dataset.DatasetMixin):
    def __init__(self, data_path=DATA_PATH):
        with open(data_path, 'rb') as f:
            self.data = pkl.load(f)

    def __len__(self):
        return len(self.data)

    # noinspection PyUnresolvedReferences
    def get_example(self, i):
        game = self.data[i]

        # 盤面を作る
        # 各チャンネルの詳細は https://github.com/Kiikurage/aai/issues/13 を参照

        plies = game['plies']
        b = board.init_game()
        n = random.randint(0, len(plies) - 1)

        for color, ply in plies[:n]:
            if ply == -1:
                continue

            y = ply // 8
            x = ply % 8
            b = board.put(b, color, x, y)

        color, ply = plies[n]

        res = board.to_state(b, color, n)

        return res, ply


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
    parser.set_defaults(test=False)
    args = parser.parse_args()

    model = SLPolicy()

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    train = PreprocessedDataset()
    test = PreprocessedDataset()

    train_iter = iterators.SerialIterator(train, 128)
    val_iter = iterators.SerialIterator(test, 128, repeat=False)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), './output')

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

    trainer.run()


if __name__ == '__main__':
    main()
