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

from reversi import board, Color
from models import SLPolicy, RLPolicy


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

    # log directory
    out = datetime.datetime.now().strftime('%m%d')
    if args.out:
        out = out + '_' + args.out
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", out))
    os.makedirs(os.path.join(out_dir, 'models'))

    player_model = RLPolicy(use_bn=args.use_bn)
    opponent_model = SLPolicy(use_bn=args.use_bn)

    # gpu
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        player_model.to_gpu()
        opponent_model.to_gpu()

    # setting
    with open(os.path.join(out_dir, 'setting.txt'), 'w') as f:
        for k, v in args._get_kwargs():
            print('{} = {}'.format(k, v))
            f.write('{} = {}\n'.format(k, v))

    # optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(player_model)

    # start training
    start = time.time()
    train_count = 0

    for epoch in range(args.epoch):
        batch = [board.to_state(board.init(), 0, 0)] * args.batch_size
        states = np.zeros(shape=[args.batch_size, 60, 6, 8, 8])
        plies = np.zeros(shape=[args.batch_size, 60])
        ply_nums = np.zeros(shape=[args.batch_size])
        results = np.zeros(shape=[args.batch_size, 1])

        # todo: restore opponent model

        turn = 0

        for player_color in [Color.Black, Color.White]:
            # rl_policyが黒スタート

            models = {player_color:player_model, 1-player_color:opponent_model}

            c = Color.Black
            pass_cnts = np.zeros(shape=[args.batch_size])
            while True:
                if pass_cnts.min() >= 2:
                    break

                scores = models[c].predict(models[c].xp.array(batch, 'float32'), c, turn)

                for i, b in enumerate(batch):
                    # gameが終わったか判定
                    if pass_cnts[i] == 2:
                        continue

                    #　確率の高い順に打てるかどうかの判定
                    for ply in np.argsort(scores[i]):
                        x = ply // 8
                        y = ply % 8
                        if board.is_valid(b, c, x, y):
                            if c == player_color:
                                states[i, turn, :, :, :] = batch[i]
                                plies[i, turn] = ply
                                ply_nums[i] += 1

                            batch[i] = board.to_state(board.put(b, c, x, y), 1-c, turn+1)
                            pass_cnts[i] = 0
                            break
                    # pass
                    else:
                        plies[i, turn] = -1
                        pass_cnts[i] += 1
                        batch[i] = board.to_state(b, 1-c, turn+1)
                c = 1 - c
                turn += 1

            for i, b in enumerate(batch):
                num_black = b[0].sum()
                num_white = b[1].sum()
                res = np.sign(num_black - num_white)
                results[i] = res if player_color == Color.Black else -res


if __name__ == '__main__':
    main()