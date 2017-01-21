import argparse
import os, sys
import numpy as np
import datetime
import time
import chainer
from chainer import cuda
from chainer import serializers

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reversi import board, Color, traverse
from models import SLPolicy, RLPolicy, RolloutPolicy


def softmax(x, mask, T=1):
    y = x - x.max()
    y = np.exp(y / T) * mask
    y /= y.sum()
    return y


def progress_report(start_time, epoch, batchsize, win_rate):
    duration = time.time() - start_time
    throughput = ((epoch + 1) * batchsize) / duration
    sys.stderr.write(
        '\repoch {} time: {} win_rate {:.3f} ({:.2f} sec/game)'.format(
            epoch, str(datetime.timedelta(seconds=duration)).split('.')[0], win_rate, throughput
        )
    )


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--init', '-i', help='path to initial olayer model')
    parser.add_argument('--opponent')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU device ID')
    parser.add_argument('--epoch', '-e', type=int, default=10000, help='# of epoch')
    parser.add_argument('--batch_size', type=int, default=256, help='size of mini-batch')
    parser.add_argument('--adam_eps', type=float, default=1e-2, help='parameter eps in adam')
    parser.add_argument('--adam_alpha', type=float, default=1e-4, help='parameter alpha in adam')
    parser.add_argument('--density', type=int, default=1, help='density of cnn kernel')
    parser.add_argument('--no_bn', dest='use_bn', action='store_false', default=True)
    parser.add_argument('--draw', dest='draw', action='store_true', default=False)
    parser.add_argument('--out', default='')
    parser.set_defaults(test=False)
    args = parser.parse_args()

    batch_size = args.batch_size

    # log directory
    out = datetime.datetime.now().strftime('%m%d')
    if args.out:
        out = out + '_' + args.out
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs_rl", out))
    os.makedirs(os.path.join(out_dir, 'models'), exist_ok=True)

    player_model = RLPolicy(use_bn=args.use_bn)
    # opponent_model = RLPolicy(use_bn=args.use_bn)
    opponent_model = RolloutPolicy()

    # load player model
    serializers.load_hdf5(args.init, player_model)

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
    optimizer = chainer.optimizers.Adam(alpha=args.adam_alpha, eps=args.adam_eps)
    optimizer.setup(player_model)

    # start training
    start = time.time()

    # opponent model
    if args.opponent is None:
        opponent = args.init
    else:
        opponent = args.opponent
    opponent_models = [opponent]

    win_rate_summary = []
    for epoch in range(args.epoch):

        # load opponent model
        if epoch % 1000 == 0:
            serializers.load_hdf5(np.random.choice(opponent_models), opponent_model)

        # initialize
        states = np.zeros(shape=(batch_size, 80, 6, 8, 8))
        plies = np.zeros(shape=(batch_size, 80), dtype='int64')
        ply_nums = np.zeros(shape=batch_size, dtype='int64')
        results = np.zeros(shape=batch_size)

        # simulation (self-play)
        for player_color in [Color.Black, Color.White]:

            models = {player_color: player_model, 1 - player_color: opponent_model}
            x_batch = [board.to_state(board.init(), 0, 0) for _ in range(batch_size // 2)]
            turn = 0
            c = Color.Black
            pass_cnts = np.zeros(shape=batch_size // 2)
            while True:
                if min(pass_cnts) >= 2:
                    break

                if c == player_color:
                    scores = models[c].predict(models[c].xp.array(x_batch, 'float32'), False)
                    scores = cuda.to_cpu(scores.data)

                for i in range(batch_size//2):
                    # gameが終わったか判定
                    if pass_cnts[i] == 2:
                        continue

                    b = x_batch[i][:2]
                    valid_mask = x_batch[i][2].ravel()

                    if valid_mask.sum() == 0:
                        plies[(batch_size // 2) * player_color + i, turn] = -1
                        pass_cnts[i] += 1
                        x_batch[i] = board.to_state(b, 1 - c, turn + 1)
                    else:
                        stone_cnt = b[0:2].sum()
                        if c == player_color:
                            if stone_cnt >= 64 - 8:
                                # 残り12手は探索で。
                                # print('in zentansaku', stone_cnt)
                                if args.draw:
                                    x, y = traverse.BitBoard(b.astype(np.bool)).traverse(c, 3)
                                else:
                                    x, y = traverse.BitBoard(b.astype(np.bool)).traverse(c, 1)
                                ply = x * 8 + y

                            else:
                                pred = softmax(scores[i].astype(np.float64), mask=valid_mask, T=1)
                                ply = np.random.choice(64, p=pred)

                                states[(batch_size // 2) * player_color + i, turn, :, :, :] = x_batch[i]
                                plies[(batch_size // 2) * player_color + i, turn] = ply
                                ply_nums[(batch_size // 2) * player_color + i] += 1
                            x = ply // 8
                            y = ply % 8

                        else:
                            stone_cnt = b[0].sum() + b[1].sum()
                            b = b.astype(np.bool)
                            bb = traverse.BitBoard(b)

                            if 64 - stone_cnt > 12:
                                x, y = bb.montecarlo(c, 10000, 1)
                            else:
                                x, y = traverse.BitBoard(b.astype(np.bool)).traverse(c, 1)

                        if not board.is_valid(b, c, x, y):
                            print(valid_mask)
                            print(scores[i])
                            print(softmax(scores[i], mask=valid_mask, T=1))
                            raise ValueError('invalid ply')

                        x_batch[i] = board.to_state(board.put(b, c, x, y), 1 - c, turn + 1)
                        pass_cnts[i] = 0

                c = 1 - c
                turn += 1

            # check win/lose
            for i, b in enumerate(x_batch):
                num_black = b[0].sum()
                num_white = b[1].sum()
                if args.draw:
                    diff = abs(num_black - num_white)
                    if diff <= 3:
                        res = 1
                    elif diff <= 10:
                        res = 0
                    else:
                        res = -0.5
                    results[(batch_size // 2) * player_color + i] = res
                else:
                    res = np.sign(num_black - num_white)
                    res = res if player_color == Color.Black else -res
                    results[(batch_size // 2) * player_color+i] = res if res >= 0 else res / 2

        # train (policy gradient)
        optimizer.update(player_model, states, plies, results, ply_nums)

        if args.draw:
            win_rate = np.mean(np.array(results) > 0)
        else:
            win_rate = np.mean(np.array(results) >= 0)

        progress_report(start, epoch, batch_size, win_rate)
        win_rate_summary.append(win_rate)
        if epoch % 100 == 0:
            serializers.save_hdf5(os.path.join(out_dir, "models", "rl_policy_{}.model".format(epoch)), player_model)
            print('\nwin_rate_summary {}'.format(np.mean(win_rate_summary)))
            win_rate_summary = []


if __name__ == '__main__':
    main()
