import argparse
import os, sys
import numpy as np
import datetime
import time
import chainer
from chainer import cuda
from chainer import serializers

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reversi import board, Color
from models import SLPolicy, RLPolicy


def softmax(x, T=1):
    y = x - x.max(axis=1, keepdims=True)
    y = np.exp(y / T)
    y /= y.sum(axis=1, keepdims=True)
    return y


def progress_report(start_time, epoch, batchsize, win_rate):
    duration = time.time() - start_time
    throughput = duration / ((epoch + 1) * batchsize)
    sys.stderr.write(
        '\repoch {} time: {} win_rate {:.3f} ({:.2f} sec/game)'.format(
            epoch, str(datetime.timedelta(seconds=duration)).split('.')[0], win_rate, throughput
        )
    )


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--init', '-i', help='path to initial olayer model')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU device ID')
    parser.add_argument('--epoch', '-e', type=int, default=10000, help='# of epoch')
    parser.add_argument('--batch_size', type=int, default=64, help='size of mini-batch')
    parser.add_argument('--density', type=int, default=1, help='density of cnn kernel')
    parser.add_argument('--no_bn', dest='use_bn', action='store_false', default=True)
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
    opponent_model = RLPolicy(use_bn=args.use_bn)

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
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(player_model)

    # start training
    start = time.time()

    # opponent model
    opponent_models = [args.init]

    for epoch in range(args.epoch):

        # load opponent model
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

                scores = models[c].predict(models[c].xp.array(x_batch, 'float32'), False)
                pred = softmax(cuda.to_cpu(scores.data))

                for i in range(batch_size//2):
                    # gameが終わったか判定
                    if pass_cnts[i] == 2:
                        continue

                    b = x_batch[i][:2]

                    # 確率の高い順に打てるかどうかの判定
                    # for ply in np.argsort(-pred[i]):
                    for _ in range(10):
                        ply = np.random.choice(64, p=pred[i])
                        x = ply // 8
                        y = ply % 8
                        if board.is_valid(b, c, x, y):
                            if c == player_color:
                                states[(batch_size // 2) * player_color + i, turn, :, :, :] = x_batch[i]
                                plies[(batch_size // 2) * player_color + i, turn] = ply
                                ply_nums[(batch_size // 2) * player_color + i] += 1

                            x_batch[i] = board.to_state(board.put(b, c, x, y), 1 - c, turn + 1)
                            pass_cnts[i] = 0
                            break
                    # case of pass
                    else:
                        for ply in np.argsort(-pred[i]):
                            x = ply // 8
                            y = ply % 8
                            if board.is_valid(b, c, x, y):
                                if c == player_color:
                                    states[(batch_size // 2) * player_color + i, turn, :, :, :] = x_batch[i]
                                    plies[(batch_size // 2) * player_color + i, turn] = ply
                                    ply_nums[(batch_size // 2) * player_color + i] += 1

                                x_batch[i] = board.to_state(board.put(b, c, x, y), 1 - c, turn + 1)
                                pass_cnts[i] = 0
                                break
                        else:
                            plies[(batch_size // 2) * player_color + i, turn] = -1
                            pass_cnts[i] += 1
                            x_batch[i] = board.to_state(b, 1 - c, turn + 1)
                c = 1 - c
                turn += 1

            # check win/lose
            for i, b in enumerate(x_batch):
                num_black = b[0].sum()
                num_white = b[1].sum()
                res = np.sign(num_black - num_white)
                results[(batch_size // 2) * player_color+i] = res if player_color == Color.Black else -res

        # train (policy gradient)
        optimizer.update(player_model, states, plies, results, ply_nums)

        win_rate = np.mean([max(r, 0) for r in results])
        progress_report(start, epoch, batch_size, win_rate)

        if epoch % 100 == 0:
            serializers.save_hdf5(os.path.join(out_dir, "models", "rl_policy_{}.model".format(epoch)), player_model)
            print()


if __name__ == '__main__':
    main()
