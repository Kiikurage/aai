import sys
import os.path as path
import argparse
import random

sys.path.append(path.join(path.dirname(__file__), "../"))

from reversi import Color, traverse, board


# noinspection PyShadowingNames
def find_ply_montecarlo(b: board.Board, c: Color):
    # noinspection PyUnresolvedReferences
    bb = traverse.BitBoard(b)
    x, y, _ = bb.montecarlo(c, 1000)
    return x, y


# noinspection PyShadowingNames
def find_ply_random(b: board.Board, c: Color):
    que = []
    for x in range(0, 8):
        for y in range(0, 8):
            if board.is_valid(b, c, x, y):
                que.append((x, y))

    if len(que) == 0:
        return -1, -1

    x, y = que[random.randint(0, len(que) - 1)]
    return x, y


strategy_dict = {
    'random': find_ply_random,
    'montecarlo': find_ply_montecarlo
}

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--black', choices=strategy_dict.keys(), default='montecarlo', help='black\'s strategy')
parser.add_argument('-w', '--white', choices=strategy_dict.keys(), default='random', help='black\'s strategy')
args = parser.parse_args()

pass_cnt = 0
c = Color.Black
b = board.init()

strategies = {
    Color.Black: (args.black, strategy_dict[args.black]),
    Color.White: (args.white, strategy_dict[args.white]),
}

while pass_cnt < 2:
    x, y = strategies[c][1](b, c)
    if x == -1:
        pass_cnt += 1
        print('{0}({1}): pass'.format('Black' if c == Color.Black else 'White', strategies[c][0]))
        print('')

    else:
        pass_cnt = 0
        b = board.put(b, c, x, y)
        print('{0}({1}): ({2}, {3})'.format('Black' if c == Color.Black else 'White', strategies[c][0], x, y))
        print(board.stringify(b))
        print('')

    c = 1 - c

num_black = b[0].sum()
num_white = b[1].sum()

print('({0})Black  {1}-{2}  White({3})'.format(
    strategies[Color.Black][0], num_black,
    num_white, strategies[Color.White][0]
))

if num_black > num_white:
    print('Black win')

elif num_white > num_black:
    print('White win')

else:
    print('Draw')