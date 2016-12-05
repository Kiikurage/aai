import sys
import os.path as path

sys.path.append(path.join(path.dirname(__file__), "../"))

from reversi import Color
from reversi.board import init, stringify, put, Board, is_valid
import random


# noinspection PyShadowingNames
def find_ply(b: Board, c: Color):
    que = []
    for x in range(0, 8):
        for y in range(0, 8):
            if is_valid(b, c, x, y):
                que.append((x, y))

    if len(que) == 0:
        return -1, -1

    x, y = que[random.randint(0, len(que) - 1)]
    return x, y


pass_cnt = 0
c = Color.Black
b = init()

while pass_cnt < 2:
    x, y = find_ply(b, c)
    if x == -1:
        pass_cnt += 1
        print('{0}: pass'.format('Black' if c == Color.Black else 'White'))
        print('')

    else:
        pass_cnt = 0
        b = put(b, c, x, y)
        print('{0}: ({1}, {2})'.format('Black' if c == Color.Black else 'White', x, y))
        print(stringify(b))
        print('')

    c = 1 - c
