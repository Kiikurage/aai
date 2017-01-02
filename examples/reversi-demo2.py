import sys
import os.path as path

sys.path.append(path.join(path.dirname(__file__), "../"))

from reversi import Color, traverse, board


# noinspection PyShadowingNames
def find_ply(b: board.Board, c: Color):
    # noinspection PyUnresolvedReferences
    bb = traverse.BitBoard(b)
    x, y, _ = bb.montecarlo(c, 6)
    return x, y


pass_cnt = 0
c = Color.Black
b = board.init()

while pass_cnt < 2:
    x, y = find_ply(b, c)
    if x == -1:
        pass_cnt += 1
        print('{0}: pass'.format('Black' if c == Color.Black else 'White'))
        print('')

    else:
        pass_cnt = 0
        b = board.put(b, c, x, y)
        print('{0}: ({1}, {2})'.format('Black' if c == Color.Black else 'White', x, y))
        print(board.stringify(b))
        print('')

    c = 1 - c
