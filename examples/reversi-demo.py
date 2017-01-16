import argparse
import random
from reversi import Color as C, traverse as T, board as B


def find_ply_montecarlo_traverse_win(board: B, color: C):
    if board[:2].sum() > 52:
        return T.BitBoard(board).traverse(color, 1)
    else:
        return T.BitBoard(board).montecarlo(color, 1000, 1)


def find_ply_montecarlo_traverse_lose(board: B, color: C):
    if board[:2].sum() > 52:
        return T.BitBoard(board).traverse(color, 2)
    else:
        return T.BitBoard(board).montecarlo(color, 1000, 2)


def find_ply_montecarlo_traverse_draw(board: B, color: C):
    if board[:2].sum() > 52:
        return T.BitBoard(board).traverse(color, 3)
    else:
        return T.BitBoard(board).montecarlo(color, 1000, 3)


def find_ply_montecarlo_win(board: B, color: C):
    return T.BitBoard(board).montecarlo(color, 1000, 1)


def find_ply_montecarlo_lose(board: B, color: C):
    return T.BitBoard(board).montecarlo(color, 1000, 2)


def find_ply_montecarlo_draw(board: B, color: C):
    return T.BitBoard(board).montecarlo(color, 1000, 3)


# noinspection PyShadowingNames
def find_ply_random(board: B, color: C):
    hands = T.BitBoard(board).find_next(color)
    return (-1, -1) if len(hands) == 0 else hands[random.randint(0, len(hands) - 1)]


strategy_dict = {
    'random': find_ply_random,

    'montecarlo-traverse-win': find_ply_montecarlo_traverse_win,
    'montecarlo-traverse-lose': find_ply_montecarlo_traverse_lose,
    'montecarlo-traverse-draw': find_ply_montecarlo_traverse_draw,

    'montecarlo-win': find_ply_montecarlo_win,
    'montecarlo-lose': find_ply_montecarlo_lose,
    'montecarlo-draw': find_ply_montecarlo_draw,
}

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--black', choices=strategy_dict.keys(), default='montecarlo-traverse-draw', help='black\'s strategy')
parser.add_argument('-w', '--white', choices=strategy_dict.keys(), default='random', help='black\'s strategy')
args = parser.parse_args()


def main():
    pass_cnt = 0
    color = C.Black
    board = B.init()

    strategies = {
        C.Black: (args.black, strategy_dict[args.black]),
        C.White: (args.white, strategy_dict[args.white]),
    }

    while pass_cnt < 2:
        x, y = strategies[color][1](board, color)
        if x == -1:
            pass_cnt += 1
            print('{0}({1}): pass'.format('Black' if color == C.Black else 'White', strategies[color][0]))
            print('')

        else:
            pass_cnt = 0
            board = B.put(board, color, x, y)
            print('{0}({1}): ({2}, {3})'.format('Black' if color == C.Black else 'White', strategies[color][0], x, y))
            print(B.stringify(board))
            print('')

        color = C.other(color)

    num_black = board[0].sum()
    num_white = board[1].sum()

    print('({0})Black  {1}-{2}  White({3})'.format(
        strategies[C.Black][0], num_black,
        num_white, strategies[C.White][0]
    ))

    if num_black > num_white:
        print('Black win')

    elif num_white > num_black:
        print('White win')

    else:
        print('Draw')


main()
