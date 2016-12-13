import sys, os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'existing', 'python'))

from reversi_main import Disc, Reversi
from game.framework import Board, State
from game.console import choose_agent


"""
CPU.__init__:
 agent選択

CPU.play():
 入力: board([0] -> black, [1] -> white, else -> 仮定なし)
 出力: 次の手(x, y)

"""


def aggregate_board(board):
    grid = [[Disc.EMPTY for _ in range(0, 8)] for _ in range(0, 8)]
    for y in range(0, 8):
        for x in range(0, 8):
            if board[0, x, y]:
                grid[y][x] = Disc.BLACK
                # black_score += 1
            elif board[1, x, y]:
                grid[y][x] = Disc.WHITE
                # white_score += 1

    _board = Board(grid, Disc.EMPTY, {Disc.EMPTY: '   ', Disc.BLACK: ' ● ', Disc.WHITE: ' ○ '})

    return _board


class CPU(object):
    def __init__(self, black=True):
        self.agent = choose_agent('Choose a cpu agent type')
        if black is True:
            self.env = Reversi(self.agent, None)
        else:
            self.env = Reversi(None, self.agent)

    def play(self, board) -> (int, int):
        _board = aggregate_board(board)
        state = State(_board, self.agent, None, 0, 0)

        action = state.agent.decide(self.env, state)

        return action