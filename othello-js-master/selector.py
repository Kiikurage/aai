import random,sys,os
import numpy as np
from chainer import serializers

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reversi import board
from models import SLPolicy

class Selector:
    #TODO 持つべきは過去の手数の履歴、と現在の状況
    def __init__(self,board_strings,moves,current_player):
        self.count = 0
        self.moves_history = []
    def selectNextMove(*arg):
        return null

#def init():


def selectMove(gameTree, policy):
    board_strings = gameTree["board"]
    moves = gameTree["moves"]
    current_player = gameTree["player"]
    count = gameTree["count"]
    print(count)

    players = ["black","white"]
    board = np.zeros((2,8,8))

    for (i,player) in enumerate(players):
        player = [1 if color==player else 0 for color in board_strings]
        board[i] = np.array(player).reshape(8,8)

    color = players.index(current_player)
    turn = count*2 - 1 + color

    print(board)
    act = policy.act(board, color, turn, temperature=1)

    print(moves)
    print("act",act//8, act%8)
    bestMove = {"x":act%8,"y":act//8}


    bestMoveIndex = 0

    if bestMove in moves:
        bestMoveIndex = moves.index(bestMove)
    print(bestMove,moves,bestMoveIndex)
    return bestMoveIndex
