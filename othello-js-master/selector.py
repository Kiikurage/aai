import random
import numpy as np

class Selector:
    #TODO 持つべきは過去の手数の履歴、と現在の状況
    def __init__(self,board_strings,moves,current_player):
        self.count = 0
        self.moves_history = []


def selectMove(board_strings,moves,current_player):
    players = ["black","white"]
    board = np.zeros((2,8,8))

    for (i,player) in enumerate(players):
        player = [1 if color==player else 0 for color in board_strings]
        board[i] = np.array(player).reshape(8,8)


    print(board)


    return random.randint(0,len(moves)-1)
