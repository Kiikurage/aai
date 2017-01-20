import random, sys, os
import numpy as np
from chainer import serializers
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
from reversi import board, traverse
from models import SLPolicy, ValueNet

from montecarlo_policy import MontecarloPolicy

policy = None
traverse_policy = None
value_net = None

sl_policy_path1 = '/home/mil/fukuta/work_space/aai/runs/0108_final/models/sl_policy_10.model'
sl_policy_path2 = '/home/mil/fukuta/work_space/aai/runs/0116_new_kifu/models/sl_policy_15.model'
value_net_path = '/home/mil/fukuta/work_space/aai/runs_value/0110_second/models/value_net_20.model'

class Selector:
    # TODO 持つべきは過去の手数の履歴、と現在の状況
    def __init__(self):
        self.count = 0
        self.moves_history = []

    def selectNextMove(*arg):
        return None


def init(ai_type):
    print(ai_type)
    a, b = ai_type.split('-')

    global policy, traverse_policy, value_net

    value_net = ValueNet()

    if a == 'slpolicy':
        policy = SLPolicy()
        serializers.load_hdf5(sl_policy_path1, policy)
    elif a == 'slpolicy2':
        policy = SLPolicy()
        serializers.load_hdf5(sl_policy_path2, policy)

    if b == 'win':
        traverse_policy = MontecarloPolicy(strategy=1)
    elif b == 'draw':
        traverse_policy = MontecarloPolicy(strategy=3)

    print('load model')


def select_move(gameTree):
    board_strings = gameTree["board"]
    moves = gameTree["moves"]
    current_player = gameTree["player"]
    count = gameTree["count"]
    # print(count)

    players = ["black", "white"]
    b = np.zeros((2, 8, 8))

    for (i, player) in enumerate(players):
        player = [1 if color == player else 0 for color in board_strings]
        b[i] = np.array(player).reshape(8, 8)

    # TODO passされたときにturn数を数えられているか確認
    color = players.index(current_player)
    turn = count * 2 - 1 + color

    # print(b)
    stone_cnt = b[0].sum() + b[1].sum()
    if 64 - stone_cnt > 12:
        print("SL", stone_cnt)
        act = policy.act(b, color, turn, temperature=1)
    else:
        print("TRAVERSE", stone_cnt)
        act = traverse_policy.act(b, color, turn, temperature=1)

    best_move = {"x": act % 8, "y": act // 8}

    if best_move in moves:
        best_move_index = moves.index(best_move)
    else:
        best_move_index = 0
    # print(best_move, moves, best_move_index)
    # print(b[0].sum(), b[1].sum())
    return best_move_index
