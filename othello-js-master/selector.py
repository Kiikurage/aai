import random, sys, os
import numpy as np
from chainer import serializers
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
from reversi import board, traverse
from models import SLPolicy, ValueNet
from montecarlo_policy import MontecarloPolicy

ai = None

sl_policy_path1 = '/home/mil/fukuta/work_space/aai/runs/0108_final/models/sl_policy_10.model'
sl_policy_path2 = '/home/mil/fukuta/work_space/aai/runs/0116_new_kifu/models/sl_policy_15.model'
value_net_path = '/home/mil/fukuta/work_space/aai/runs_value/0110_second/models/value_net_20.model'


def softmax(x, T=1.):
    y = x - x.max()
    y = np.exp(y / T)
    y /= y.sum()
    return y


class Selector:
    # TODO 持つべきは過去の手数の履歴、と現在の状況
    def __init__(self, ai_type):
        self.basic_policy, self.strategy = ai_type.split('-')

        self.value_net = ValueNet()
        serializers.load_hdf5(value_net_path, self.value_net)

        if self.basic_policy == 'slpolicy':
            self.policy = SLPolicy()
            serializers.load_hdf5(sl_policy_path1, self.policy)
        elif self.basic_policy == 'slpolicy2':
            self.policy = SLPolicy()
            serializers.load_hdf5(sl_policy_path2, self.policy)
        elif self.basic_policy == 'valuepolicy':
            self.policy = self.value_net
        else:
            raise ValueError('invalid policy')

        if self.strategy == 'win':
            self.traverse_policy = MontecarloPolicy(strategy=1)
        elif self.strategy == 'draw':
            self.traverse_policy = MontecarloPolicy(strategy=3)

        self.count = 0
        self.moves_history = []

    def act(self, b, color, turn):
        state = board.to_state(b, color, turn)
        if state[2].sum() == 0:
            print('pass')
            return -1

        current_score = self.value_net.predict(self.value_net.xp.array([state])).data[0]
        current_score = softmax(current_score)
        print(np.argmax(current_score) - 20)

        if self.basic_policy == 'slpolicy' or 'slpolicy2' or 'valuepolicy':
            action = self.act1(b, color, turn)
        else:
            action = -1

        return action

    def act1(self, b, color, turn):
        stone_cnt = b[0].sum() + b[1].sum()
        if 64 - stone_cnt > 12:
            action = self.policy.act(b, color, turn, temperature=1)
        else:
            print("TRAVERSE", stone_cnt)
            action = self.traverse_policy.act(b, color, turn, temperature=1)

        return action


def init(ai_type):
    print(ai_type)
    global ai
    ai = Selector(ai_type)
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
    stone_cnt = b[0].sum() + b[1].sum()
    turn = stone_cnt - 4

    act = ai.act(b, color, turn)

    best_move = {"x": act % 8, "y": act // 8}

    if best_move in moves:
        best_move_index = moves.index(best_move)
    else:
        best_move_index = 0
    # print(best_move, moves, best_move_index)
    # print(b[0].sum(), b[1].sum())
    return best_move_index
