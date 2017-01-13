from reversi import traverse as T, board as B, Color
import numpy as np
from matplotlib import pyplot as plt


def main():

    board, prob = T.BitBoard(B.init()).get_score_prob2(Color.Black, 54, 1000)
    print(B.stringify(board.to_board()))

    f = plt.figure()
    a = f.add_subplot(111)
    a.plot(np.arange(127) - 64, np.array(prob))
    a.set_ylim((0, 1))
    a.set_xlim(-64, 64)
    a.set_xlabel('#BLACK - #WHITE')
    a.set_ylabel('probability')
    f.savefig("/Users/kikura/Desktop/prob.png")


main()