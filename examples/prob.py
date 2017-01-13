from reversi import board, traverse, Color
from matplotlib import pyplot as plt
import numpy as np

b = board.init()
bb = traverse.BitBoard(b)
prob = bb.get_score_prob(Color.Black, 1000)

f = plt.figure()
a = f.add_subplot(111)
a.plot(np.arange(127) - 64, np.array(prob))
a.set_ylim((0, 1))
a.set_xlim(-64, 64)
a.set_xlabel('#BLACK - #WHITE')
a.set_ylabel('probability')
f.show()
