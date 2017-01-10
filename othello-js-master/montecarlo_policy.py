import numpy as np
from reversi import traverse, board


class MontecarloPolicy:
    # noinspection PyMethodMayBeStatic
    def act(self, b, c, turn, temperature=0):
        n = b[0].sum() + b[1].sum()
        b = b.astype(np.bool)
        bb = traverse.BitBoard(b)

        if n >= 64 - 14:
            best_rate = -float("Infinity")
            best_x = -1
            best_y = -1

            for x in range(8):
                for y in range(8):
                    if board.is_valid(b, c, x, y):
                        a, total = traverse.BitBoard(board.put(b, c, x, y)).traverse(c, int(64 - n))
                        if 1.0 * a > best_rate:
                            best_rate = 1.0 * a
                            best_x = x
                            best_y = y
                            print(best_rate, best_x, best_y)

            print("BEST:", best_rate, best_x, best_y)
            return best_x * 8 + best_y

        else:
            x, y, _ = bb.montecarlo(c, 10000, 3)
            return x * 8 + y
