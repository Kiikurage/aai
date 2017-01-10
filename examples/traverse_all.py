from reversi import traverse, board, Color
import numpy as np


def init_board_until(n):
    b = board.init()
    cnt = 4
    c = Color.Black

    while cnt < n:
        valid_hands = []
        for x in range(0, 8):
            for y in range(0, 8):
                if board.is_valid(b, c, x, y):
                    valid_hands.append((x, y))

        if len(valid_hands) == 0:
            return None

        x, y = valid_hands[np.random.randint(0, len(valid_hands))]
        b = board.put(b, c, x, y)
        cnt += 1
        c = 1 - c

    return c, b


c, b = init_board_until(10)
print(board.stringify(b))

results = traverse.BitBoard(b).traverse(c, 1)

print(len(results))
for bb in results[:10]:
    print(board.stringify(bb.to_board()))
