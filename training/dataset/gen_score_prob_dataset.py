from reversi import board as B, traverse as T, Color
import numpy as np

bitboard = T.BitBoard(B.init())

for max_num_stone in range(54, 4, -1):

    bitboards, probs = bitboard.get_score_prob2(Color.Black, max_num_stone, 100)
    print(len(bitboards))
    np.savez_compressed("./dataset.npz",
                        boards=np.array(bitboards),
                        probs=np.array(probs))
    print("save >> {0}".format(max_num_stone))
