#ifndef REVERSI_MACRO_H
#define REVERSI_MACRO_H

#define BitForceSet(board, x, y, bit) ((board)[(x)/4] = ((board)[(x)/4] ^ (0b11 << ((x) * 8 + (y)) * 2)) | ((bit) << ((x) * 8 + (y)) * 2))
#define BitSet(board, x, y, bit) ((board)[(x)/4] |= ((bit) << ((x) * 8 + (y)) * 2))

#endif //REVERSI_MACRO_H
