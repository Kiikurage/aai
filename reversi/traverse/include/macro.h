#ifndef REVERSI_MACRO_H
#define REVERSI_MACRO_H


#define BitSet8(ptr, from, size, bit)   (((char *)(ptr))[(from) / 8] =  ((char *)(ptr))[(from) / 8]                                                              | ((bit) << (8 - (size) - (from) % 8)))
#define BitFSet8(ptr, from, size, bit)  (((char *)(ptr))[(from) / 8] = (((char *)(ptr))[(from) / 8] & (0xFF ^ (((1<<(size)) - 1) << (8 - (size) - (from) % 8)))) | ((bit) << (8 - (size) - (from) % 8)))
#define BitGet8(ptr, from, size)       ((((char *)(ptr))[(from) / 8] >> (8 - (size) - (from) % 8)) & ((1<<(size)) - 1))

#define BitSet(board, x, y, bit)    BitSet8(board, ((x)*8+(y))*2, 2, bit)
#define BitFSet(board, x, y, bit)   BitFSet8(board, ((x)*8+(y))*2, 2, bit)
#define BitGet(board, x, y)         BitGet8(board, ((x)*8+(y))*2, 2)
#define BitCheck(board, x, y, bit) (BitGet(board, x, y) == bit)

#endif //REVERSI_MACRO_H

