//
// Created by kikura on 2017/01/03.
//

#ifndef REVERSI_XORSHIFT_H
#define REVERSI_XORSHIFT_H

#include <stdint.h>
#include <stdio.h>


static uint32_t x = 123456789;
static uint32_t y = 362436069;
static uint32_t z = 521288629;
static uint32_t w = 88675123;
static char init_flag = 0;

uint32_t xor128(void) {
    if (!init_flag) {
        uint32_t _data[4];
        FILE *fp;
        fp = fopen("/dev/urandom", "r");
        fread(_data, sizeof(uint32_t), 4, fp);
        fclose(fp);
        x = _data[0] ^ x;
        y = _data[1] ^ y;
        z = _data[2] ^ z;
        w = _data[3] ^ w;
    };
    uint32_t t;

    t = x ^ (x << 11);
    x = y;
    y = z;
    z = w;
    return w = (w ^ (w >> 19)) ^ (t ^ (t >> 8));
}

#endif //REVERSI_XORSHIFT_H
