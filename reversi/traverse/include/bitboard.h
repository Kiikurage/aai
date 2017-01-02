//
// Created by kikura on 2017/01/02.
//

#ifndef REVERSI_BITBOARD_H
#define REVERSI_BITBOARD_H

#include <Python.h>
#include <numpy/ndarraytypes.h>
#include <numpy/ndarrayobject.h>
#include <x86intrin.h>
#include <structmember.h>
#include <macro.h>
#include <xorshift.h>

typedef char Color;
const Color EMPTY = 0b11;
const Color BLACK = 0b00;
const Color WHITE = 0b01;

#define other(c) ((c)==BLACK?WHITE:BLACK)

typedef struct {
    PyObject_HEAD
    __m128 data;
} BitBoard;

static __m128 put_and_flip(__m128 board, const Color color, const int x, const int y) {
    // TODO bit演算で。

    __m128 result = board;

    if (!BitCheck(&result, x, y, EMPTY)) return result;
    BitFSet(&result, x, y, color);

    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            if (dx == 0 && dy == 0) continue;

            int px = x + dx;
            int py = y + dy;
            if (px < 0 || px >= 8 || py < 0 || py >= 8 || !BitCheck(&result, px, py, other(color))) continue;

            while (1) {
                px += dx;
                py += dy;

                if (px < 0 || px >= 8 || py < 0 || py >= 8) break;

                if (!BitCheck(&result, px, py, other(color))) {
                    if (BitCheck(&result, px, py, color)) {
                        while (1) {
                            px -= dx;
                            py -= dy;

                            if (px == x && py == y) break;

                            BitFSet(&result, px, py, color);
                        }
                    }
                    break;
                }
            }
        }
    }

    return result;
};

static void find_next(__m128 board, const Color color, int *buf_x, int *buf_y, int *n_hands) {
    // TODO bit演算で。
    char *tmp = calloc(sizeof(char), 64);

    for (int i = 0; i < 64; i++) {
        const int x = i / 8;
        const int y = i % 8;

        if (!BitCheck(&board, x, y, EMPTY)) continue;

        char success = 0;

        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                if (dx == 0 && dy == 0) continue;

                int px = x + dx;
                int py = y + dy;
                if (px < 0 || px >= 8 || py < 0 || py >= 8 || !BitCheck(&board, px, py, other(color))) continue;

                while (1) {
                    px += dx;
                    py += dy;

                    if (px < 0 || px >= 8 || py < 0 || py >= 8) break;

                    if (!BitCheck(&board, px, py, other(color))) {
                        if (BitCheck(&board, px, py, color)) {
                            success = 1;
                            tmp[i] = 1;
                        }
                        break;
                    }
                }

                if (success) break;
            }
            if (success) break;
        }
    }

    *n_hands = 0;
    for (int i = 0; i < 64; i++) {
        if (tmp[i] == 0) continue;
        buf_x[*n_hands] = i / 8;
        buf_y[*n_hands] = i % 8;
        (*n_hands)++;
    }

    free(tmp);
};

typedef struct {
    int black;
    int white;
    int empty;
} Summary;

static Summary summary(__m128 board) {
    Summary summary = (Summary) {0, 0, 0};

    for (int x = 0; x < 8; x++) {
        for (int y = 0; y < 8; y++) {
            if (BitCheck(&board, x, y, BLACK)) {
                summary.black++;
            } else if (BitCheck(&board, x, y, WHITE)) {
                summary.white++;
            } else {
                summary.empty++;
            }
        }
    }

    return summary;
};

static void BitBoard_dealloc(BitBoard *self) {
    //Nothing To Do.
}

static PyObject *BitBoard_montecalro(BitBoard *self, PyObject *args) {
    const Color start_color;
    const int limit;
    if (!PyArg_ParseTuple(args, "ii", &start_color, &limit)) return NULL;

    int *buf_x = (int *) malloc(sizeof(int) * 64);
    int *buf_y = (int *) malloc(sizeof(int) * 64);
    int n_valid_hands = 0;
    find_next(self->data, start_color, buf_x, buf_y, &n_valid_hands);

    if (n_valid_hands == 0) return Py_BuildValue("(iii)", -1, -1, 0);

    int *buf_win_count = (int *) calloc(sizeof(int), (size_t) n_valid_hands);
    int *buf_x2 = (int *) malloc(sizeof(int) * 64);
    int *buf_y2 = (int *) malloc(sizeof(int) * 64);

    for (int i_hand = 0; i_hand < n_valid_hands; i_hand++) {

        for (int i_try = 0; i_try < limit; i_try++) {
            __m128 board = put_and_flip(self->data, start_color, buf_x[i_hand], buf_y[i_hand]);
            int pass_count = 0;
            Color current = other(start_color);
            while (1) {
                int n_valid_hands2 = 0;
                find_next(board, current, buf_x2, buf_y2, &n_valid_hands2);

                if (n_valid_hands2 == 0) {
                    pass_count++;
                    if (pass_count >= 2) break;

                    current = other(current);
                    continue;

                } else {
                    pass_count = 0;

                    const int selected_hand = xor128() % n_valid_hands2;
                    board = put_and_flip(board, current, buf_x2[selected_hand], buf_y2[selected_hand]);
                    current = other(current);
                }
            }

            Summary s = summary(board);
            buf_win_count[i_hand] += start_color == BLACK ? s.black - s.white : s.white - s.black;
        }
    }

    free(buf_x2);
    free(buf_y2);

    int best_win_count = buf_win_count[0];
    int best_x = buf_x[0];
    int best_y = buf_y[0];
    for (int i_hand = 1; i_hand < n_valid_hands; i_hand++) {
        if (best_win_count > buf_win_count[i_hand]) continue;

        best_win_count = buf_win_count[i_hand];
        best_x = buf_x[i_hand];
        best_y = buf_y[i_hand];
    }

    free(buf_x);
    free(buf_y);
    free(buf_win_count);
    return Py_BuildValue("(iii)", best_x, best_y, best_win_count);
};

static PyObject *BitBoard_montecalro_draw(BitBoard *self, PyObject *args) {
    //引き分けを目指す
    const Color start_color;
    const int limit;
    if (!PyArg_ParseTuple(args, "ii", &start_color, &limit)) return NULL;

    int *buf_x = (int *) malloc(sizeof(int) * 64);
    int *buf_y = (int *) malloc(sizeof(int) * 64);
    int n_valid_hands = 0;
    find_next(self->data, start_color, buf_x, buf_y, &n_valid_hands);

    if (n_valid_hands == 0) return Py_BuildValue("(iii)", -1, -1, 0);

    int *buf_win_count = (int *) calloc(sizeof(int), (size_t) n_valid_hands);
    int *buf_x2 = (int *) malloc(sizeof(int) * 64);
    int *buf_y2 = (int *) malloc(sizeof(int) * 64);

    for (int i_hand = 0; i_hand < n_valid_hands; i_hand++) {

        for (int i_try = 0; i_try < limit; i_try++) {
            __m128 board = put_and_flip(self->data, start_color, buf_x[i_hand], buf_y[i_hand]);
            int pass_count = 0;
            Color current = other(start_color);
            while (1) {
                int n_valid_hands2 = 0;
                find_next(board, current, buf_x2, buf_y2, &n_valid_hands2);

                if (n_valid_hands2 == 0) {
                    pass_count++;
                    if (pass_count >= 2) break;

                    current = other(current);
                    continue;

                } else {
                    pass_count = 0;

                    const int selected_hand = xor128() % n_valid_hands2;
                    board = put_and_flip(board, current, buf_x2[selected_hand], buf_y2[selected_hand]);
                    current = other(current);
                }
            }

            Summary s = summary(board);
            buf_win_count[i_hand] += -(s.white - s.black) * (s.white - s.black);
        }
    }

    free(buf_x2);
    free(buf_y2);

    int best_win_count = buf_win_count[0];
    int best_x = buf_x[0];
    int best_y = buf_y[0];
    for (int i_hand = 1; i_hand < n_valid_hands; i_hand++) {
        if (best_win_count > buf_win_count[i_hand]) continue;

        best_win_count = buf_win_count[i_hand];
        best_x = buf_x[i_hand];
        best_y = buf_y[i_hand];
    }

    free(buf_x);
    free(buf_y);
    free(buf_win_count);
    return Py_BuildValue("(iii)", best_x, best_y, best_win_count);
};

static PyObject *BitBoard_montecalro_negative(BitBoard *self, PyObject *args) {
    //負けを目指す
    const Color start_color;
    const int limit;
    if (!PyArg_ParseTuple(args, "ii", &start_color, &limit)) return NULL;

    int *buf_x = (int *) malloc(sizeof(int) * 64);
    int *buf_y = (int *) malloc(sizeof(int) * 64);
    int n_valid_hands = 0;
    find_next(self->data, start_color, buf_x, buf_y, &n_valid_hands);

    if (n_valid_hands == 0) return Py_BuildValue("(iii)", -1, -1, 0);

    int *buf_win_count = (int *) calloc(sizeof(int), (size_t) n_valid_hands);
    int *buf_x2 = (int *) malloc(sizeof(int) * 64);
    int *buf_y2 = (int *) malloc(sizeof(int) * 64);

    for (int i_hand = 0; i_hand < n_valid_hands; i_hand++) {

        for (int i_try = 0; i_try < limit; i_try++) {
            __m128 board = put_and_flip(self->data, start_color, buf_x[i_hand], buf_y[i_hand]);
            int pass_count = 0;
            Color current = other(start_color);
            while (1) {
                int n_valid_hands2 = 0;
                find_next(board, current, buf_x2, buf_y2, &n_valid_hands2);

                if (n_valid_hands2 == 0) {
                    pass_count++;
                    if (pass_count >= 2) break;

                    current = other(current);
                    continue;

                } else {
                    pass_count = 0;

                    const int selected_hand = xor128() % n_valid_hands2;
                    board = put_and_flip(board, current, buf_x2[selected_hand], buf_y2[selected_hand]);
                    current = other(current);
                }
            }

            Summary s = summary(board);
            buf_win_count[i_hand] += start_color == WHITE ? s.black - s.white : s.white - s.black;
        }
    }

    free(buf_x2);
    free(buf_y2);

    int best_win_count = buf_win_count[0];
    int best_x = buf_x[0];
    int best_y = buf_y[0];
    for (int i_hand = 1; i_hand < n_valid_hands; i_hand++) {
        if (best_win_count > buf_win_count[i_hand]) continue;

        best_win_count = buf_win_count[i_hand];
        best_x = buf_x[i_hand];
        best_y = buf_y[i_hand];
    }

    free(buf_x);
    free(buf_y);
    free(buf_win_count);
    return Py_BuildValue("(iii)", best_x, best_y, best_win_count);
};

static PyObject *BitBoard_to_board(BitBoard *self) {
    PyArrayObject *result = (PyArrayObject *) PyArray_SimpleNew(3, ((npy_intp[3]) {2, 8, 8}), NPY_BOOL);
    PyArray_FILLWBYTE(result, NPY_FALSE);

    for (int i = 0; i < 64; i++) {
        int cell = BitGet(&self->data, i / 8, i % 8);
        if (cell == BLACK) {
            *(npy_bool *) PyArray_GETPTR3(result, 0, i / 8, i % 8) = NPY_TRUE;

        } else if (cell == WHITE) {
            *(npy_bool *) PyArray_GETPTR3(result, 1, i / 8, i % 8) = NPY_TRUE;
        }
    }

    return (PyObject *) result;
};

static PyMethodDef BitBoard_methods[] = {
    {"to_board",   (PyCFunction) BitBoard_to_board,   METH_NOARGS,
        "to_board()\n"
            "--\n"
            "\n"
            "BitBoardをBoardへ変換します\n"
            "\n"
            "Parameters\n"
            "----------\n"
            "無し\n"
            "\n"
            "Returns\n"
            "-------\n"
            "board : Board\n"
            "  Boardオブジェクト"
    },
    {"montecarlo", (PyCFunction) BitBoard_montecalro, METH_VARARGS,
        "montecarlo(color, limit)\n"
            "--\n"
            "\n"
            "モンテカルロ探索により、最善手を探索します。\n"
            "\n"
            "Parameters\n"
            "----------\n"
            "color: int\n"
            "  最初に指す色。0が黒、1が白を表す。\n"
            "limit: int\n"
            "  打ち切り時間。この時間をすぎると探索を打ち切ります。単位は秒です。"
            "\n"
            "Returns\n"
            "-------\n"
            "x : int\n"
            "  探索結果のx座標。引数colorで指定した指し手にとって、次に指すべき最善手。\n"
            "y : int\n"
            "  探索結果のy座標。引数colorで指定した指し手にとって、次に指すべき最善手。\n"
            "n : int\n"
            "  完了した試行回数\n"
    },
    {"montecarlo_draw", (PyCFunction) BitBoard_montecalro_draw, METH_VARARGS,
        "montecarlo(color, limit)\n"
            "--\n"
            "\n"
            "モンテカルロ探索により、最善手を探索します。\n"
            "\n"
            "Parameters\n"
            "----------\n"
            "color: int\n"
            "  最初に指す色。0が黒、1が白を表す。\n"
            "limit: int\n"
            "  打ち切り時間。この時間をすぎると探索を打ち切ります。単位は秒です。"
            "\n"
            "Returns\n"
            "-------\n"
            "x : int\n"
            "  探索結果のx座標。引数colorで指定した指し手にとって、次に指すべき最善手。\n"
            "y : int\n"
            "  探索結果のy座標。引数colorで指定した指し手にとって、次に指すべき最善手。\n"
            "n : int\n"
            "  完了した試行回数\n"
    },
    {"montecarlo_negative", (PyCFunction) BitBoard_montecalro_negative, METH_VARARGS,
        "montecarlo(color, limit)\n"
            "--\n"
            "\n"
            "モンテカルロ探索により、最善手を探索します。\n"
            "\n"
            "Parameters\n"
            "----------\n"
            "color: int\n"
            "  最初に指す色。0が黒、1が白を表す。\n"
            "limit: int\n"
            "  打ち切り時間。この時間をすぎると探索を打ち切ります。単位は秒です。"
            "\n"
            "Returns\n"
            "-------\n"
            "x : int\n"
            "  探索結果のx座標。引数colorで指定した指し手にとって、次に指すべき最善手。\n"
            "y : int\n"
            "  探索結果のy座標。引数colorで指定した指し手にとって、次に指すべき最善手。\n"
            "n : int\n"
            "  完了した試行回数\n"
    },
    {NULL}  // END Marker
};

static PyMemberDef BitBoard_members[] = {
    {NULL}  // END Marker
};

static int BitBoard_init(BitBoard *self, PyObject *args) {
    PyArrayObject *board;

    if (!PyArg_ParseTuple(args, "O", &board)) return -1;

    if (board) {
        for (int i = 0; i < 64; i++) {
            if (*(npy_bool *) PyArray_GETPTR3(board, 0, i / 8, i % 8)) {
                BitFSet(&self->data, i / 8, i % 8, BLACK);

            } else if (*(npy_bool *) PyArray_GETPTR3(board, 1, i / 8, i % 8)) {
                BitFSet(&self->data, i / 8, i % 8, WHITE);

            } else {
                BitFSet(&self->data, i / 8, i % 8, EMPTY);
            }
        }
    }

    return 0;
}

static PyObject *BitBoard_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    BitBoard *self;

    self = (BitBoard *) type->tp_alloc(type, 0);
    if (self != NULL) {
        int *p_data = (int *) (&self->data);
        for (int i = 0; i < 4; i++) p_data[i] = 0xFFFFFFFF;
    }

    return (PyObject *) self;
}

static PyTypeObject BitBoard_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "traverse.BitBoard",           /* tp_name */
    sizeof(BitBoard),              /* tp_basicsize */
    0,                             /* tp_itemsize */
    (destructor) BitBoard_dealloc, /* tp_dealloc */
    0,                             /* tp_print */
    0,                             /* tp_getattr */
    0,                             /* tp_setattr */
    0,                             /* tp_reserved */
    0,                             /* tp_repr */
    0,                             /* tp_as_number */
    0,                             /* tp_as_sequence */
    0,                             /* tp_as_mapping */
    0,                             /* tp_hash  */
    0,                             /* tp_call */
    0,                             /* tp_str */
    0,                             /* tp_getattro */
    0,                             /* tp_setattro */
    0,                             /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,            /* tp_flags */
    "BitBoard objects",            /* tp_doc */
    0,                             /* tp_traverse */
    0,                             /* tp_clear */
    0,                             /* tp_richcompare */
    0,                             /* tp_weaklistoffset */
    0,                             /* tp_iter */
    0,                             /* tp_iternext */
    BitBoard_methods,              /* tp_methods */
    BitBoard_members,              /* tp_members */
    0,                             /* tp_getset */
    0,                             /* tp_base */
    0,                             /* tp_dict */
    0,                             /* tp_descr_get */
    0,                             /* tp_descr_set */
    0,                             /* tp_dictoffset */
    (initproc) BitBoard_init,      /* tp_init */
    0,                             /* tp_alloc */
    BitBoard_new,                  /* tp_new */
};


#endif //REVERSI_BITBOARD_H
