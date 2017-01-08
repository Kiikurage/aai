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

typedef __m128 BitBoardData;

typedef struct {
    PyObject_HEAD
    BitBoardData data;
} BitBoard;

static BitBoardData put_and_flip(BitBoardData board, const Color color, const int x, const int y) {
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

static void find_next(BitBoardData board, const Color color, int *buf_x, int *buf_y, int *n_valid_hands) {
    // TODO bit演算で。
    *n_valid_hands = 0;

    for (int i = 0; i < 64; i++) {
        const int x = i / 8;
        const int y = i % 8;
        if (!BitCheck(&board, x, y, EMPTY)) continue;

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

                    if (BitCheck(&board, px, py, EMPTY)) break;

                    if (BitCheck(&board, px, py, color)) {
                        buf_x[*n_valid_hands] = i / 8;
                        buf_y[*n_valid_hands] = i % 8;
                        (*n_valid_hands)++;

                        goto next;
                    }
                }
            }
        }

        next:
        continue;
    }
};

typedef struct {
    int black;
    int white;
    int empty;
} Summary;

static Summary summary(BitBoardData board) {
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

static float evaluate_function_mode_win(Color start_color, Summary s) { return start_color == BLACK ? s.black - s.white : s.white - s.black; }

static float evaluate_function_mode_lose(Color start_color, Summary s) { return start_color == WHITE ? s.black - s.white : s.white - s.black; }

static float evaluate_function_mode_draw(Color start_color, Summary s) { return -(s.white - s.black) * (s.white - s.black); }

typedef enum {
    MONTECALRO_MODE_WIN = 1,
    MONTECALRO_MODE_LOSE = 2,
    MONTECALRO_MODE_DRAW = 3,
} MontecalroMode;

typedef struct TraverseNode {
    BitBoardData data;
    int depth;
    struct TraverseNode *next;
    int pass_count;
    Color current_color;
} TraverseNode;

static PyObject *BitBoard_traverse(BitBoard *self, PyObject *args);

static PyObject *BitBoard_montecalro(BitBoard *self, PyObject *args, PyObject *keywds) {
    const Color start_color;
    const int num_branch;
    const MontecalroMode mode = MONTECALRO_MODE_WIN;
    static char *kwlist[] = {"color", "num_branch", "mode", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "ii|i", kwlist, &start_color, &num_branch, &mode)) return NULL;

    float (*evaluate_function)(Color, Summary);

    switch (mode) {
        case MONTECALRO_MODE_WIN:
            evaluate_function = &evaluate_function_mode_win;
            break;

        case MONTECALRO_MODE_LOSE:
            evaluate_function = &evaluate_function_mode_lose;
            break;

        case MONTECALRO_MODE_DRAW:
        default:
            evaluate_function = &evaluate_function_mode_draw;
            break;
    }

    int *buf_x = (int *) malloc(sizeof(int) * 64);
    int *buf_y = (int *) malloc(sizeof(int) * 64);
    int n_valid_hands = 0;
    find_next(self->data, start_color, buf_x, buf_y, &n_valid_hands);

    if (n_valid_hands == 0) return Py_BuildValue("(iii)", -1, -1, 0);

    int *buf_win_count = (int *) calloc(sizeof(int), (size_t) n_valid_hands);
    int *buf_x2 = (int *) malloc(sizeof(int) * 64);
    int *buf_y2 = (int *) malloc(sizeof(int) * 64);

    for (int i_hand = 0; i_hand < n_valid_hands; i_hand++) {

        for (int i_branch = 0; i_branch < num_branch; i_branch++) {
            BitBoardData board = put_and_flip(self->data, start_color, buf_x[i_hand], buf_y[i_hand]);
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
            buf_win_count[i_hand] += evaluate_function(start_color, s);
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
    {"montecarlo", (PyCFunction) BitBoard_montecalro, METH_VARARGS | METH_KEYWORDS,
        "montecarlo(color, num_branch, mode=1)\n"
            "--\n"
            "\n"
            "モンテカルロ探索により、最善手を探索します。\n"
            "\n"
            "Parameters\n"
            "----------\n"
            "color: int\n"
            "  最初に指す色。0が黒、1が白を表す。\n"
            "num_branch: int\n"
            "  各有効手に対する読み切り回数。例えばnum_branch=1000の場合、各有効手について1000回、ゲーム終了までの試行を行います。\n"
            "mode: int\n"
            "  目指す終局状態。mode=1: 勝ち, mode=2: 負け, mode=3: 引き分け"
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
    {"traverse",   (PyCFunction) BitBoard_traverse,   METH_VARARGS,
        "traverse(color, depth)\n"
            "--\n"
            "\n"
            "指定された深さだけ全探索を行い、取りうる手を返します。\n"
            "\n"
            "Parameters\n"
            "----------\n"
            "color: int\n"
            "  最初に指す色。0が黒、1が白を表す。\n"
            "depth: int\n"
            "  読みの深さ\n"
            "\n"
            "Returns\n"
            "-------\n"
            "bitboards : Tuole of BitBoard\n"
            "  読み終わった状態のBitBoardの配列\n"
    },
    {NULL}  // END Marker
};

static PyMemberDef BitBoard_members[] = {
    {NULL}  // END Marker
};

static int BitBoard_init(BitBoard *self, PyObject *args) {
    PyArrayObject *board = NULL;

    if (!PyArg_ParseTuple(args, "|O", &board)) return -1;

    if (board != NULL) {
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

static PyObject *BitBoard_traverse(BitBoard *self, PyObject *args) {
    const Color start_color;
    const int max_depth;
    if (!PyArg_ParseTuple(args, "ii", &start_color, &max_depth)) return NULL;

    TraverseNode *cursor = malloc(sizeof(TraverseNode));
    cursor->data = self->data;
    cursor->depth = 0;
    cursor->next = 0;
    cursor->pass_count = 0;
    cursor->current_color = start_color;
    TraverseNode *end = cursor;

    int buf_size = 1024;
    BitBoardData *bitboards = malloc(sizeof(BitBoardData) * buf_size);
    int num_bitboard = 0;

    int n_valid_hands = 0;
    int *buf_x = (int *) malloc(sizeof(int) * 64);
    int *buf_y = (int *) malloc(sizeof(int) * 64);

    do {
        if (cursor->depth >= max_depth) {
            //打ち切り
            bitboards[num_bitboard] = cursor->data;
            num_bitboard++;
            if (num_bitboard >= buf_size) {
                BitBoardData *tmp = malloc(sizeof(BitBoardData) * buf_size * 2);
                memcpy(tmp, bitboards, sizeof(BitBoardData) * buf_size);
                free(bitboards);
                bitboards = tmp;
                buf_size *= 2;
            }

        } else {
            find_next(cursor->data, cursor->current_color, buf_x, buf_y, &n_valid_hands);
            if (n_valid_hands == 0) {
                //有効手無し

                if (cursor->pass_count == 1) {
                    //２連続パス＝終了
                    bitboards[num_bitboard] = cursor->data;
                    num_bitboard++;
                    if (num_bitboard >= buf_size) {
                        BitBoardData *tmp = malloc(sizeof(BitBoardData) * buf_size * 2);
                        memcpy(tmp, bitboards, sizeof(BitBoardData) * buf_size);
                        free(bitboards);
                        bitboards = tmp;
                        buf_size *= 2;
                    }

                } else {
                    //普通のパス
                    end->next = (TraverseNode *) malloc(sizeof(TraverseNode));
                    end = end->next;

                    end->data = cursor->data;
                    end->depth = cursor->depth + 1;
                    end->next = 0;
                    end->pass_count = 1;
                    end->current_color = other(cursor->current_color);
                }
            } else {
                //有効手あり　次の局面の列挙
                for (int i = 0; i < n_valid_hands; i++) {
                    end->next = (TraverseNode *) malloc(sizeof(TraverseNode));
                    end = end->next;

                    end->data = put_and_flip(cursor->data, cursor->current_color, buf_x[i], buf_y[i]);
                    end->depth = cursor->depth + 1;
                    end->next = 0;
                    end->pass_count = 0;
                    end->current_color = other(cursor->current_color);
                }
            }
        }

        TraverseNode *tmp = cursor->next;
        free(cursor);
        cursor = tmp;
    } while (cursor != 0);

    free(buf_x);
    free(buf_y);

    PyObject *tuple = PyTuple_New(num_bitboard);
    for (int i = 0; i < num_bitboard; i++) {
        BitBoard *bb = (BitBoard *) PyObject_CallFunction((PyObject *) &BitBoard_Type, NULL);
        bb->data = bitboards[i];
        PyTuple_SetItem(tuple, i, (PyObject *) bb);
    }

    free(bitboards);

    return tuple;
};
#endif //REVERSI_BITBOARD_H
