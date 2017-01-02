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

typedef char Color;
const Color EMPTY = 0b00;
const Color BLACK = 0b01;
const Color WHITE = 0b10;

typedef struct {
    PyObject_HEAD
    __m128 data;
} BitBoard;

static void BitBoard_dealloc(BitBoard *self) {
    //Nothing To Do.
}

static PyObject *BitBoard_to_board(BitBoard *self) {
    PyArrayObject *result = (PyArrayObject *) PyArray_SimpleNew(3, ((npy_intp[3]) {2, 8, 8}), NPY_BOOL);
    PyArray_FILLWBYTE(result, NPY_FALSE);

    for (int x = 0; x < 8; x++) {
        for (int y = 0; y < 8; y++) {
            if (BitCheck(&self->data, x, y, BLACK)) {
                *(npy_bool *) PyArray_GETPTR3(result, 0, x, y) = NPY_TRUE;

            } else if (BitCheck(&self->data, x, y, WHITE)) {
                *(npy_bool *) PyArray_GETPTR3(result, 1, x, y) = NPY_TRUE;
            }
        }
    }

    return (PyObject *) result;
};

static __m128 put_and_flip(__m128 board, const Color color, const int x, const int y) {
    // TODO bit演算で。

    __m128 result = board;
    const int other = color == BLACK ? WHITE : BLACK;

    if (!BitCheck(&result, x, y, EMPTY)) return result;
    BitFSet(&result, x, y, color);

    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            if (dx == 0 && dy == 0) continue;

            int px = x + dx;
            int py = y + dy;
            if (px < 0 || px >= 8 || py < 0 || py >= 8 || !BitCheck(&result, px, py, other)) continue;

            while (1) {
                px += dx;
                py += dy;

                if (px < 0 || px >= 8 || py < 0 || py >= 8) break;

                if (!BitCheck(&result, px, py, other)) {
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

static __m64 find_next(__m128 board, const Color color) {
    // TODO bit演算で。

    __m64 result = _mm_setzero_si64();
    const int other = color == BLACK ? WHITE : BLACK;

    for (int x = 0; x < 8; x++) {
        for (int y = 0; y < 8; y++) {
            if (!BitCheck(&board, x, y, EMPTY)) continue;
            char success = 0;

            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    if (dx == 0 && dy == 0) continue;

                    int px = x + dx;
                    int py = y + dy;
                    if (px < 0 || px >= 8 || py < 0 || py >= 8 || !BitCheck(&board, px, py, other)) continue;

                    while (1) {
                        px += dx;
                        py += dy;

                        if (px < 0 || px >= 8 || py < 0 || py >= 8) break;

                        if (!BitCheck(&board, px, py, other)) {
                            if (BitCheck(&board, px, py, color)) {
                                success = 1;
                                BitFSet8(&result, x * 8 + y, 1, 0b1);
                            }
                            break;
                        }
                    }

                    if (success) break;
                }
                if (success) break;
            }
        }
    }

    return result;
};

static PyMethodDef BitBoard_methods[] = {
    {"to_board", (PyCFunction) BitBoard_to_board, METH_NOARGS,
        "to_board()\n"
            "--\n"
            "\n"
            "convert BitBoard into normal Board\n"
            "\n"
            "Parameters\n"
            "----------\n"
            "None\n"
            "\n"
            "Returns\n"
            "-------\n"
            "board : Board\n"
            "  board object"
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
        for (int x = 0; x < 8; x++) {
            for (int y = 0; y < 8; y++) {
                if (*(npy_bool *) PyArray_GETPTR3(board, 0, x, y)) {
                    BitFSet(&self->data, x, y, BLACK);

                } else if (*(npy_bool *) PyArray_GETPTR3(board, 1, x, y)) {
                    BitFSet(&self->data, x, y, WHITE);
                }
            }
        }
    }

    self->data = put_and_flip(self->data, BLACK, 2, 3);

    return 0;
}

static PyObject *BitBoard_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    BitBoard *self;

    self = (BitBoard *) type->tp_alloc(type, 0);
    if (self != NULL) {
        self->data = _mm_setzero_ps();
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
