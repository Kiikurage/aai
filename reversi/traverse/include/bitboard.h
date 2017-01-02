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

const int EMPTY = 0b00;
const int BLACK = 0b01;
const int WHITE = 0b10;

typedef struct {
    PyObject_HEAD
    __m128 data;
} BitBoard;

static void BitBoard_dealloc(BitBoard *self) {
    //Nothing To Do.
}

static PyMethodDef BitBoard_methods[] = {
    {NULL}  // END Marker
};

static PyMemberDef BitBoard_members[] = {
    {NULL}  // END Marker
};

static int BitBoard_init(BitBoard *self, PyObject *args, PyObject *kwds) {
    PyArrayObject *board;

    static char *kwlist[] = {"board", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist, &board)) return -1;

    if (board) {
        int tmp[4] __attribute__((aligned(16))) = {0, 0, 0, 0};

        for (int x = 0; x < 8; x++) {
            for (int y = 0; y < 8; y++) {
                if (*(char *) PyArray_GETPTR3(board, 0, x, y)) {
                    BitSet(tmp, x, y, BLACK);

                } else if (*(char *) PyArray_GETPTR3(board, 1, x, y)) {
                    BitSet(tmp, x, y, WHITE);
                }
            }
        }

        self->data = _mm_load_ps((const float *) tmp);
    }

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
