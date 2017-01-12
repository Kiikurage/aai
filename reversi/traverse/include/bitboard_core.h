#ifndef REVERSI_BITBOARD_H
#define REVERSI_BITBOARD_H

#include <Python.h>
#include <numpy/ndarraytypes.h>
#include <numpy/ndarrayobject.h>
#include <x86intrin.h>
#include <structmember.h>
#include <macro.h>
#include "bitboard.h"

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

static PyObject *BitBoard_find_next(BitBoard *self, PyObject *args, PyObject *keywds) {
    int buf_x[64];
    int buf_y[64];
    int num_hands;

    const Color start_color;
    if (!PyArg_ParseTuple(args, "i", &start_color)) return NULL;

    find_next(self->data, start_color, buf_x, buf_y, &num_hands);
    PyObject *res = PyTuple_New(num_hands);

    for (int i = 0; i < num_hands; i++) {
        PyTuple_SET_ITEM(res, i, Py_BuildValue("(ii)", buf_x[i], buf_y[i]));
    }

    return res;
}

static PyObject *BitBoard_montecalro(BitBoard *self, PyObject *args, PyObject *keywds) {
    const Color start_color;
    const int num_branch;
    const MontecalroMode mode = MONTECALRO_MODE_WIN;
    static char *kwlist[] = {"color", "num_branch", "mode", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "ii|i", kwlist, &start_color, &num_branch, &mode)) return NULL;

    MontecarloResult result = montecalro(self->data, start_color, num_branch, mode);

    return Py_BuildValue("(iii)", result.best_x, result.best_y, result.best_win_count);
}

static PyObject *BitBoard_get_score_prob(BitBoard *self, PyObject *args) {
    const Color start_color;
    int num_branch = 1000;
    if (!PyArg_ParseTuple(args, "ii", &start_color, &num_branch)) return NULL;

    float prob[127];

    get_score_prob(self->data, prob, start_color, num_branch);
    PyObject *res = PyTuple_New(127);

    for (int i = 0; i < 127; i++) {
        PyTuple_SET_ITEM(res, i, PyFloat_FromDouble(prob[i]));
    }

    return res;
}

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
    {"find_next", (PyCFunction) BitBoard_find_next, METH_VARARGS,
        "find_next(color)\n"
            "--\n"
            "\n"
            "有効な手を調べます。\n"
            "\n"
            "Parameters\n"
            "----------\n"
            "color: int\n"
            "  指す色。0が黒、1が白を表す。\n"
            "\n"
            "Returns\n"
            "-------\n"
            "hands: Tuple(int, int)\n"
            "  させる手のタプル\n"
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
    {"get_score_prob", (PyCFunction) BitBoard_get_score_prob, METH_VARARGS,
        "get_score_prob(color, num_branch, mode=1)\n"
            "--\n"
            "\n"
            "モンテカルロ探索により、与えられた盤面から終局状態を予想します。\n"
            "\n"
            "Parameters\n"
            "----------\n"
            "color: int\n"
            "  最初に指す色。0が黒、1が白を表す。\n"
            "\n"
            "Returns\n"
            "-------\n"
            "prob : Tuple(int, int...)\n"
            "  確率。長さ127のタプルで、石数-64~64の終局状態の確率をそれぞれ表している。\n"
    },
    {NULL}  // END Marker
};

static PyMemberDef BitBoard_members[] = {
    {NULL}  // END Marker
};

static int BitBoard_init(BitBoard *self, PyObject *args) {
    PyArrayObject *board = NULL;

    if (!PyArg_ParseTuple(args, "O", &board)) return -1;

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
    BitBoard *self = (BitBoard *) type->tp_alloc(type, 0);

    if (self != NULL) {
        int *p_data = (int *) (&self->data);
        for (int i = 0; i < 4; i++) p_data[i] = 0xFFFFFFFF;
    }

    return (PyObject *) self;
}

static void BitBoard_dealloc(BitBoard *self) {
    //Nothing To Do.
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
