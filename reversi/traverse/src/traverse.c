//
// Created by kikura on 2017/01/02.
//

#include <traverse.h>

static PyMethodDef methods[] = {
//    {"create_bitboard", create_bitboard, METH_VARARGS, doc_create_bitboard},
    {NULL, NULL, 0, NULL},  // END marker
};

static struct PyModuleDef Traverse = {PyModuleDef_HEAD_INIT, "traverse", "", -1, methods, NULL, NULL, NULL, NULL};

PyMODINIT_FUNC PyInit_traverse(void) {
    BitBoard_Type.tp_new = PyType_GenericNew;
    if (PyType_Ready(&BitBoard_Type) < 0)
        return NULL;

    PyObject *module = PyModule_Create(&Traverse);
    if (module == NULL)
        return NULL;

    Py_INCREF(&BitBoard_Type);
    PyModule_AddObject(module, "BitBoard", (PyObject *) &BitBoard_Type);

    import_array();

    return module;
}