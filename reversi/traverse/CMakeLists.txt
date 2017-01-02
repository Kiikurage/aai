#
# CLion (IntelliJ のC言語用IDE) で補完を正常に効かせるためのダミーファイル。
# 実際のビルド設定はsetup.pyで行う。
#

cmake_minimum_required(VERSION 3.6)
project(reversi)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -std=c11 -Wall -march=native")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DDEBUG")

##------------------------------------------------------------------------------------------------------------------------
## include

set(PYTHON_DIR $ENV{PYENV_ROOT}/versions/anaconda3-4.1.1)

#python
include_directories("${PYTHON_DIR}/lib/python3.5/site-packages/numpy/core/include")

#numpy
include_directories("${PYTHON_DIR}/include/python3.5m")

include_directories("./include")
add_executable(.
        src/traverse.c include/traverse.h
        include/bitboard.h
        include/macro.h include/xorshift.h)
