import re
import sys
import os.path as path

from board import Color, init_game, format_board, get_reversible_count, put, Board, is_valid
import numpy as np


#input file name
file_name = '01E4.gam.1.new'

#game_index=0

with open('../train_data/kifu/'+file_name) as f:
    lines = f.readlines()


if re.search('\s.\d\s', lines[game_index]) != None:
    game = re.split('\s.\d\s',lines[game_index])[0]
elif re.search('\s\d\s', lines[game_index]) != None:
    game = re.split('\s\d\s',lines[game_index])[0]

print(len(game))
def alphabet2num(alphabet):
    array = ['A','B','C','D','E','F','G','H']
    return array.index(alphabet)

pass_cnt = 0
c = Color.Black
b = init_game()
count=0
length = len(game)
while count < length:
    if game[count]==' ':
        pass_cnt += 1
        print('{0}: pass'.format('Black' if c == Color.Black else 'White'))
        print('')
        count +=1

    else:
        pass_cnt = 0
        x,y = (alphabet2num(game[count]),int(game[count+1])-1)
        b = put(b, c, x,y)
        print(count)
        print('{0}: ({1}, {2})'.format('Black' if c == Color.Black else 'White', x, y))
        print(format_board(b))
        print('')
        count +=2
    #print(count)

    c = 1 - c
