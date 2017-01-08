from reversi import traverse, board, Color

results = traverse.BitBoard(board.init()).traverse(Color.Black, 10)
	
print(len(results))
for bb in results[:10]:
	print(board.stringify(bb.to_board()))
