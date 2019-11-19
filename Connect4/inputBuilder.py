import chess
import chess.pgn
import os
import numpy as np
from connect4 import Connect4

cwd = os.getcwd()

def buildInput(board):

	oldBoard = board.copy(stack=1)
	try:
		oldBoard.pop()
	except IndexError:
		oldBoard.clear()

	if board.turn:
		currentBoard = board
		flipped = False
	else:
		flipped = True
		currentBoard = board.mirror()
		oldBoard = oldBoard.mirror()

	def buildBoard(board):
		pawns = np.zeros((8,8))
		knights = np.zeros((8,8))
		bishops = np.zeros((8,8))
		rooks = np.zeros((8,8))
		queens = np.zeros((8,8))
		kings = np.zeros((8,8))

		for square in chess.SQUARES:
			x = square // 8
			y = square % 8
			piece = board.piece_at(square)
			if piece:
				pieceType = piece.piece_type
				if piece.color == chess.WHITE:
					if pieceType == chess.PAWN:
						pawns[x][y] = 1
					if pieceType == chess.KNIGHT:
						knights[x][y] = 1
					if pieceType == chess.BISHOP:
						bishops[x][y] = 1
					if pieceType == chess.ROOK:
						rooks[x][y] = 1
					if pieceType == chess.QUEEN:
						queens[x][y] = 1
					if pieceType == chess.KING:
						kings[x][y] = 1
				else:
					if pieceType == chess.PAWN:
						pawns[x][y] = -1
					if pieceType == chess.KNIGHT:
						knights[x][y] = -1
					if pieceType == chess.BISHOP:
						bishops[x][y] = -1
					if pieceType == chess.ROOK:
						rooks[x][y] = -1
					if pieceType == chess.QUEEN:
						queens[x][y] = -1
					if pieceType == chess.KING:
						kings[x][y] = -1

		return np.array([
			pawns,
			knights,
			bishops,
			rooks,
			queens,
			kings,
		])

	turn = np.ones((8,8)) if not flipped else np.zeros((8,8))
	wCastleQ = np.ones((8,8)) if currentBoard.has_kingside_castling_rights(chess.WHITE) else np.zeros((8,8))
	wCastleK = np.ones((8,8)) if currentBoard.has_queenside_castling_rights(chess.WHITE) else np.zeros((8,8))
	bCastleQ = np.ones((8,8)) if currentBoard.has_kingside_castling_rights(chess.BLACK) else np.zeros((8,8))
	bCastleK = np.ones((8,8)) if currentBoard.has_queenside_castling_rights(chess.BLACK) else np.zeros((8,8))
	fiftyMove = np.full((8,8), currentBoard.halfmove_clock/50)
	ones = np.ones((8,8))
	# if len(currentBoard.move_stack) > 2 and currentBoard.move_stack[-1].from_square == currentBoard.move_stack[-3].to_square and currentBoard.move_stack[-1].to_square == currentBoard.move_stack[-3].from_square:
	# 	repetition = np.ones((8,8))
	# else:
	# 	repetition = np.zeros((8,8))


	return np.array([
		*buildBoard(board),
		*buildBoard(oldBoard),
		turn,
		wCastleQ,
		wCastleK,
		bCastleQ,
		bCastleK,
		fiftyMove,
		ones,
		# repetition,
	])

def reverseInput(board):
	wPawns, wKnights, wBishops, wRooks, wQueens, wKings, bPawns, bKnights, bBishops, bRooks, bQueens, bKings, turn = board

	board = chess.Board()
	board.clear()
	board.turn = chess.WHITE if turn[0][0] == 1 else chess.BLACK

	for x, row in enumerate(wPawns):
		for y, square in enumerate(row):
			if square == 1:
				board.set_piece_at(8*x+y, chess.Piece(chess.PAWN, chess.WHITE))

	for x, row in enumerate(wKnights):
		for y, square in enumerate(row):
			if square == 1:
				board.set_piece_at(8*x+y, chess.Piece(chess.KNIGHT, chess.WHITE))

	for x, row in enumerate(wBishops):
		for y, square in enumerate(row):
			if square == 1:
				board.set_piece_at(8*x+y, chess.Piece(chess.BISHOP, chess.WHITE))

	for x, row in enumerate(wRooks):
		for y, square in enumerate(row):
			if square == 1:
				board.set_piece_at(8*x+y, chess.Piece(chess.ROOK, chess.WHITE))

	for x, row in enumerate(wQueens):
		for y, square in enumerate(row):
			if square == 1:
				board.set_piece_at(8*x+y, chess.Piece(chess.QUEEN, chess.WHITE))

	for x, row in enumerate(wKings):
		for y, square in enumerate(row):
			if square == 1:
				board.set_piece_at(8*x+y, chess.Piece(chess.KING, chess.WHITE))

	for x, row in enumerate(bPawns):
		for y, square in enumerate(row):
			if square == 1:
				board.set_piece_at(8*x+y, chess.Piece(chess.PAWN, chess.BLACK))

	for x, row in enumerate(bKnights):
		for y, square in enumerate(row):
			if square == 1:
				board.set_piece_at(8*x+y, chess.Piece(chess.KNIGHT, chess.BLACK))

	for x, row in enumerate(bBishops):
		for y, square in enumerate(row):
			if square == 1:
				board.set_piece_at(8*x+y, chess.Piece(chess.BISHOP, chess.BLACK))

	for x, row in enumerate(bRooks):
		for y, square in enumerate(row):
			if square == 1:
				board.set_piece_at(8*x+y, chess.Piece(chess.ROOK, chess.BLACK))

	for x, row in enumerate(bQueens):
		for y, square in enumerate(row):
			if square == 1:
				board.set_piece_at(8*x+y, chess.Piece(chess.QUEEN, chess.BLACK))

	for x, row in enumerate(bKings):
		for y, square in enumerate(row):
			if square == 1:
				board.set_piece_at(8*x+y, chess.Piece(chess.KING, chess.BLACK))

	return board

def score(board):
    score = 0
    for square in chess.SQUARES:
        if board.piece_at(square) != None:
            multiplier = 1 if board.piece_at(square).color == chess.WHITE else -1
            if board.piece_type_at(square) == chess.PAWN:
                score += multiplier * 1
            if board.piece_type_at(square) == chess.KNIGHT:
                score += multiplier * 3
            if board.piece_type_at(square) == chess.BISHOP:
                score += multiplier * 3
            if board.piece_type_at(square) == chess.ROOK:
                score += multiplier * 5
            if board.piece_type_at(square) == chess.QUEEN:
                score += multiplier * 9
    return score

board = Connect4()
print(board)
