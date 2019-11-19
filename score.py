import chess
import time
import random

defaultBoard = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

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


def pickMove(board):
    moves = [move for move in board.legal_moves]
    for move in moves:
        newBoard = board.copy()
        newBoard.push(move)
        theScore = score(newBoard)
        move.score = theScore

    multiplier = 1 if board.turn == chess.WHITE else -1
    bestScore = max(move.score*multiplier for move in moves)
    bestMoves = [move for move in moves if move.score == bestScore*multiplier]
    chosenMove = random.choice(bestMoves)
    return chosenMove


board = chess.Board(fen = defaultBoard)

while not board.is_game_over():
    print("~~~~~")
    print(score(board))
    print(board)
    print("~~~~~")
    board.push(pickMove(board))
    # time.sleep(1)
