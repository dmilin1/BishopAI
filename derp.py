import chess
import chess.syzygy
import random


def solve(board, tablebase):

    moves = [move for move in board.legal_moves]
    minMoves = 100000 if board.turn else -100000
    bestMoves = []
    for move in moves:
        newBoard = board.copy(stack=False)
        newBoard.push(move)
        score = -tablebase.get_dtz(newBoard)
        print(str(move) + ' ' + str(score))
        if board.turn:
            if score < minMoves and score > 0:
                minMoves = score
                bestMoves = [move]
            if score == minMoves:
                bestMoves.append(move)
        else:
            if score > minMoves and score < 0:
                minMoves = score
                bestMoves = [move]
            if score == minMoves:
                bestMoves.append(move)

    print('Moves to Win:' + str(minMoves))
    print('Best Moves:' + str(bestMoves))
    return bestMoves

    def alphaBeta(alpha, beta, depthleft, board):
        if depthleft == 0:
            return [tablebase.get_dtz(board), None]
        for move in board.legal_moves:
            newBoard = board.copy(stack=False)
            newBoard.push(move)
            score = [-alphaBeta([-beta[0], beta[1]], [-alpha[0], alpha[1]], depthleft - 1, newBoard)[0], move];
            if score[0] >= beta[0]:
                return beta
            if score[0] > alpha[0]:
                alpha = score
        return alpha

    movesToWin, bestMove = alphaBeta( [-100000, None], [100000, None], 4, board)
    print('Moves to Win:' + str(tablebase.get_dtz(board)))
    print('Best Move:' + str(bestMove))
    return bestMove


with chess.syzygy.open_tablebase("./syzygy/") as tablebase:
    board = chess.Board(fen = '8/5k2/8/K7/P3p3/8/1r4B1/8 b - - 1 39')
    print(tablebase.get_dtz(board))
    print(board)
    solve(board, tablebase)
