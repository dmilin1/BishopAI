import chess
import chess.syzygy
import random
import time
import queue
from Tree import Tree
from MonteCarloTree import MCTree

scoreTotal = 0
depthTotal = 0

totalTime = 0





pawnTableBlack = ([
    99, 99, 99, 99, 99, 99, 99, 99,
    36, 36, 36, 36, 36, 36, 36, 36,
    14, 14, 20, 24, 24, 20, 16, 16,
    10, 13, 13, 20, 20, 18, 12, 12,
     6, 12, 12, 20, 20, 10, 10, 10,
    12, 12, 10, 12, 12, 10, 55, 50,
    10, 10, 10,  0,  0, 14, 55, 50,
     0,  0,  0,  0,  0,  0,  0,  0,
])

pawnTableWhite = ([
     0,  0,  0,  0,  0,  0,  0,  0,
    10, 10, 10,  0,  0, 14, 55, 50,
    12, 12, 10, 12, 12, 10, 55, 50,
     6, 12, 12, 20, 20, 10, 10, 10,
    10, 13, 13, 20, 20, 18, 12, 12,
    14, 14, 20, 24, 24, 20, 16, 16,
    36, 36, 36, 36, 36, 36, 36, 36,
    99, 99, 99, 99, 99, 99, 99, 99,
])



knightTableBlack = ([
     0, 10, 20, 20, 20, 20, 10,  0,
    10, 24, 26, 26, 26, 26, 24, 10,
    10, 28, 40, 50, 50, 40, 28, 10,
    10, 23, 36, 40, 40, 36, 23, 10,
    10, 22, 28, 30, 30, 28, 22, 10,
     0, 26, 26, 30, 30, 26, 26,  0,
     5, 20, 20, 23, 20, 20, 20,  5,
     0,  5, 15, 15, 15, 15,  0,  0,
])

knightTableWhite = ([
     0,  5, 15, 15, 15, 15,  0,  0,
     5, 20, 20, 23, 20, 20, 20,  5,
     0, 26, 26, 30, 30, 26, 26,  0,
    10, 22, 28, 30, 30, 28, 22, 10,
    10, 23, 36, 40, 40, 36, 23, 10,
    10, 28, 40, 50, 50, 40, 28, 10,
    10, 24, 26, 26, 26, 26, 24, 10,
     0, 10, 20, 20, 20, 20, 10,  0,
])



bishopTableBlack = ([
    10, 10, 10, 10, 10, 10, 10, 10,
     0, 22, 24, 24, 24, 24, 22,  0,
    42, 48, 50, 52, 52, 50, 48, 42,
    41, 42, 45, 48, 48, 45, 42, 41,
    38, 45, 42, 48, 48, 42, 45, 38,
    35, 45, 45, 42, 42, 45, 45, 35,
    30, 48, 42, 35, 35, 42, 48, 30,
    24, 30, 24, 30, 30, 20, 30, 24,
])

bishopTableWhite = ([
    24, 30, 24, 30, 30, 20, 30, 24,
    30, 48, 42, 35, 35, 42, 48, 30,
    35, 45, 45, 42, 42, 45, 45, 35,
    38, 45, 42, 48, 48, 42, 45, 38,
    41, 42, 45, 48, 48, 45, 42, 41,
    42, 48, 50, 52, 52, 50, 48, 42,
     0, 22, 24, 24, 24, 24, 22,  0,
    10, 10, 10, 10, 10, 10, 10, 10,
])



rookTableBlack = ([
    30, 30, 30, 30, 30, 30, 30, 30,
    50, 50, 50, 50, 50, 50, 50, 50,
     8,  8, 12, 20, 20, 12,  8,  8,
     0,  0,  4,  6,  6,  4,  0,  0,
     0,  0,  4,  6,  6,  4,  0,  0,
     0,  0,  4,  6,  6,  4,  0,  0,
     0,  0,  4,  6,  6,  4,  0,  0,
     0,  0,  4,  6,  6,  4,  0,  0,
])

rookTableWhite = ([
     0,  0,  4,  6,  6,  4,  0,  0,
     0,  0,  4,  6,  6,  4,  0,  0,
     0,  0,  4,  6,  6,  4,  0,  0,
     0,  0,  4,  6,  6,  4,  0,  0,
     0,  0,  4,  6,  6,  4,  0,  0,
     8,  8, 12, 20, 20, 12,  8,  8,
    50, 50, 50, 50, 50, 50, 50, 50,
    30, 30, 30, 30, 30, 30, 30, 30,
])



queenTableBlack = ([
    38, 46, 50, 54, 54, 50, 46, 42,
    38, 46, 54, 60, 60, 54, 54, 50,
    34, 42, 46, 60, 60, 50, 50, 46,
    32, 36, 34, 34, 34, 42, 40, 40,
    32, 34, 34, 34, 34, 38, 36, 36,
    32, 34, 34, 38, 38, 36, 34, 34,
    32, 32, 36, 18, 36, 32, 32, 32,
    20, 30, 30, 30, 30, 30, 30, 20,
])

queenTableWhite = ([
    20, 30, 30, 30, 30, 30, 30, 20,
    32, 32, 36, 18, 36, 32, 32, 32,
    32, 34, 34, 38, 38, 36, 34, 34,
    32, 34, 34, 34, 34, 38, 36, 36,
    32, 36, 34, 34, 34, 42, 40, 40,
    34, 42, 46, 60, 60, 50, 50, 46,
    38, 46, 54, 60, 60, 54, 54, 50,
    38, 46, 50, 54, 54, 50, 46, 42,
])



kingTableBlack = ([
     0,  0,  0,  0,  0,  0,  0,  0,
    10, 10, 10, 10, 10, 10, 10, 10,
    20, 20, 20, 20, 20, 20, 20, 20,
    35, 35, 35, 35, 35, 35, 35, 35,
    50, 50, 50, 50, 50, 50, 50, 50,
    60, 60, 60, 60, 60, 60, 60, 60,
    86, 90, 84, 80, 80, 80, 88, 88,
    88, 92, 90, 80, 90, 80, 90, 88,
])

kingTableWhite = ([
    88, 92, 90, 80, 90, 80, 90, 88,
    86, 90, 84, 80, 80, 80, 88, 88,
    60, 60, 60, 60, 60, 60, 60, 60,
    50, 50, 50, 50, 50, 50, 50, 50,
    35, 35, 35, 35, 35, 35, 35, 35,
    20, 20, 20, 20, 20, 20, 20, 20,
    10, 10, 10, 10, 10, 10, 10, 10,
     0,  0,  0,  0,  0,  0,  0,  0,
])





class BishopAI:

    def __init__(self):
        self.board = chess.Board()
        print('\n~~~~~~\n')
        print(self.board)

    def catchUp(self, moveList):
        if moveList != '':
            for move in moveList.split(" "):
                self.board.push_uci(move)
            print('\n~~~~~~\n')
            print(self.board)

    def whosTurn(self):
        return self.board.turn

    def movePlayed(self, move):
        if move != None:
            self.board.push_uci(move)

    def solve(self, board, tablebase):

        moves = [move for move in board.legal_moves]
        minMoves = 100000 if board.turn else -100000
        bestMoves = []
        for move in moves:
            newBoard = board.copy(stack=False)
            newBoard.push(move)
            score = -tablebase.get_dtz(newBoard)
            print(str(move) + ' ' + str(score))
            if board.turn:
                if score == minMoves:
                    bestMoves.append(move)
                if score < minMoves and score > 0:
                    minMoves = score
                    bestMoves = [move]
            else:
                if score == minMoves:
                    bestMoves.append(move)
                if score > minMoves and score < 0:
                    minMoves = score
                    bestMoves = [move]

        print('Moves to Win:' + str(minMoves))
        print('Best Moves:' + str(bestMoves))
        return bestMoves



    def getMove(self):
        print(self.board.fen())

        solvableMoves = None

        with chess.syzygy.open_tablebase("./syzygy/") as tablebase:
            if tablebase.get_wdl(self.board) != None:
                winCertainty = tablebase.get_wdl(self.board)
                print('Win Certainty: ' + str(winCertainty))
                if winCertainty == 2 * ( -1 if self.board.turn else 1):
                    return None
                if winCertainty > -2 and winCertainty < 2:
                    [self.solve(self.board, tablebase), 'draw']

                solvableMoves = self.solve(self.board, tablebase)


        global scoreTotal
        scoreTotal = 0
        start = time.time()
        if solvableMoves:
            newMove = self.pickMove(self.board, solvableMoves=solvableMoves)
        else:
            newMove = self.pickMove(self.board)
        end = time.time()
        print(scoreTotal)
        print(end-start)
        print(self.board)

        # return 'THROW ERROR'
        if newMove:
            return newMove.uci()
        else:
            return None


    def score(self, board):
        score = 0
        if board.is_checkmate():
            score += -10000 if board.turn else 10000
        if board.is_check():
            score += -25 if board.turn else 25
        for square in chess.SQUARES:
            score += self.pieceScore(board.piece_at(square), square)
        global scoreTotal
        scoreTotal += 1
        if scoreTotal % 1000 == 0:
            print(scoreTotal)
        return score

    def pieceScore(self, piece, square):
        if piece == None:
            return 0
        if piece.piece_type == chess.PAWN:
            return (1 if piece.color == chess.WHITE else -1) * (100
                + (pawnTableWhite[square]
                  if piece.color == chess.WHITE else
                  pawnTableBlack[square]))
        elif piece.piece_type == chess.KNIGHT:
            return (1 if piece.color == chess.WHITE else -1) * (300
                + (knightTableWhite[square]
                  if piece.color == chess.WHITE else
                  knightTableBlack[square]))
        elif piece.piece_type == chess.BISHOP:
            return (1 if piece.color == chess.WHITE else -1) * (300
                + (bishopTableWhite[square]
                  if piece.color == chess.WHITE else
                  bishopTableBlack[square]))
        elif piece.piece_type == chess.ROOK:
            return (1 if piece.color == chess.WHITE else -1) * (500
                + (rookTableWhite[square]
                  if piece.color == chess.WHITE else
                  rookTableBlack[square]))
        elif piece.piece_type == chess.QUEEN:
            return (1 if piece.color == chess.WHITE else -1) * (900
                + (queenTableWhite[square]
                  if piece.color == chess.WHITE else
                  queenTableBlack[square]))
        elif piece.piece_type == chess.KING:
            return ((1 if piece.color == chess.WHITE else -1) *
                  (kingTableWhite[square]
                  if piece.color == chess.WHITE else
                  kingTableBlack[square]))

    def rescore(self, oldBoard, newBoard, move, oldScore):
        if newBoard.is_checkmate():
            return -10000 if newBoard.turn else 10000
        newScore = oldScore
        oldPosition = move.from_square
        newPosition = move.to_square
        newScore -= self.pieceScore(oldBoard.piece_at(oldPosition), oldPosition)
        newScore += self.pieceScore(newBoard.piece_at(newPosition), newPosition)
        newScore -= self.pieceScore(oldBoard.piece_at(newPosition), newPosition)
        if newBoard.is_check():
            newScore += -25 if newBoard.turn else 25
        global scoreTotal
        scoreTotal += 1
        if scoreTotal % 1000 == 0:
            print(scoreTotal)
        return newScore


    def pickMove(self, board, depth=5, solvableMoves=None):
        print("thinking")
        print(board)
        global totalTime

        def minimax(maximize, layer, currentDepth=1, parent=None):

            def maxOrMin(bool, input):
                return max(input) if bool else min(input)

            def getExtrema(shouldMax, layer):
                extremaVal = maxOrMin(shouldMax,[item[2] for item in layer])
                matchingItems = []
                for item in layer:
                    if item[2] == extremaVal:
                        if parent != None:
                            item[1] = parent[1]
                        matchingItems.append(item)
                return random.choice(matchingItems)


            if currentDepth == depth:
                return getExtrema(maximize, layer)
            return getExtrema(maximize, [minimax(not maximize, item[3], currentDepth+1, item) if item[3] != [] else item for item in layer])

        def alphaBetaMax(board, depthLeft, alpha=[-100000, None], beta=[100000, None], parentMove=None):
            if depthLeft == 0:
                return [self.score(board), parentMove]
            moves = [move for move in board.legal_moves]
            for move in moves:
                newBoard = board.copy()
                newBoard.push(move)
                score = alphaBetaMin(newBoard, depthLeft - 1, alpha, beta, parentMove if parentMove != None else move)
                if score[0] >= beta[0]:
                    return beta
                if score[0] > alpha[0]:
                    alpha = score
            return alpha

        def alphaBetaMin(board, depthLeft, alpha=[-100000, None], beta=[100000, None], parentMove=None):
            if depthLeft == 0:
                return [-self.score(board), parentMove]
            moves = [move for move in board.legal_moves]
            for move in moves:
                newBoard = board.copy()
                newBoard.push(move)
                score = alphaBetaMax(newBoard, depthLeft - 1, alpha, beta, parentMove if parentMove != None else move)
                if score[0] <= alpha[0]:
                    return alpha
                if score[0] < beta[0]:
                    beta = score
            return beta

        # tree = self.getTree(board, depth)
        # return minimax(board.turn, tree)[1]

        def customTree(board, maxTime=12, maxDepth=4, solvableMoves=None):
            treeRoot = Tree(score = self.score(board))
            start = time.time()
            timeUsed = 0
            currentTree = treeRoot
            currentBoard = board
            q = queue.Queue()
            global totalTime

            quiescentSearch = False

            while timeUsed < maxTime or quiescentSearch:
                if solvableMoves:
                    moves = solvableMoves
                else:
                    moves = [move for move in currentBoard.legal_moves]
                # if currentTree.depth >= maxDepth:
                #     break
                for move in moves:
                    newBoard = currentBoard.copy(stack=2)
                    newBoard.push(move)
                    newTree = currentTree.append(self.rescore(currentBoard, newBoard, move, currentTree.score), move)
                    testStart = time.time()
                    newTree.isCapture = currentBoard.is_capture(move)
                    totalTime += time.time() - testStart
                    if timeUsed < maxTime:
                        q.put([newBoard, newTree])
                currentBoard, currentTree = q.get()

                if quiescentSearch and timeUsed >= maxTime:
                    while True:
                        if not currentTree.isCapture:
                            try:
                                currentBoard, currentTree = q.get(block=False)
                            except:
                                quiescentSearch = False
                                break
                        else:
                            break

                timeUsed = time.time() - start



            print('Depth: ' + str(treeRoot.getDepth()))
            if self.whosTurn():
                bestMoves = treeRoot.getMax()
            else:
                bestMoves = treeRoot.getMin()
            if len(bestMoves) == 0:
                raise Exception("wtf")
                # return None
            print("Total Time: " + str(totalTime))
            move = random.choice(bestMoves)
            print('Move: ' + str(move))
            return move

        def monteCarlo(board, maxTime=15, maxDepth=5):
            treeRoot = MCTree(board)
            start = time.time()
            initialScore = self.score(treeRoot.board)
            global totalTime
            simCount = 0

            def simulate(board, initialScore, depth = 0):
                global totalTime
                if depth >= maxDepth:
                    score = self.score(board)
                    return 0.5 + ((score - initialScore) / (10 * depth))
                moves = [move for move in board.legal_moves]
                try:
                    board.push(random.choice(moves))
                except:
                    result = board.result().split('-')[0]
                    if result == '1/2':
                        return 0.5
                    return int(result)
                return simulate(board, initialScore, depth = depth + 1)

            while time.time() - start < maxTime:
                testStart = time.time()
                leaf = treeRoot.getLeaf()
                totalTime += time.time() - testStart
                score = simulate(leaf.board, initialScore, depth = leaf.depth)
                simCount += 1
                leaf.addMatch(score)


            print("Total Time: " + str(totalTime))
            print('Sim Count: ' + str(simCount))
            print('Depth: ' + str(treeRoot.getDepth()))
            if self.whosTurn():
                bestMove = treeRoot.getMax()
            else:
                bestMove = treeRoot.getMin()
            return bestMove

        return customTree(board, solvableMoves=solvableMoves)

    def getTree(self, board, depth):
        layer = []
        moves = [move for move in board.legal_moves]
        for move in moves:
            newBoard = board.copy()
            newBoard.push(move)
            theScore = self.score(newBoard)
            if depth <= 1:
                layer.append([newBoard, move, theScore])
            else:
                layer.append([newBoard, move, theScore, self.getTree(newBoard, depth - 1)])
        return layer

# derp = BishopAI()
# derp.scoreAdd = 0
# start = time.time()
# derp.getMove()
# end = time.time()
# print(end-start)
# print(derp.scoreAdd)
# print(derp.board.pieces)
