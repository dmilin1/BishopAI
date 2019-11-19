import random
import chess

class MCTree:

    def __init__(self, board, parent = None, move = None):
        self.tree = []
        self.board = board
        self.wins = 0
        self.total = 0
        self.parent = parent
        self.move = move
        self.depth = 0 if parent == None else parent.depth + 1
        self.isLeaf = True
        self.possibleMoves = None
        self.unloadedMoves = None
        self.calcScores = []


    def newChild(self, board, move):
        newTree = MCTree(board, parent = self, move = move)
        newTree.depth = self.depth + 1
        self.isLeaf = False
        return newTree

    def isLeaf(self):
        if len(self.tree) == 0:
            return True
        return False

    def detach(self):
        self.parent = None

    def addMatch(self, score):
        self.calcScores.append(score)
        self.wins += score
        self.total += 1
        if self.parent:
            self.parent.addMatch(score)

    def getLeaf(self):

        def loadRandMove():
            if len(self.unloadedMoves) == 0:
                return self
            unloadedIndex = random.randint(0, len(self.unloadedMoves)-1)
            moveIndex = self.unloadedMoves[unloadedIndex]
            move = self.possibleMoves[moveIndex]
            del self.unloadedMoves[unloadedIndex]
            newBoard = self.board.copy()
            newBoard.push(move)
            self.possibleMoves[moveIndex] = self.newChild(newBoard, move)
            return self.possibleMoves[moveIndex]

        if not self.possibleMoves:
            self.possibleMoves = [move for move in self.board.legal_moves]
            self.unloadedMoves = list(range(len(self.possibleMoves)))
            return loadRandMove()

        moveIndex = random.randint(0, len(self.possibleMoves)-1)
        move = self.possibleMoves[moveIndex]

        if type(move) == chess.Move:
            newBoard = self.board.copy()
            newBoard.push(move)
            self.possibleMoves[moveIndex] = self.newChild(newBoard, move)
            self.unloadedMoves.remove(moveIndex)
            return self.possibleMoves[moveIndex]

        return move.getLeaf()


    def getMax(self):
        largestScore = -100000
        largestChild = None
        for child in self.possibleMoves:
            if type(child) == type(self):
                tempScore = child.wins/child.total
                if tempScore > largestScore:
                    largestScore = tempScore
                    largestChild = child
        for child in self.possibleMoves:
            if type(child) == type(self):
                print(str(child.move) + '\t' + str(round(child.wins/child.total, 3)) + '\t' + str(round(child.wins,1)) + '/' + str(child.total) + ('\t\t*' if child == largestChild else ''))
        return largestChild.move

    def getMin(self):
        smallestScore = 100000
        smallestChild = None
        for child in self.tree:
            if type(child) == type(self):
                tempScore = child.wins/child.total
                if tempScore < smallestScore:
                    smallestScore = tempScore
                    smallestChild = child
        for child in self.possibleMoves:
            if type(child) == type(self):
                print(str(child.move) + '\t' + str(round(child.wins/child.total, 3)) + '\t' + str(round(child.wins,1)) + '/' + str(child.total) + ('\t\t*' if child == smallestChild else ''))
        return smallestChild.move

    def getDepth(self):
        if len(self.tree) == 0:
            return self.depth
        return max(child.getDepth() for child in self.tree)
