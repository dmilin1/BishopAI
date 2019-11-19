# imports
import numpy as np
import random
import time

BOARD_SIZE=(6,7)

# The Connect4 class handles game state and game history.
# This means a game completed with Connect4 represents and
# stores the data required to reproduce an entire game of
# Connect4. Individual positions are stored as Connect4Board.
class Connect4:

    def __init__(self, board=None):
        # self.turn is True for 1st to move player and False for 2nd to move player
        self.turn = True
        # self.boardStack represents the board history in Connect4Board objects.
        # Connect4 is initialized with an empty Connect4Board object in the stack
        # unless otherwise specified
        self.boardStack = [Connect4Board(board=board)]
        # self.result represents the result of the game in its current position.
        # It should be automatically recalculated every time a new move/board is
        # added to the move stack.
        # None = game is still in progress
        # 1 = first player to move has won
        # 0 = the game is a tie
        # -1 = second player to move has won
        self.result = None
        if board is None:
            self.rowHeights = np.zeros((BOARD_SIZE[1],), dtype=int)
        else:
            self.rowHeights = self.getRowHeights(board)

    # This is a str() override. Calling print or str() on a Connect4 object
    # will return a string representation of the newest board on the stack
    def __str__(self):
        return str(self.boardStack[-1])

    # returns the newest board on the stack
    def currentBoard(self):
        return self.boardStack[-1].board

    # returns a generator that represents all possible moves for the
    # current board. Moves are an integer that represents the slot that
    # a piece can be placed in. For example, "list(Connect4().legal_moves())"
    # will return "[0, 1, 2, 3, 4, 5, 6, 7]"
    def legal_moves(self):
        for i in range(BOARD_SIZE[1]):
            if self.currentBoard()[0][i] == 0:
                yield i

    # accepts an integer representing a legal move (read legal_moves) and
    # pushes a new board onto self.boardStack
    def push(self, num):
        if self.currentBoard()[0][num] != 0:
            raise Exception('Illegal Move')
        newBoard = np.copy(self.currentBoard())
        y = BOARD_SIZE[0]-self.rowHeights[num]-1
        x = num
        newBoard[y][x] = 1 if self.turn else -1
        self.rowHeights[num] += 1

        self.boardStack.append(Connect4Board(board=newBoard, prevMove=num))
        self.turn = not self.turn
        self.calcResult(y, x)

    # restores the previous position and returns the last move from the stack
    def pop(self):
        newestBoard = self.boardStack.pop()
        self.rowHeights[newestBoard.prevMove] -= 1
        return newestBoard.prevMove

    # returns the last move from the stack
    def peek(self):
        return self.boardStack[-1].prevMove

    # sets self.result to the appropriate value based on the state of the
    # newest board in the stack. If the game is not over, result is None.
    def calcResult(self, y, x):
        board = self.currentBoard()
        target = board[y][x]
        # check horizontal
        directions = [[0,1], [1,0], [1,1], [1,-1]]
        for direction in directions:
            width = 1
            i = 1
            try:
                while board[y+i*direction[0]][x+i*direction[1]] == target and x+i*direction[1] >= 0:
                    width += 1
                    i += 1
                    if width >=4:
                        self.result = target
                        return self.result
            except:
                pass
            i = 1
            try:
                while board[y-i*direction[0]][x-i*direction[1]] == target and y-i*direction[0] >= 0 and x-i*direction[1] >= 0:
                    width += 1
                    i += 1
                    if width >=4:
                        self.result = target
                        return self.result
            except:
                pass

        return self.result

    # returns a mirrored, shallow stack version of Connect4. In other words,
    # the stack will only have the current board and the ownership of all
    # pieces will be reversed. All of player 1's pieces are now player 2's
    # and all of player 2's pieces are now player 1's
    def mirror(self):
        return Connect4(board=self.currentBoard()*-1)

    def getRowHeights(self, board):
        heights = np.zeros((BOARD_SIZE[1],), dtype=int)
        for y in range(BOARD_SIZE[0]):
            for x in range(BOARD_SIZE[1]):
                if board[y][x] != 0:
                    heights[x] += 1
        return heights

# A single game board representing the positions of pieces
# but no game data (i.e. who's turn). Previous move is stored
# in prevMove for simplification of stack handling in Connect4.
class Connect4Board:

    def __init__(self, board=None, prevMove=None):
        self.prevMove = prevMove
        if board is not None:
            self.board = board
        else:
            self.board = np.zeros(BOARD_SIZE)

        # self.board is a numpy array where the 1st player's pieces
        # are represented by 1 and second player's by -1. Empty
        # squares are represented by 0. Coordinates are represented
        # backwards in numpy arrays. For example, array[y][x].

    # returns the current board as string like this:
    #
    #  · · · · · · ·
    #  · · · · · · ·
    #  · · · · · · ·
    #  · · · o · · ·
    #  · · x x · · ·
    #  · x x o o · ·
    #
    # player one is represented by x
    # player two is represented by o
    def __str__(self):
        string = ''
        for row, _ in enumerate(self.board):
            for position in self.board[row]:
                string += ' '
                if position == 0:
                    string += '·'
                elif position == 1:
                    string += 'x'
                elif position == -1:
                    string += 'o'
            string += '\n'
        return string


# Test code to make sure the classes work correctly

# myboard = np.array([
#     [ 0, 0, 0, 0, 0, 0, 0],
#     [ 0, 0, 0, 0, 0, 0, 0],
#     [ 0, 0, 0, 0, 1, 0, 0],
#     [-1, 0, 0, 0,-1,-1, 0],
#     [ 1, 0, 1, 0,-1,-1, 0],
#     [ 1, 0, 1,-1,-1, 1, 1]
# ])
#
# start = time.time()
# for i in range(10000):
#     board = Connect4()
#     moves = list(board.legal_moves())
#     while board.result == None and len(moves) > 0:
#         move = random.choice(moves)
#         board.push(move)
#         moves = list(board.legal_moves())
#
# print(time.time()-start)

# board = Connect4(board=myboard)
# board.push(1)
# print(board)
# print(board.result)
