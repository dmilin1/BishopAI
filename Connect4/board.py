import piece
import _pickle as pickle

defaultBoard = [["bR", "bN", "bB", "bQ", "bK", "bB", "bN", "bR"],
                ["bP", "bP", "bP", "bP", "bP", "bP", "bP", "bP"],
                ["__", "__", "__", "__", "__", "__", "__", "__"],
                ["__", "__", "__", "__", "__", "__", "__", "__"],
                ["__", "__", "__", "__", "__", "__", "__", "__"],
                ["__", "__", "__", "__", "__", "__", "__", "__"],
                ["wP", "wP", "wP", "wP", "wP", "wP", "wP", "wP"],
                ["wR", "wN", "wB", "wQ", "wK", "wB", "wN", "wR"]]

emptyBoard =   [["__", "__", "__", "__", "__", "__", "__", "__"],
                ["__", "__", "__", "__", "__", "__", "__", "__"],
                ["__", "__", "__", "__", "__", "__", "__", "__"],
                ["__", "__", "__", "__", "__", "__", "__", "__"],
                ["__", "__", "__", "__", "__", "__", "__", "__"],
                ["__", "__", "__", "__", "__", "__", "__", "__"],
                ["__", "__", "__", "__", "__", "__", "__", "__"],
                ["__", "__", "__", "__", "__", "__", "__", "__"]]


class Board(object):



    # creates the board using the initial board setup if no board is passed
    def __init__(self, board = defaultBoard):

        self.board = []    # the board stored as a 2d array of piece objects

        self.nextBoardsLoaded = False
        self.nextBoards = []    # all possible next boards. Must call getAllMoves() to initialize
                                # getAllMoves sets it to None if no valid moves exist

        self.whitesTurn = True    # when White is about to move, this is True. Black's turn is False.

        self.state = None    # None for nothing, 'wCheck' or 'bCheck' for check,
                             # 'wCheckmate' or 'bCheckmate' for checkmate,
                             # 'invalid' for an unreachable board state


        for i, row in enumerate(board):
            self.board.append([])
            for j, thePiece in enumerate(row):
                if type(thePiece) == str:
                    createdPiece = piece.makePiece(thePiece, (i,j))
                    self.board[i].append(createdPiece)
                else:
                    self.board[i].append(thePiece)

    def __str__(self):
        theString = ""
        for row in self.board:
            for thePiece in row:
                if thePiece != None:
                    theString += thePiece.name
                else:
                    theString += "__"
                theString += " "
            theString += "\n"
        return theString

    def copy(self):
        board = pickle.loads(pickle.dumps(self.board, -1))
        boardCopy = Board(board=board)
        return boardCopy


    # returns an array of Board objects that represent possible board states
    def getAllMoves(self, initialCall = True):
        if self.nextBoardsLoaded == True:
            raise Exception('already loaded moves')
        if self.state == 'invalid':
            raise Exception('trying to load next moves of invalid board state')

        def inCheck(theBoard):
            whiteInCheck = False
            blackInCheck = False
            theBoard.getAllMoves(initialCall = False)
            for boardState in theBoard.nextBoards:
                foundBlackKing = False
                foundWhiteKing = False
                for row in boardState.board:
                    for thePiece in row:
                        if thePiece != None:
                            if thePiece.name == 'wK':
                                foundWhiteKing = True
                            if thePiece.name == 'bK':
                                foundBlackKing = True
                if not foundBlackKing:
                    blackInCheck = True
                    boardState.state = 'invalid'
                if not foundWhiteKing:
                    whiteInCheck = True
                    boardState.state = 'invalid'
            return [whiteInCheck, blackInCheck]

        player = 'w' if self.whitesTurn else 'b' # sets player to the color of the side taking the turn
        for row in self.board:
            for thePiece in row:
                if thePiece != None and thePiece.color == player: # gets all pieces of the player's color
                    for move in thePiece.getMoves(self):
                        newPiece = move[0]
                        newBoard = self.copy()
                        newBoard.setPosition(thePiece.position, None)
                        newBoard.setPosition(newPiece.position, newPiece)
                        newBoard.whitesTurn = not self.whitesTurn
                        if initialCall:
                            # print("~~~~~~~~~~~~~~~~~~~~~~~ CHECK ~~~~~~~~~~~~~~~~~~~~~~~" if checkForCheck(newBoard) else "")
                            # print(newBoard)
                            self.nextBoardsLoaded = True
                            whiteInCheck, blackInCheck = inCheck(newBoard)
                            if self.whitesTurn:
                                if self.state == None:
                                    if whiteInCheck:
                                        newBoard.state = 'invalid'
                                    elif blackInCheck:
                                        newBoard.state = 'bCheck'
                                if self.state == 'wCheck':
                                    if whiteInCheck:
                                        newBoard.state = 'invalid'
                                    elif blackInCheck:
                                        newBoard.state = 'bCheck'
                                if self.state == 'bCheck':
                                    raise Exception('Something went wrong')

                            else:
                                if self.state == None:
                                    if blackInCheck:
                                        newBoard.state = 'invalid'
                                    elif whiteInCheck:
                                        newBoard.state = 'wCheck'
                                if self.state == 'bCheck':
                                    if blackInCheck:
                                        newBoard.state = 'invalid'
                                    elif whiteInCheck:
                                        newBoard.state = 'bCheck'
                                if self.state == 'wCheck':
                                    raise Exception('Something went wrong')

                        self.nextBoards.append(newBoard)
                        print(newBoard)


        # check for check, checkmate, or stalemate
        # fix En Passant Possible variable

    def getPosition(self, position):
        if position[0] < 0 or position[1] < 0:
            raise IndexError
        return self.board[position[0]][position[1]]

    def setPosition(self, position, value):
        if position[0] < 0 or position[1] < 0:
            raise IndexError
        self.board[position[0]][position[1]] = value

testBoard =    [["bR", "bN", "bB", "bQ", "bK", "bB", "bN", "bR"],
                ["bP", "bP", "bP", "bP", "bP", "bP", "bP", "bP"],
                ["__", "__", "__", "__", "__", "__", "__", "__"],
                ["__", "__", "__", "__", "__", "__", "__", "__"],
                ["__", "__", "__", "__", "__", "__", "__", "__"],
                ["__", "__", "__", "__", "__", "__", "__", "__"],
                ["wP", "wP", "wP", "wP", "wP", "wP", "wP", "wP"],
                ["wR", "wN", "wB", "wQ", "wK", "wB", "wN", "wR"]]

derp = Board(board = testBoard)

derp.getAllMoves()
