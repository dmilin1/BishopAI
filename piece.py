
# internally used superclass
# includes methods needed by all piece types
class Piece(object):    # DO NOT USE THIS. Call makePiece() instead.

    name = None    # full name of piece e.g. "bK" for black King
    type = None    # just the type of piece e.g. "K" for King
    color = None   # just the color of the piece e.g. "b" for black
    opponentColor = None    # opposite color

    position = [None, None]    # the coordinates of the piece e.g. (3, 7)


    def __init__(self, name, position):    # initialization for all pieces
        self.name = name
        self.type = name[1:]
        self.color = name[0]
        self.opponentColor = 'b' if name[0] == 'w' else 'w'
        self.position = position

    def modifyPosition(self, up, right):    # returns current position shifted by given inputs
        return [self.position[0] - up, self.position[1] + right]


class King(Piece):    # initialization for King pieces only

    hasMoved = False    # Kings can only castle if they haven't moved yet

    def __init__(self, name, position):
        Piece.__init__(self, name, position)


    # returns list of potential moves
    # each item of the list is another list which contains the following:
    #
    #   the modified piece which contains new coordinates
    #   the relative change in coordinates
    #   !optional! command used by board to handle additional instructions e.g. castling
    #
    def getMoves(self, chessBoard):
        moves = []

        # move one step in 8 directions
        for up, right in [[1, 1], [1, -1], [-1, 1], [-1, -1], [1, 0], [0, 1], [-1, 0], [0, -1]]:
            try:
                if chessBoard.getPosition(self.modifyPosition(up,right)) == None:
                    newPiece = makePiece(self.color + "K", position=self.modifyPosition(up,right))
                    moves.append([newPiece])
                if chessBoard.getPosition(self.modifyPosition(up,right)) != None:
                    if chessBoard.getPosition(self.modifyPosition(up,right)).color == self.opponentColor:
                        newPiece = makePiece(self.color + "K", position=self.modifyPosition(up,right))
                        moves.append([newPiece])
                    break
            except IndexError:
                break

        for move in moves:
            move.insert(1, [move[0].position[0] - self.position[0],move[0].position[1] - self.position[1]])
        return moves

class Queen(Piece):    # initialization for Queen pieces only

    def __init__(self, name, position):
        Piece.__init__(self, name, position)


    # returns list of potential moves
    # each item of the list is another list which contains the following:
    #
    #   the modified piece which contains new coordinates
    #   the relative change in coordinates
    #   !optional! command used by board to handle additional instructions e.g. castling
    #
    def getMoves(self, chessBoard):
        moves = []

        # move 8 directions
        for up, right in [[1, 1], [1, -1], [-1, 1], [-1, -1], [1, 0], [0, 1], [-1, 0], [0, -1]]:
            for moveDistance in [1, 2, 3, 4, 5, 6, 7]:
                try:
                    if chessBoard.getPosition(self.modifyPosition(up*moveDistance,right*moveDistance)) == None:
                        newPiece = makePiece(self.color + "Q", position=self.modifyPosition(up*moveDistance,right*moveDistance))
                        moves.append([newPiece])
                    if chessBoard.getPosition(self.modifyPosition(up*moveDistance,right*moveDistance)) != None:
                        if chessBoard.getPosition(self.modifyPosition(up*moveDistance,right*moveDistance)).color == self.opponentColor:
                            newPiece = makePiece(self.color + "Q", position=self.modifyPosition(up*moveDistance,right*moveDistance))
                            moves.append([newPiece])
                        break
                except IndexError:
                    break

        for move in moves:
            move.insert(1, [move[0].position[0] - self.position[0],move[0].position[1] - self.position[1]])
        return moves

class Bishop(Piece):    # initialization for Bishop pieces only

    def __init__(self, name, position):
        Piece.__init__(self, name, position)


    # returns list of potential moves
    # each item of the list is another list which contains the following:
    #
    #   the modified piece which contains new coordinates
    #   the relative change in coordinates
    #   !optional! command used by board to handle additional instructions e.g. castling
    #
    def getMoves(self, chessBoard):
        moves = []

        # move 4 directions
        for up, right in [[1, 1], [1, -1], [-1, 1], [-1, -1]]:
            for moveDistance in [1, 2, 3, 4, 5, 6, 7]:
                try:
                    if chessBoard.getPosition(self.modifyPosition(up*moveDistance,right*moveDistance)) == None:
                        newPiece = makePiece(self.color + "B", position=self.modifyPosition(up*moveDistance,right*moveDistance))
                        moves.append([newPiece])
                    if chessBoard.getPosition(self.modifyPosition(up*moveDistance,right*moveDistance)) != None:
                        if chessBoard.getPosition(self.modifyPosition(up*moveDistance,right*moveDistance)).color == self.opponentColor:
                            newPiece = makePiece(self.color + "B", position=self.modifyPosition(up*moveDistance,right*moveDistance))
                            moves.append([newPiece])
                        break
                except IndexError:
                    break

        for move in moves:
            move.insert(1, [move[0].position[0] - self.position[0],move[0].position[1] - self.position[1]])
        return moves

class Knight(Piece):    # initialization for Knight pieces only

    def __init__(self, name, position):
        Piece.__init__(self, name, position)


    # returns list of potential moves
    # each item of the list is another list which contains the following:
    #
    #   the modified piece which contains new coordinates
    #   the relative change in coordinates
    #   !optional! command used by board to handle additional instructions e.g. castling
    #
    def getMoves(self, chessBoard):
        moves = []

        # check all jump combinations
        for up, right in [[2, 1], [1, 2], [-1, 2], [-2, 1], [-2, -1], [-1, -2], [1, -2], [2, -1]]:
            try:
                if chessBoard.getPosition(self.modifyPosition(up,right)) == None:
                    newPiece = makePiece(self.color + "N", position=self.modifyPosition(up,right))
                    moves.append([newPiece])
                if chessBoard.getPosition(self.modifyPosition(up,right)) != None:
                    if chessBoard.getPosition(self.modifyPosition(up,right)).color == self.opponentColor:
                        newPiece = makePiece(self.color + "N", position=self.modifyPosition(up,right))
                        moves.append([newPiece])
            except IndexError:
                pass

        for move in moves:
            move.insert(1, [move[0].position[0] - self.position[0],move[0].position[1] - self.position[1]])
        return moves

class Rook(Piece):    # initialization for Rook pieces only

    hasMoved = False    # Rooks can only be used to castle if they haven't moved yet


    def __init__(self, name, position):
        Piece.__init__(self, name, position)

    # returns list of potential moves
    # each item of the list is another list which contains the following:
    #
    #   the modified piece which contains new coordinates
    #   the relative change in coordinates
    #   !optional! command used by board to handle additional instructions e.g. castling
    #
    def getMoves(self, chessBoard):
        moves = []

        # move left
        for moveLeft in [-1, -2, -3, -4, -5, -6, -7]:
            try:
                if chessBoard.getPosition(self.modifyPosition(0,moveLeft)) == None:
                    newPiece = makePiece(self.color + "R", position=self.modifyPosition(0,moveLeft))
                    newPiece.hasMoved = True
                    moves.append([newPiece])
                if chessBoard.getPosition(self.modifyPosition(0,moveLeft)) != None:
                    if chessBoard.getPosition(self.modifyPosition(0,moveLeft)).color == self.opponentColor:
                        newPiece = makePiece(self.color + "R", position=self.modifyPosition(0,moveLeft))
                        newPiece.hasMoved = True
                        moves.append([newPiece])
                    break
            except IndexError:
                break

        # move right
        for moveRight in [1, 2, 3, 4, 5, 6, 7]:
            try:
                if chessBoard.getPosition(self.modifyPosition(0,moveRight)) == None:
                    newPiece = makePiece(self.color + "R", position=self.modifyPosition(0,moveRight))
                    newPiece.hasMoved = True
                    moves.append([newPiece])
                if chessBoard.getPosition(self.modifyPosition(0,moveRight)) != None:
                    if chessBoard.getPosition(self.modifyPosition(0,moveRight)).color == self.opponentColor:
                        newPiece = makePiece(self.color + "R", position=self.modifyPosition(0,moveRight))
                        newPiece.hasMoved = True
                        moves.append([newPiece])
                    break
            except IndexError:
                break

        # move up
        for moveUp in [1, 2, 3, 4, 5, 6, 7]:
            try:
                if chessBoard.getPosition(self.modifyPosition(moveUp,0)) == None:
                    newPiece = makePiece(self.color + "R", position=self.modifyPosition(moveUp,0))
                    newPiece.hasMoved = True
                    moves.append([newPiece])
                if chessBoard.getPosition(self.modifyPosition(moveUp,0)) != None:
                    if chessBoard.getPosition(self.modifyPosition(moveUp,0)).color == self.opponentColor:
                        newPiece = makePiece(self.color + "R", position=self.modifyPosition(moveUp,0))
                        newPiece.hasMoved = True
                        moves.append([newPiece])
                    break
            except IndexError:
                break

        # move down
        for moveDown in [-1, -2, -3, -4, -5, -6, -7]:
            try:
                if chessBoard.getPosition(self.modifyPosition(moveDown,0)) == None:
                    newPiece = makePiece(self.color + "R", position=self.modifyPosition(moveDown,0))
                    newPiece.hasMoved = True
                    moves.append([newPiece])
                if chessBoard.getPosition(self.modifyPosition(moveDown,0)) != None:
                    if chessBoard.getPosition(self.modifyPosition(moveDown,0)).color == self.opponentColor:
                        newPiece = makePiece(self.color + "R", position=self.modifyPosition(moveDown,0))
                        newPiece.hasMoved = True
                        moves.append([newPiece])
                    break
            except IndexError:
                break


        for move in moves:
            move.insert(1, [move[0].position[0] - self.position[0],move[0].position[1] - self.position[1]])
        return moves



class Pawn(Piece):    # initialization for Pawn pieces only

    hasMoved = False    # pawns can move 2 spaces on their first turn
    enPassantPossible = False    # google "En Passant", not easy to explain


    def __init__(self, name, position):
        Piece.__init__(self, name, position)

    # returns list of potential moves
    # each item of the list is another list which contains the following:
    #
    #   the modified piece which contains new coordinates
    #   the relative change in coordinates
    #   !optional! command used by board to handle additional instructions e.g. castling
    #
    def getMoves(self, chessBoard):
        moves = []
        if self.color == 'w':

            # move forward 1 spot
            try:
                if chessBoard.getPosition(self.modifyPosition(1,0)) == None:
                    newPiece = makePiece(self.color + "P", position=self.modifyPosition(1,0))
                    newPiece.hasMoved = True
                    moves.append([newPiece])
            except IndexError:
                pass

            # move forward 2 spots (only on first move)
            try:
                if chessBoard.getPosition(self.modifyPosition(2,0)) == None and not self.hasMoved:
                    newPiece = makePiece(self.color + "P", position=self.modifyPosition(2,0))
                    newPiece.hasMoved = True
                    newPiece.enPassantPossible = True
                    moves.append([newPiece])
            except IndexError:
                pass

            # attack forward left and forward right
            for leftRight in [-1,1]:
                try:
                    if chessBoard.getPosition(self.modifyPosition(1,leftRight)) != None:
                        if chessBoard.getPosition(self.modifyPosition(1,leftRight)).color == 'b':
                            newPiece = makePiece(self.color + "P", position=self.modifyPosition(1,leftRight))
                            newPiece.hasMoved = True
                            moves.append([newPiece])
                except IndexError:
                    pass

            # en passant attack forward left and forward right
            for leftRight in [-1,1]:
                try:
                    if chessBoard.getPosition(self.modifyPosition(0,leftRight)) != None:
                        if chessBoard.getPosition(self.modifyPosition(0,leftRight)).name == 'bP':
                            if chessBoard.getPosition(self.modifyPosition(0,leftRight)).enPassantPossible == True:
                                newPiece = makePiece(self.color + "P", position=self.modifyPosition(1,leftRight))
                                newPiece.hasMoved = True
                                moves.append([newPiece, "en passant"])
                except IndexError:
                    pass

        if self.color == 'b':

            # move forward 1 spot
            try:
                if chessBoard.getPosition(self.modifyPosition(-1,0)) == None:
                    newPiece = makePiece(self.color + "P", position=self.modifyPosition(-1,0))
                    newPiece.hasMoved = True
                    moves.append([newPiece])
            except IndexError:
                pass

            # move forward 2 spots (only on first move)
            try:
                if chessBoard.getPosition(self.modifyPosition(-2,0)) == None and not self.hasMoved:
                    newPiece = makePiece(self.color + "P", position=self.modifyPosition(-2,0))
                    newPiece.hasMoved = True
                    newPiece.enPassantPossible = True
                    moves.append([newPiece])
            except IndexError:
                pass

            # attack forward left and forward right
            for leftRight in [-1,1]:
                try:
                    if chessBoard.getPosition(self.modifyPosition(-1,leftRight)) != None:
                        if chessBoard.getPosition(self.modifyPosition(-1,leftRight)).color == 'w':
                            newPiece = makePiece(self.color + "P", position=self.modifyPosition(-1,leftRight))
                            newPiece.hasMoved = True
                            moves.append([newPiece])
                except IndexError:
                    pass

            # en passant attack forward left and forward right
            for leftRight in [-1,1]:
                try:
                    if chessBoard.getPosition(self.modifyPosition(0,leftRight)) != None:
                        if chessBoard.getPosition(self.modifyPosition(0,leftRight)).name == 'wP':
                            if chessBoard.getPosition(self.modifyPosition(0,leftRight)).enPassantPossible == True:
                                newPiece = makePiece(self.color + "P", position=self.modifyPosition(-1,leftRight))
                                newPiece.hasMoved = True
                                moves.append([newPiece, "en passant"])
                except IndexError:
                    pass

        for move in moves:
            move.insert(1, [move[0].position[0] - self.position[0],move[0].position[1] - self.position[1]])

        return moves






def makePiece(name, *args, **kwargs):
    type = name[1:]
    color = name[0]

    if type == "K":
        return King(name, *args, **kwargs)
    elif type == "Q":
        return Queen(name, *args, **kwargs)
    elif type == "B":
        return Bishop(name, *args, **kwargs)
    elif type == "N":
        return Knight(name, *args, **kwargs)
    elif type == "R":
        return Rook(name, *args, **kwargs)
    elif type == "P":
        return Pawn(name, *args, **kwargs)
    elif type == "_":
        return None
