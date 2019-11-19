
class Tree:

    def __init__(self, score = None, parent = None, move = None):
        self.tree = []
        self.score = score
        self.parent = parent
        self.move = move
        self.depth = 0 if parent == None else parent.depth + 1
        self.isLeaf = True

    def __repr__(self):
        return str(self.tree)


    def append(self, score, move):
        newTree = Tree(score = score, parent = self, move=move)
        self.tree.append(newTree)
        self.isLeaf = False
        return newTree

    def isLeaf(self):
        if len(self.tree) == 0:
            return True
        return False

    def getMax(self, initialCall = True):
        if self.isLeaf:
            return self.score
        maxArr = []
        for child in self.tree:
            if initialCall:
                maxArr.append([child.getMin(initialCall = False), child.move])
            else:
                maxArr.append(child.getMin(initialCall = False))
        if initialCall:
            maxVal = max(val[0] for val in maxArr)
            bestMoves = []
            for valMove in maxArr:
                if valMove[0] == maxVal:
                    bestMoves.append(valMove[1])
            print("Move Score: " + str(maxVal-self.score))
            return bestMoves
        return max(maxArr)

    def getMin(self, initialCall = True):
        if self.isLeaf:
            return self.score
        minArr = []
        for child in self.tree:
            if initialCall:
                minArr.append([child.getMax(initialCall = False), child.move])
            else:
                minArr.append(child.getMax(initialCall = False))
        if initialCall:
            minVal = min(val[0] for val in minArr)
            bestMoves = []
            for valMove in minArr:
                if valMove[0] == minVal:
                    bestMoves.append(valMove[1])
            print("Move Score: " + str(minVal-self.score))
            return bestMoves
        return min(minArr)

    def getDepth(self):
        if len(self.tree) == 0:
            return self.depth
        return max(child.getDepth() for child in self.tree)
