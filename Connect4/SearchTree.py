import random

class SearchTree:

	def __init__(self, prevMove=None, policy=0):
		self.board = None
		self.prevMove = prevMove
		self.policy = policy
		self.score = None
		self.evaluated = False
		self.Q = 0
		self.N = 1
		self.depth = 0
		self.parent = None
		self.children = []
		self.endGame = None

	def evaluation(self, board, score):
		self.board = board
		self.score = score
		self.evaluated = True
		if self.parent:
			self.parent.backpropogate(score)

	def addChild(self, tree):
		tree.depth = self.depth + 1
		tree.parent = self
		self.children.append(tree)

	def backpropogate(self, score):
		self.N += 1
		self.Q += score
		if self.parent:
			self.parent.backpropogate(1-score)

	def getDepth(self, initialCall=True):
		depth = self.depth if len(self.children) == 0 else max([child.getDepth(initialCall=False) for child in self.children])
		return depth-self.depth if initialCall else depth

	def calcUTC(self):
		return (self.Q/self.N)+4*self.policy*((self.parent.N)**0.5/(1+self.N))

	def alphaBeta(self, color):
		if color:
			return max(self.children, key=lambda tree: tree.alphaBetaHelper(not color))
		else:
			return min(self.children, key=lambda tree: tree.alphaBetaHelper(not color))

	def alphaBetaHelper(self, color, alpha=-100000, beta=100000):
		if not self.evaluated:
			return self.score
		if color:
			value = -100000
			for child in self.children:
				if not child.evaluated:
					continue
				value = max(value, child.alphaBetaHelper(not color, alpha, beta))
				alpha = max(alpha, value)
				if alpha > beta:
					break
			return value
		else:
			value = 100000
			for child in self.children:
				if not child.evaluated:
					continue
				value = min(value, child.alphaBetaHelper(not color, alpha, beta))
				alpha = min(alpha, value)
				if alpha > beta:
					break
			return value

	def __lt__(self, other):
		return self.searchValue > other.searchValue

	def __eq__(self, other):
		return self.searchValue == other.searchValue
