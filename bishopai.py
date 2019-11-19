import chess
import chess.syzygy
import tensorflow as tf
import random
import time
import queue
from Tree import Tree
from SearchTree import SearchTree
from MonteCarloTree import MCTree
from ai import AI
from inputBuilder import buildInput
import numpy as np

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

	def __init__(self, ai=None):
		self.board = chess.Board()
		print('\n~~~~~~\n')
		print(self.board)

		if ai is None:
			self.ai = AI()
			self.ai.newModel()
		else:
			if isinstance(ai, str):
				self.ai = AI()
				self.ai.loadModel(ai)
			else:
				self.ai = ai

		global graph
		graph = tf.get_default_graph()

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



	def getMove(self, *args, **kwargs):
		print(self.board.fen())

		solvableMoves = None

		# with chess.syzygy.open_tablebase("./syzygy/") as tablebase:
		# 	if tablebase.get_wdl(self.board) != None:
		# 		winCertainty = tablebase.get_wdl(self.board)
		# 		print('Win Certainty: ' + str(winCertainty))
		# 		if winCertainty == 2 * ( -1 if self.board.turn else 1):
		# 			return None
		# 		if winCertainty > -2 and winCertainty < 2:
		# 			[self.solve(self.board, tablebase), 'draw']
		#
		# 		solvableMoves = self.solve(self.board, tablebase)


		global scoreTotal
		scoreTotal = 0
		start = time.time()
		if solvableMoves:
			newMove = self.pickMove(self.board, solvableMoves=solvableMoves)
		else:
			newMove = self.pickMove(self.board, *args, **kwargs)
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

	def scoreMl(self, board):

		with graph.as_default():
			score, move = self.ai.model.predict(np.array([buildInput(board)]))
			return score, move

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


	def pickMove(self, board, maxTime=10, maxSimulations=100000, depth=5, temperature=0, verbose=True, returnStatistics=False, solvableMoves=None):
		if verbose:
			print("thinking")
		global totalTime

		def treeSearch(board, maxTime, maxSimulations, temperature, verbose, returnStatistics=returnStatistics):
			startTime = time.time()
			global totalTime

			def normalize(arr):
				theSum = sum(arr)
				if theSum == 0:
					print('ERR: DIVIDE BY ZERO IN NORMALIZATION')
					theSum = 1
				return [x/theSum for x in arr]

			def bloomBranch(tree, movePolicies):
				allMoves = [move for move in tree.board.legal_moves]
				movePoliciesNormalized = np.zeros((len(allMoves),))
				for i, move in enumerate(allMoves):
					movePoliciesNormalized[i] = movePolicies[move.from_square*64+move.to_square]
				movePoliciesNormalized **= 0.5
				movePoliciesNormalized = normalize(movePoliciesNormalized)
				for i, move in enumerate(allMoves):
					policy = movePoliciesNormalized[i]
					newTree = SearchTree(prevMove=move, policy=policy)
					tree.addChild(newTree)

			def rollout(tree):
				if tree.evaluated == False:
					newBoard = tree.parent.board.copy()
					newBoard.push(tree.prevMove)
					score, movePolicies = self.scoreMl(newBoard)
					score = score[0][0]
					score = (score + 1)/2
					# print(score)
					# score = (tree.parent.score - score) * (1 if newBoard.turn else -1)
					movePolicies = movePolicies[0]
					tree.evaluation(newBoard, score)
					if tree.board.result() == '*':
						bloomBranch(tree, movePolicies)
				else:
					if tree.children:
						rollout(max(tree.children, key=lambda child: child.calcUTC()))
					else:
						if verbose:
							print('End game in sight. Ending Evaluated ' + str(tree.N) + ' times' )
						# print('game over\n---')
						# print(tree.board)
						# print(tree.board.is_game_over())
						score = 2 if tree.board.result() != '1/2-1/2' else 0
						tree.backpropogate(score)

			score, movePolicies = self.scoreMl(board)
			score = score[0][0]
			score = (score + 1)/2
			movePolicies = movePolicies[0]
			treeRoot = SearchTree()
			treeRoot.evaluation(board, score)
			bloomBranch(treeRoot, movePolicies)

			simCount = 0
			while time.time() - startTime < maxTime and simCount < maxSimulations:
				rollout(treeRoot)
				simCount += 1

			def normalize(arr):
				min = np.amin(arr)
				max = np.amax(arr)
				if min == max:
					return np.ones(np.shape(arr))/len(arr)
				return (arr-min)/(max-min)

			if verbose:
				for child in treeRoot.children:
					print('\n')
					print(child.board)
					print('Position Score: ' + str(child.score))
					print('Policy Score:' + str(child.policy))
					print('Visits: ' + str(child.N))
					print('Arm Avg: ' + str(child.Q / child.N))

				print('Max Search Depth: ' + str(treeRoot.getDepth()))
				print('Positions Evaluated: ' + str(treeRoot.N))
				print('Alpha Beta: \n' + str(max(treeRoot.children, key=lambda child: child.N).board))
			# quit()

			choiceOdds = [child.N**(1/max(0.02,temperature)) for child in treeRoot.children]
			bestChild = random.choices(treeRoot.children, weights=choiceOdds, k=1)[0]
			chosenMove = bestChild.prevMove
			print('Evaluations: ' + str([str(child.prevMove) + ": " + str(child.Q/child.N) for child in treeRoot.children]))
			print('Policies: ' + str([str(child.prevMove) + ": " + str(child.policy) for child in treeRoot.children]))
			print('Scores: ' + str([str(child.prevMove) + ": " + str(child.score) for child in treeRoot.children]))
			# exit()
			if returnStatistics:
				policy = np.zeros((7,))
				for child in treeRoot.children:
					policy[child.prevMove] = child.N
				print(policy)
				def normalize(arr):
					theSum = sum(arr)
					if theSum == 0:
						print('ERR: DIVIDE BY ZERO IN NORMALIZATION')
						theSum = 1
					return [x/theSum for x in arr]
				policy = normalize(policy)
				evaluation = bestChild.Q/bestChild.N
				return chosenMove, policy, evaluation
			else:
				return chosenMove

		return treeSearch(board, maxTime, maxSimulations, temperature, verbose)

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
