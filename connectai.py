import random
import time
import os
import numpy as np
import tensorflow as tf
import queue
from Tree import Tree
from SearchTree import SearchTree
from MonteCarloTree import MCTree
from keras.models import model_from_json
from inputBuilder import buildInput
from ai import AI

cwd = os.getcwd()


scoreTotal = 0
depthTotal = 0

totalTime = 0




class ConnectAI:

	def __init__(self, ai=None):
		self.board = chess.Board()
		print('\n~~~~~~\n')
		print(self.board)

		if ai is None:
			self.ai = AI()
			self.ai.loadModel('selftrained')
		else:
			if isinstance(ai, str):
				self.ai = AI()
				self.ai.loadModel(str)
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

	def scoreMl(self, board):
		def normalize(arr):
			min = np.amin(arr)
			max = np.amax(arr)
			return (arr-min)/(max-min)

		with graph.as_default():
			score, moveFrom, moveTo = self.ai.model.predict(np.array([buildInput(board)]))
			return score, moveFrom, moveTo


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


	def pickMove(self, board, maxTime=10, depth=5, temperature=0, verbose=True, returnStatistics=False, solvableMoves=None):
		if verbose:
			print("thinking")
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

		def customTree2(board, maxTime=12, maxDepth=4, solvableMoves=None):
			treeRoot = Tree(score = self.scoreMl(board))
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
					newTree = currentTree.append(self.scoreMl(newBoard), move)
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

		def treeSearch(board, maxTime=10, maxDepth=4, temperature=temperature, verbose=True, returnStatistics=returnStatistics):
			startTime = time.time()
			global totalTime

			def growTree(tree):
				currentBoard = tree.board
				moves = [move for move in currentBoard.legal_moves]
				newTrees = []
				positionsEvaluated = 0
				for move in moves:
					newBoard = tree.board.copy()
					newBoard.push(move)
					score, moveFrom, moveTo = self.scoreMl(newBoard)
					positionsEvaluated += 1
					score = score[0][0]
					moveFrom = moveFrom[0]
					moveTo = moveTo[0]
					policy = (moveFrom[move.from_square] + moveTo[move.to_square])/2
					newTree = SearchTree(board=newBoard, prevMove=move, score=score, policy=policy)
					tree.addChild(newTree)
					newTrees.append(newTree)
				for newTree in newTrees:
					newTree.calcSearchValue()
					q.put(newTree)
				return positionsEvaluated

			def getMovesWithoutRepeat(board):
				moves = []
				legalMoves = list(board.legal_moves)
				for move in board.legal_moves:
					if len(board.move_stack) > 2 and len(legalMoves) > 1 and board.move_stack[-2].from_square == move.to_square and board.move_stack[-2].to_square == move.from_square:
						continue
					moves.append(move)
				return moves

			def bloomBranch(tree, moveFrom, moveTo):
				for move in getMovesWithoutRepeat(tree.board):
					policy = (moveFrom[move.from_square] + moveTo[move.to_square])/2
					newTree = SearchTree(prevMove=move, policy=policy)
					tree.addChild(newTree)

			def rollout(tree):
				if tree.evaluated == False:
					newBoard = tree.parent.board.copy()
					newBoard.push(tree.prevMove)
					score, moveFrom, moveTo = self.scoreMl(newBoard)
					score = score[0][0]
					score = (score + 1)/2
					# print(score)
					# score = (tree.parent.score - score) * (1 if newBoard.turn else -1)
					moveFrom = moveFrom[0]
					moveTo = moveTo[0]
					tree.evaluation(newBoard, score)
					bloomBranch(tree, moveFrom, moveTo)
				else:
					if tree.children:
						rollout(max(tree.children, key=lambda child: child.calcUTC()))
					else:
						if verbose:
							print('End game in sight. Ending Evaluated ' + str(tree.N) + ' times' )
						# print('game over\n---')
						# print(tree.board)
						# print(tree.board.is_game_over())
						result = tree.board.result()
						if result.startswith('1-'):
							score = 1
						elif result.startswith('0-'):
							score = -1
						else:
							score = 0
						tree.backpropogate(score)

			score, moveFrom, moveTo = self.scoreMl(board)
			score = score[0][0]
			score = (score + 1)/2
			moveFrom = moveFrom[0]
			moveTo = moveTo[0]
			treeRoot = SearchTree()
			treeRoot.evaluation(board, score)
			bloomBranch(treeRoot, moveFrom, moveTo)

			while time.time() - startTime < maxTime:
				rollout(treeRoot)

			def balanceTree(tree):
				positionsEvaluated = 0
				if not tree.children:
					if tree.board.turn != board.turn:
						return growTree(tree)
					return 0
				else:
					for child in tree.children:
						positionsEvaluated += balanceTree(child)
				return positionsEvaluated

			# print('balancing tree')
			# positionsEvaluated += balanceTree(treeRoot)

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
			bestChild = max(treeRoot.children, key=lambda child: (1+random.random()*2*temperature-temperature)*child.N)
			chosenMove = bestChild.prevMove
			if returnStatistics:
				stats = normalize([child.N for child in treeRoot.children])
				statSum = sum(stats)
				moveFrom = np.zeros((64))
				moveTo = np.zeros((64))
				for i, child in enumerate(treeRoot.children):
					move = child.prevMove
					moveFrom[move.from_square] += stats[i]
					moveTo[move.to_square] += stats[i]
				moveFrom /= statSum
				moveTo /= statSum
				evaluation = bestChild.Q/bestChild.N
				return chosenMove, moveFrom, moveTo, evaluation
			else:
				return chosenMove

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
				moves = getMovesWithoutRepeat(board)
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

		return treeSearch(board, maxTime=maxTime, temperature=temperature, verbose=verbose)

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
