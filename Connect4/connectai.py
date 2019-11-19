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
from connect4 import Connect4

cwd = os.getcwd()


scoreTotal = 0
depthTotal = 0

totalTime = 0




class ConnectAI:

	def __init__(self, ai=None):
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

	def scoreMl(self, board):

		with graph.as_default():
			score, move = self.ai.model.predict(np.array([board.buildInput()]))
			return score, move

	def pickMove(self, board, maxTime=10, maxSimulations=100000, depth=5, temperature=0, verbose=True, returnStatistics=False, solvableMoves=None):
		if verbose:
			print("thinking")
		global totalTime

		def treeSearch(board, maxTime=10, maxSimulations=maxSimulations, maxDepth=4, temperature=temperature, verbose=True, returnStatistics=returnStatistics):
			startTime = time.time()
			global totalTime

			def normalize(arr):
				theSum = sum(arr)
				if theSum == 0:
					print('ERR: DIVIDE BY ZERO IN NORMALIZATION')
					theSum = 1
				return [x/theSum for x in arr]

			def bloomBranch(tree, movePolicies):
				movePoliciesNormalized = np.zeros((7,))
				for move in tree.board.legal_moves():
					movePoliciesNormalized[move] = movePolicies[move]
				movePoliciesNormalized = normalize(movePoliciesNormalized)
				for move in tree.board.legal_moves():
					policy = movePoliciesNormalized[move]
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
					if tree.board.result == None:
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
						score = 1 if tree.board.result != 0 else 0
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

		return treeSearch(board, maxTime=maxTime, maxSimulations=maxSimulations, temperature=temperature, verbose=verbose)

# ai = AI(inputShape=(1,6,7))
# ai.newModelConnect4()
# derp = ConnectAI(ai=ai)
# derp.pickMove(Connect4())
