import chess
import chess.engine
import os
import time
import random
from keras.models import model_from_json
import keras.callbacks
from inputBuilder import buildInput
import numpy as np
from bishopai import BishopAI
from connectai import ConnectAI
from connect4 import Connect4, HumanInterface
from ai import AI

cwd = os.getcwd()

# TRY THOMPSON SAMPLING TO SOLVE MULTI ARMED BANDIT PROBLEM
# https://lilianweng.github.io/lil-log/2018/01/23/the-multi-armed-bandit-problem-and-its-solutions.html
# https://sudeepraja.github.io/Bandits/

def generateGame():
	board = chess.Board()
	moveCount = 0
	result = '*'
	matchData = []
	while result == '*':
		moveCount += 1
		result = board.result()
		if moveCount > 350:
			result = '1/2-1/2'
			break
		if result != '*':
			if result.startswith('1-'):
				eval = 1
			elif result.startswith('0-'):
				eval = -1
			else:
				eval = 0
			matchData.append([buildInput(board), np.zeros((64)), np.zeros(64), eval, board.turn])
			break
		# print('\n\n')
		# print(board)
		move, moveFrom, moveTo, evaluation = bishopAi.pickMove(board, maxTime=2, temperature=5/moveCount**2, returnStatistics=True, verbose=False)
		matchData.append([buildInput(board), moveFrom, moveTo, evaluation, board.turn])
		board.push(move)

	if result.startswith('1-'):
		print('\n\nWhite wins in ' + str(moveCount) + ' moves')
		for data in matchData:
			data[3] = (data[3] + (1 if data[4] else -1))/2
	elif result.startswith('0-'):
		print('\n\nBlack wins in ' + str(moveCount) + ' moves')
		for data in matchData:
			data[3] = (data[3] + (-1 if data[4] else 1))/2
	else:
		print('\n\nTie in ' + str(moveCount) + ' moves')
		for data in matchData:
			data[3] = (data[3] + 0)/2

	print(board)
	return matchData

def generateSelfTrainData():
	data = []
	while 'pigs' != 'fly':
		data += generateGame()
		random.shuffle(data)
		sampleData = data[-int(len(data)/5):]
		del data[-int(len(data)/5):]
		boards, moveFroms, moveTos, scores, _ = zip(*sampleData)
		print('Batch Size: ' + str(len(sampleData)))
		yield np.array(boards), [np.array(scores), np.array(moveFroms), np.array(moveTos)]


def generateGameConnect4(aiLearn, aiPlay):
	board = Connect4()
	matchData = []
	moveCount = 0
	learnSide = random.random() > 0.5
	while board.result == None:
		moveCount += 1
		# print('\n\n')
		# print(board)
		if board.turn == learnSide:
			player = aiLearn
		else:
			player = aiPlay
		move, policy, evaluation = player.pickMove(board, maxTime=1.25, maxSimulations=800, temperature=1, returnStatistics=True, verbose=False)
		matchData.append([board.buildInput(), policy, evaluation, board.turn])
		board.push(move)

		print('Last Move: ' + str(move))
		print('Last Probability: ' + str(policy))
		print('Learner: ' + str(board.turn != learnSide))
		print('\n')

	if (board.result == 1) == learnSide:
		print('Learning Side Won')
	else:
		print('Learning Side Lost')

	if board.result == 1:
		print('\n\nFirst player wins in ' + str(moveCount) + ' moves')
		for data in matchData:
			data[2] = (-((moveCount-7)/175)+1) * (-1 if data[3] else 1)
	elif board.result == -1:
		print('\n\nSecond player wins in ' + str(moveCount) + ' moves')
		for data in matchData:
			data[2] = (-((moveCount-7)/175)+1) * (-1 if data[3] else 1)
	else:
		print('\n\nTie in ' + str(moveCount) + ' moves')
		for data in matchData:
			data[2] = 0

	print(board)
	return matchData

def generateSelfTrainDataConnect4(aiLearn, aiPlay):
	data = []
	while 'pigs' != 'fly':
		while len(data) < 5000:
			data += generateGameConnect4(aiLearn, aiPlay)
		random.shuffle(data)
		sampleData = data[-int(len(data)/5):]
		del data[-int(len(data)/5):]
		boards, policies, scores, _ = zip(*sampleData)
		print('Batch Size: ' + str(len(sampleData)))
		yield np.array(boards), [np.array(scores), np.array(policies)]


class SaveModel(keras.callbacks.Callback):
	def on_batch_end(self, batch, logs=None):
		if (batch % 1 == 0):
			ai.saveModel('selftrained')

def selfLearn(aiLearn, aiPlay, game="chess"):
	tensorboard = keras.callbacks.TensorBoard(log_dir=cwd+"/tensorboard/logs/" + str(time.time()), update_freq='batch')
	if game == "chess":
		ai.model.fit_generator(generateSelfTrainData(), steps_per_epoch=250, epochs=1, callbacks=[SaveModel(), tensorboard])
	if game == "connect4":
		aiLearn.ai.model.fit_generator(generateSelfTrainDataConnect4(aiLearn, aiPlay), steps_per_epoch=250, epochs=1000, callbacks=[SaveModel(), tensorboard])

# ai = AI(inputShape=(19,8,8))
# ai.newModel()
# ai.trainRandom(count=10)
# ai.trainHuman('checkmateonly')
# ai.saveModel('model')

# ai = AI(inputShape=(18,8,8))
# ai.loadModel('selftrained')
# bishopAi = BishopAI(ai=ai)
# selfLearn()

# ai = AI(inputShape=(1,6,7))
# # ai.newModelConnect4()
# ai.loadModel('selftrained')
# aiLearn = ConnectAI(ai=ai)
# ai2 = AI(inputShape=(1,6,7))
# # ai2.newModelConnect4()
# ai2.loadModel('selftrained')
# aiPlay = ConnectAI(ai=ai2)
# selfLearn(aiLearn, aiPlay, game="connect4")
# # generateGameConnect4()


ai = AI(inputShape=(1,6,7))
# ai.loadModel('selftrained')
ai.newModelConnect4()
connectAI = ConnectAI(ai=ai)
board = Connect4()
while board.result == None:
	if not board.turn:
		move = HumanInterface(board).requestMove()
	else:
		move = connectAI.pickMove(board, maxTime=5)
	board.push(move)
if board.result == 1:
	print('X (Player 1) WINS!')
elif board.result == -1:
	print('O (Player 2) WINS!')
else:
	print("IT'S A TIE")


# input = np.array([buildInput(chess.Board())])
# bigInput = np.array([buildInput(chess.Board()) for i in range(100)])
# total = 0
# for i in range(100):
# 	start = time.time()
# 	ai.model.predict(input)
# 	end = time.time()
# 	timeTotal = end-start
# 	if i != 0:
# 		total += timeTotal
#
# print(total/99)
#
# total = 0
# for i in range(100):
# 	start = time.time()
# 	ai.model.predict(bigInput)
# 	end = time.time()
# 	timeTotal = end-start
# 	if i != 0:
# 		total += timeTotal
#
# print(total/99)
