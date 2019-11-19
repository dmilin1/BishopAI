import chess
import chess.pgn
import chess.engine
import os
import numpy as np
import random
import h5py
from keras.models import Sequential, Model, model_from_json, load_model
from keras.layers import Dense, Activation, SeparableConv2D, Conv2D, Conv3D, Flatten, BatchNormalization, Dropout, Input, Add, Concatenate, LeakyReLU, GaussianNoise
from keras.optimizers import SGD, Adagrad, Adam
from keras.regularizers import l2
import keras.metrics
import keras.initializers
import keras.callbacks
import keras.backend as K
from scipy.stats import describe
from inputBuilder import buildInput, reverseInput, score


cwd = os.getcwd()

class AI:

	def __init__(self, inputShape=(19,8,8)):
		self.inputShape = inputShape

	def mean_pred(self, y_true, y_pred):
		return K.mean(K.abs(y_pred - y_true))

	def getGames(self, file):
		pgn = open(cwd + '/dataset/' + file + '.pgn')
		data = []
		while 'pigs' != 'fly':
			while len(data) < 25000:
				data += self.getGame(pgn, returnBoards=False)
			random.shuffle(data)
			sampleData = data[-int(len(data)/10):]
			del data[-int(len(data)/10):]
			boards, policies, scores = zip(*sampleData)
			# print('Batch Size: ' + str(len(sampleData)))
			yield np.array(boards), [np.array(scores), np.array(policies)]

	def getGame(self, pgn, returnBoards=False):
		matchData = []
		boards = []
		game = chess.pgn.read_game(pgn)
		try:
			board = game.board()
		except Exception as e:
			print('skipping match')
			print(e)
			return []

		# wElo = int(game.headers['WhiteElo'])
		# bElo = int(game.headers['BlackElo'])
		# eloMinimum = 1400
		# eloDifference = 300
		#
		# if wElo < eloMinimum or bElo < eloMinimum:
		# 	continue
		# if abs(wElo - bElo) > eloDifference:
		# 	continue

		result = game.headers['Result']
		if result == '1-0':
			result = 1
		elif result == '0-1':
			result = -1
		else:
			result = 0
			return []

		moves = list(game.mainline_moves())
		for moveNum, move in enumerate(moves):
			board.push(move)
			if moveNum < 0:
				continue

			policyMove = np.zeros((4096,))
			if (moveNum + 1 < len(moves)):
				policyMove[moves[moveNum+1].from_square*64+moves[moveNum+1].to_square] = 1

			matchData.append([buildInput(board), policyMove, result if board.turn else result * -1])
			# print(chosenMove)

			if returnBoards:
				boards.append(board.copy(stack=0))
		# rand = random.random()
		# boardSet.append(np.full((13, 8, 8), rand))
		# winners.append(round(rand)-0.5)
		if not returnBoards:
			return matchData
		else:
			return matchData, boards

	def newModel(self):
		kernel_reg=0.00001
		seed=1
		kernel_init=keras.initializers.glorot_uniform(seed=seed)
		kernel_init_dense=keras.initializers.glorot_uniform(seed=seed)

		x_in = x = Input(self.inputShape)

		# x = GaussianNoise(0.1)(x)

		x = Conv2D(256, (5,5), kernel_regularizer=l2(kernel_reg), kernel_initializer=kernel_init, padding='same', data_format='channels_first', use_bias=False)(x)

		x = Activation('relu')(x)

		x = BatchNormalization()(x)


		def residualLayers(inputLayering, count):
			if (count == 0):
				return inputLayering

			x = Conv2D(256, (3,3), kernel_regularizer=l2(kernel_reg), kernel_initializer=kernel_init, padding='same', data_format='channels_first', use_bias=False)(inputLayering)

			x = Activation('relu')(x)

			x = BatchNormalization()(x)

			x = Conv2D(256, (3,3), kernel_regularizer=l2(kernel_reg), kernel_initializer=kernel_init, padding='same', data_format='channels_first', use_bias=False)(inputLayering)

			x = Add()([x, inputLayering])

			x = Activation('relu')(x)

			x = BatchNormalization()(x)

			return residualLayers(x, count - 1)


		resid = residualLayers(x, 7)


		x = Conv2D(4, (8,8), kernel_regularizer=l2(kernel_reg), kernel_initializer=kernel_init, padding='same', data_format='channels_first', use_bias=False)(resid)

		x = Activation('relu')(x)

		x = BatchNormalization()(x)

		x = Flatten()(x)

		x = Dense(768, use_bias=False, kernel_regularizer=l2(kernel_reg), kernel_initializer=kernel_init_dense)(x)

		x = Activation('relu')(x)

		x = Dense(512, use_bias=False, kernel_regularizer=l2(kernel_reg), kernel_initializer=kernel_init_dense)(x)

		x = Activation('relu')(x)

		x = Dense(256, use_bias=False, kernel_regularizer=l2(kernel_reg), kernel_initializer=kernel_init_dense)(x)

		x = Activation('relu')(x)

		x = Dense(1, kernel_initializer=kernel_init_dense)(x)

		value_out = Activation('tanh', name='value')(x)


		def policy(inputLayering, name):
			x = Conv2D(8, (8,8), kernel_regularizer=l2(kernel_reg), kernel_initializer=kernel_init, padding='same', data_format='channels_first', use_bias=False)(inputLayering)

			x = Activation('relu')(x)

			x = BatchNormalization()(x)

			x = Flatten()(x)

			x = Dense(512, use_bias=False, kernel_regularizer=l2(kernel_reg), kernel_initializer=kernel_init_dense)(x)

			x = Activation('relu')(x)

			x = Dense(4096, use_bias=False, kernel_regularizer=l2(kernel_reg), kernel_initializer=kernel_init_dense)(x)

			return Activation('softmax', name=name)(x)

		policy_out = policy(resid, 'policy')

		self.model = Model(x_in, [value_out, policy_out])

		print(self.model.summary())

		self.model.compile(loss=['mean_squared_error','categorical_crossentropy'],
					  optimizer=Adam(),
					  metrics={
						'value': self.mean_pred,
						'policy': keras.metrics.categorical_accuracy,
					  })

	def newModelSimple(self):
		kernel_reg=0.00001
		seed=1
		kernel_init=keras.initializers.glorot_uniform(seed=seed)
		kernel_init_dense=keras.initializers.glorot_uniform(seed=seed)

		x_in = x = Input(self.inputShape)

		x = Flatten()(x)

		# x = GaussianNoise(0.1)(x)

		# x = Conv2D(256, (5,5), kernel_regularizer=l2(kernel_reg), kernel_initializer=kernel_init, padding='same', data_format='channels_first', use_bias=False)(x)
		#
		# x = Activation('relu')(x)
		#
		# x = BatchNormalization()(x)


		def residualLayers(inputLayering, count):
			if (count == 0):
				return inputLayering

			x = Conv2D(256, (3,3), kernel_regularizer=l2(kernel_reg), kernel_initializer=kernel_init, padding='same', data_format='channels_first', use_bias=False)(inputLayering)

			x = Activation('relu')(x)

			x = BatchNormalization()(x)

			x = Conv2D(256, (3,3), kernel_regularizer=l2(kernel_reg), kernel_initializer=kernel_init, padding='same', data_format='channels_first', use_bias=False)(inputLayering)

			x = Add()([x, inputLayering])

			x = Activation('relu')(x)

			x = BatchNormalization()(x)

			return residualLayers(x, count - 1)


		resid = residualLayers(x, 0)


		x = Dense(1024, use_bias=False, kernel_regularizer=l2(kernel_reg), kernel_initializer=kernel_init_dense)(resid)

		x = Activation('relu')(x)

		x = Dense(768, use_bias=False, kernel_regularizer=l2(kernel_reg), kernel_initializer=kernel_init_dense)(x)

		x = Activation('relu')(x)

		x = Dense(512, use_bias=False, kernel_regularizer=l2(kernel_reg), kernel_initializer=kernel_init_dense)(x)

		x = Activation('relu')(x)

		x = Dense(256, use_bias=False, kernel_regularizer=l2(kernel_reg), kernel_initializer=kernel_init_dense)(x)

		x = Activation('relu')(x)

		x = Dense(1, kernel_initializer=kernel_init_dense)(x)

		value_out = Activation('tanh', name='value')(x)


		def policy(inputLayering, name):
			x = Dense(1024, use_bias=False, kernel_regularizer=l2(kernel_reg), kernel_initializer=kernel_init_dense)(inputLayering)

			x = Activation('relu')(x)

			x = Dense(512, use_bias=False, kernel_regularizer=l2(kernel_reg), kernel_initializer=kernel_init_dense)(x)

			x = Activation('relu')(x)

			x = Dense(64, use_bias=False, kernel_regularizer=l2(kernel_reg), kernel_initializer=kernel_init_dense)(x)

			return Activation('softmax', name=name)(x)

		policy_from_out = policy(resid, 'policy_from')

		policy_to_out = policy(resid, 'policy_to')

		self.model = Model(x_in, [value_out, policy_from_out, policy_to_out])

		print(self.model.summary())

		self.model.compile(loss='mean_squared_error',
					  optimizer=Adam(),
					  metrics={
						'value': self.mean_pred,
						'policy_from': keras.metrics.categorical_accuracy,
						'policy_to': keras.metrics.categorical_accuracy,
					  })

	def loadModel(self, modelName):
		dependencies = {
			'mean_pred': self.mean_pred,
		}
		self.model = load_model(cwd + '/models/' + modelName + '.h5', custom_objects=dependencies)

	def saveModel(self, modelName):
		print('saving model')
		# Save the weights
		self.model.save(cwd + '/models/' + modelName + '.h5')
		print('model saved')

	# Trains the network on random input and output.
	# For some reason, doing this before running real data makes the network learn faster.
	# I'm not sure why.
	def trainRandom(self, count=1000):
		class MyCustomCallback(keras.callbacks.Callback):

			def on_batch_begin(self, batch, logs=None):
				for layer in self.model.layers:
					if layer.get_config()['name'] == 'target':
						print('')
						print(describe(np.array(layer.get_weights()).flatten()))

		def randomInput():
			while True:
				yield (np.random.rand(*((30,) + self.inputShape))*2-1, [np.random.rand(30,1)*2-1, np.random.rand(30,64), np.random.rand(30,64)])

		self.model.fit_generator(randomInput(), steps_per_epoch=count, epochs=1, callbacks=[MyCustomCallback()])

		for count, (games, results) in enumerate(randomInput()):
			i = random.randint(0, 29)
			print(self.model.predict(np.array([games[i]]))[0])
			print(results[0][i])
			print('')
			if count >= 10:
				break

	def trainHuman(self, file):
		class MyCustomCallback(keras.callbacks.Callback):

			def on_batch_begin(self, batch, logs=None):
				for layer in self.model.layers:
					if layer.get_config()['name'] == 'target':
						print('')
						print(describe(np.array(layer.get_weights()).flatten()))


		self.model.fit_generator(self.getGames(file), steps_per_epoch=250, epochs=14, callbacks=[MyCustomCallback()])

		# for count, (games, results, boards) in enumerate(self.getGame(file, returnBoards=True)):
		# 	i = random.randint(0, 250)
		# 	print(boards[i])
		# 	print(self.model.predict(np.array([games[i]]))[0])
		# 	print(results[0][i])
		# 	print('')
		# 	if count >= 10:
		# 		break

def normalize(arr):
	min = np.amin(arr)
	max = np.amax(arr)
	return (arr-min)/(max-min)


# ai = AI(inputShape=(18,8,8))
# ai.newModel()
# ai.trainRandom(count=1000)
# ai.trainHuman('checkmateonly')
# ai.saveModel('model')

# ai.loadModel('model')
# score, policyFrom, policyTo = ai.model.predict(np.array([buildInput(chess.Board())]))
# print(score)
# print(normalize(policyFrom))
# print(normalize(policyTo))
