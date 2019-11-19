import chess
import chess.pgn
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, SeparableConv2D, Conv2D, Conv3D, Flatten
from keras.optimizers import SGD
from inputBuilder import buildInput


cwd = os.getcwd()

pgn = open(cwd + '/dataset.pgn')



def getGame():
    while True:
        boardSet = []
        winners = []
        for i in range(10):
            game = chess.pgn.read_game(pgn)
            board = game.board()

            result = game.headers['Result']
            if result == '1-0':
                result = 1
            elif result == '0-1':
                result = -1
            else:
                result = 0

            for move in game.mainline_moves():
                board.push(move)
                boardSet.append(buildInput(board))
                winners.append(result)

        yield (np.array(boardSet), np.array(winners), )



model = Sequential([
    Conv2D(256, (4,4), input_shape=(13,8,8), data_format='channels_first'),
    Activation('relu'),
    Conv2D(256, (3,3), data_format='channels_first'),
    Activation('relu'),
    Conv2D(256, (3,3), data_format='channels_first'),
    Activation('relu'),
    Flatten(),
    Dense(128),
    Activation('relu'),
    Dense(64),
    Activation('relu'),
    Dense(1),
    Activation('sigmoid'),
])

print(model.summary())

model.compile(loss='mean_squared_error',
              optimizer=SGD(lr=0.01, momentum=0.01, decay=0.001, nesterov=False),
              metrics=['accuracy'])

model.fit_generator(getGame(), steps_per_epoch=1000, epochs=100)
