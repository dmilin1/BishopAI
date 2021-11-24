# BishopAI

------------

BishopAI has been succeeded by the [knAIghtedBishop](https://github.com/dmilin1/knAIghtedBishop) project.

------------

BishopAI is a flexible software suite capable of generating machine learning models and using them to solve 2 player, sequential, [perfect information](https://en.wikipedia.org/wiki/Perfect_information "perfect information") games. The included models are capable of playing Chess and Connect4 at an above average level.

Networks can be trained in two ways. The first is with a dataset of preplayed games, as is used in the Chess model. The chess model was trained for around 30 minutes on roughly 160,000 preplayed games. The second is with self play, as is used in the Connect 4 model. The included level 3 model Connect 4 was trained over roughly 1 week on an Nvidia GTX 1080 Ti.

This version of BishopAI additionally includes two human interface layers so humans can play against the Chess and Connect 4 networks. The Chess interface layer takes advantage of [Lichess.com](https://lichess.org/@/BishopAI "Lichess.com")&#39;s API and allows Lichess account owners to play against the bot on the website so long as the bot is running live on a computer somewhere. The Connect 4 interface layer is a simple command line GUI allowing players to view the game and select a row to play in.

### Requirements

- Python 3.6
- Tensorflow (GPU optional but highly recommended)
- The following pip installable python packages: python-chess, numpy, h5py, keras, tensorflow, scipy, berserk
- a [Lichess.com](https://lichess.org/ "Lichess.com") account


### How To Use

Make sure all the requirements have been met. The setup steps for the requirements differ heavily between systems so this guide will only cover information assuming the requirements have been properly met.


------------



To play Chess:

1. Clone the [GitHub project](https://github.com/dmilin1/BishopAI "GitHub project") to a file location of your choosing.

2. cd into the downloaded directory.

3. start the lichess server with "py lichess.py"

4. Log into Lichess.com and [visit this link](https://lichess.org/@/BishopAI "visit this link").

5. Click the icon with crossed swords in the top right corner to challenge the bot and choose which side to play. If everything is set up properly, the bot should accept the challenge and gameplay should begin.

6. The bot is configured by default to think for 12 seconds before chosing a move. The skill level this represents differs based on the computational power of the computer running the program. The thinking time can be changed by modifying line 16 in "gameInstance.py".


------------


To play Connect 4:

1. Clone the [GitHub project](https://github.com/dmilin1/BishopAI "GitHub project") to a file location of your choosing.

2. cd into the "Connect 4" file inside the downloaded directory.

3. start a Connect 4 game with the command "py test.py"

4. Follow the command line instructions to play the game.
