import threading
import bishopai

class Game(threading.Thread):
    def __init__(self, client, game_id, **kwargs):
        super().__init__(**kwargs)
        self.game_id = game_id
        self.client = client
        self.stream = client.bots.stream_game_state(game_id)
        self.current_state = next(self.stream)
        self.color = self.current_state['white']['id'] == 'bishopai'
        self.game = bishopai.BishopAI(ai='model')

    def play(self):
        if self.game.whosTurn() == self.color:
            move = self.game.getMove(maxTime=12)
            if move:
                if type(move) == list:
                    if move[1] == "draw":
                        self.client.bots.post_message(self.game_id, 'With optimal play, game is drawn.')
                        self.client.bots.make_move(self.game_id, move[0])
                else:
                    self.client.bots.make_move(self.game_id, move)
            else:
                self.client.bots.resign_game(self.game_id)
                print("resigned")

    def run(self):
        self.game.catchUp(self.current_state['state']['moves'])
        print(self.game.score(self.game.board))
        self.play()
        for event in self.stream:
            if event['type'] == 'gameState':
                self.handle_state_change(event)
            elif event['type'] == 'chatLine':
                self.handle_chat_line(event)
            else:
                print(event)

    def handle_state_change(self, game_state):
        self.game.movePlayed(game_state['moves'].split(" ")[-1])
        self.play()

    def handle_chat_line(self, chat_line):
        print(chat_line)
