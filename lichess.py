import berserk
import gameInstance


token = 'XITpex1q2NiO9SjR'

session = berserk.TokenSession(token)
client = berserk.Client(session)




def should_accept(event):
    if event['challenge']['challenger']['id'] == 'dmilin' or True:
        return True
    else:
        return False

is_polite = True
for event in client.bots.stream_incoming_events():
    if event['type'] == 'challenge':
        if should_accept(event):
            client.bots.accept_challenge(event['challenge']['id'])
        elif is_polite:
            client.bots.decline_challenge(event['challenge']['id'])
    elif event['type'] == 'gameStart':
        print(event)
        game = gameInstance.Game(client, event['game']['id'])
        game.start()
    else:
        print(event)
