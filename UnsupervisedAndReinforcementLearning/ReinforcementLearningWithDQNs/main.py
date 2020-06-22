from Agent import Agent
from Utils import get_last_atari_game_versions

if __name__ == '__main__':
    list_games = get_last_atari_game_versions()
    list_games = ['Pong-v4', 'Breakout-v4', 'Assault-v4', 'Boxing-v4', 'Alien-v4', 'DemonAttack-v4',]
    for game in list_games:
        Agent(game, train_decoder_first=False).train(save_network=True, visualize=False)