from gym.wrappers import Monitor
import os
import shutil
from matplotlib import pyplot as plt
import numpy as np

from gym.envs import registry
from atari_py import list_games

MAX_FRAMES = 1000
VIDEOS_DIR_NAME = 'Videos'

def save_video(env, policy, path):
    with Monitor(env=env, directory=path, force=True) as monitor:
        state = monitor.reset()
        cum_reward = 0
        for frame in range(MAX_FRAMES):
            action = policy.get_action(state=state)
            state, reward, done, _ = monitor.step(action)
            cum_reward += reward
            if done:
                print("Video Saved")
                break
    video_name = [file for file in os.listdir(path) if file.endswith('.mp4')][0]
    new_file_name = os.path.join(os.path.dirname(path), os.path.basename(path)+' - Reward '+str(cum_reward)+'.mp4')
    if os.path.isfile(new_file_name):
        os.remove(new_file_name)
    os.rename(os.path.join(path, video_name), new_file_name)
    shutil.rmtree(path)


def save_plot(frame_idx, rewards, losses, path_to_save=None, file_name='Result.png'):
    try:
        plt.figure(figsize=(20,5))
        plt.subplot(211)
        plt.title('Trained for %s frames. Last Reward: %s' % (frame_idx, np.mean(rewards[-10:])))
        plt.plot(rewards[-10000:])
        plt.subplot(212)
        plt.title('Loss')
        plt.plot(losses[-10000:])
        if path_to_save is None:
            plt.show()
        else:
            plt.savefig(os.path.join(path_to_save, file_name), dpi=440)
        plt.close()
    except:
        plt.close()
def get_last_atari_game_versions():
    atari_games = []
    for game in list_games():
        versions = [game_id for game_id in list(registry.env_specs.keys()) if game.replace('_','') + 'noframeskip-v' in game_id.lower()]
        if len(versions)>0:
            atari_games.append(str(np.sort(versions)[-1]))
    return atari_games