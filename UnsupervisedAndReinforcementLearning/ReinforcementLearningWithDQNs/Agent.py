import gym
from gym.wrappers import AtariPreprocessing, FrameStack
from Network import Network
import os
from Utils import save_video, VIDEOS_DIR_NAME


RESULTS_PATH, SPECIFIC_NETWORKS_PATH = 'Results', os.path.join('Networks', 'Specific Networks')
DEFAULT_VIDEO_EPISODES, DEFAULT_VIDEO_FRAMES = 10, 400
STACKED_FRAMES = 4
DEFAULT_SCREEN_SIZE = 84

class Agent:
    def __init__(self, env, stacked_frames=STACKED_FRAMES, screen_size = DEFAULT_SCREEN_SIZE, charge_policy = True, train_decoder_first=False):
        """
        Generates
        :param env:
        :param results_path:
        """
        if type(env) == str:
            env = gym.make(env)
        self.env_name = env.spec.id
        if len(env.observation_space.shape) == 3:
            env = AtariPreprocessing(env,noop_max=10, frame_skip=1, scale_obs=True, screen_size=screen_size, terminal_on_life_loss=True)
        self.env = FrameStack(env, num_stack=stacked_frames)
        #self.env = FrameStack(env, num_stack=4)
        self.policy_path = os.path.join(SPECIFIC_NETWORKS_PATH, self.env_name+'.pth') if charge_policy else None
        self.train_encoder_first = train_decoder_first
        self.policy = Network(input_size=self.env.observation_space.shape,num_actions=env.action_space.n,
                              load_from_path=self.policy_path, prepare_conv=train_decoder_first)


    def train(self, results_path=RESULTS_PATH, save_video_result=True, videos_to_save = DEFAULT_VIDEO_EPISODES, save_network = True, visualize=False):
        results_path = os.path.join(results_path, self.env_name, 'ConvBlocked' if self.train_encoder_first else 'ConvNotBlocked')
        # Generate the path where saving the results
        if not os.path.isdir(results_path):
            os.makedirs(results_path)
        # Train the policy
        self.policy.train(env=self.env, plotting_path=results_path, train_conv_first=self.train_encoder_first, show=visualize)

        if save_video_result:
            for video in range(videos_to_save):
                save_video(env=self.env, policy=self.policy,
                    path=os.path.join(os.path.join(results_path, VIDEOS_DIR_NAME, 'Final Result'), 'Game '+str(video)))

        if save_network:
            self.policy.save(self.policy_path)

