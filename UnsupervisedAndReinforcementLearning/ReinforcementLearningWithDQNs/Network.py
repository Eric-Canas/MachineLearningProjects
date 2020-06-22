import math, random
import os
import numpy as np
from Buffer import ReplayBuffer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from Utils import save_video, VIDEOS_DIR_NAME, save_plot
import warnings
from Raimbow import Raimbow
ACTIONS_PER_TRAIN_STEP = 4

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    print("Using GPU Environment")

DEFAULT_BATCH_SIZE, DEFAULT_BUFFER_SIZE, DEFAULT_GAMMA = 32, 50000, 0.99
DEFAULT_NUM_FRAMES, DEFAULT_DQN_UPDATE_RATIO = 8000000, 10000
PLOT_EVERY = 100000
DEFAULT_VIDEOS_TO_SAVE = 50
CONV_TRAINING_FRAMES = 10000
MIN_RANDOM_ACTIONS = 10000
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

EPSILON_START, EPSILON_FINAL, EPSILON_DECAY = 1., 0.02, 20000

class Network:
    def __init__(self, input_size, num_actions, gamma=DEFAULT_GAMMA, buffer_size=DEFAULT_BUFFER_SIZE,
                 batch_size=DEFAULT_BATCH_SIZE, load_from_path = None, prepare_conv=False):
        """
        Include the double network and is in charge of train and manage it
        :param input_size:
        :param num_actions:
        :param buffer_size: int. Size of the replay buffer
        :param batch_size: int. Size of the Batch
        """
        # Instantiate both models
        net = Raimbow if len(input_size) == 3 else DQN
        self.current_model = net(input_size=input_size, num_actions=num_actions, prepare_decoder=prepare_conv)
        if load_from_path is not None:
            self.load_weights(path=load_from_path)
        self.target_model = net(input_size=input_size, num_actions=num_actions, prepare_decoder=prepare_conv)

        # Put them into the GPU if available
        if USE_CUDA:
            self.current_model = self.current_model.cuda()
            self.target_model = self.target_model.cuda()

        # Initialize the Adam optimizer and the replay buffer
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.current_model.parameters()),lr=0.00001)
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

        # Make both networks start with the same weights
        self.update_target()

        # Save the rest of parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.input_channels = input_size

    def get_action(self, state):
        return self.current_model.act(state, epsilon=0.)

    def update_target(self):
        """
        Updates the target model with the weights of the current model
        """
        self.target_model.load_state_dict(self.current_model.state_dict())

    def compute_td_loss(self, samples):
        """
        Compute the loss of batch size samples of the buffer, and train the current model network with that loss
        :param samples: tuple of samples. Samples must have the format (state, action, reward, next_state, done)
        :return:
        float. Loss computed at this learning step
        """
        # Take N playing samples
        state, action, reward, next_state, done = samples

        # Transform them into torch variables, for being used on GPU during the training
        state = Variable(torch.FloatTensor(np.float32(state)))
        next_state = Variable(torch.FloatTensor(np.float32(next_state)))
        action = Variable(torch.LongTensor(action))
        reward = Variable(torch.FloatTensor(reward))
        done = Variable(torch.FloatTensor(done))

        # Get the q value of this state and all the q values of the following step
        q_value = self.current_model(state).gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_values = self.current_model(next_state)
        # Get the q values of the following step following the static policy of the target model
        next_q_state_values = self.target_model(next_state)
        # For all the q_values of the next state get the one of the action which would be selected by the static policy
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)

        # Calculate the expected q value as the inmediate reward plus gamma by the expected reward at t+1 (if not ended)
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        # Calculate the Mean Square Error
        loss = nn.functional.smooth_l1_loss(q_value, Variable(expected_q_value.data))

        # Backpropagates the loss
        self.optimizer.zero_grad()
        loss.backward()
        # Learn
        self.optimizer.step()

        # Return the loss of this step
        return loss

    def compute_conv_loss(self, frames):
        """
        Compute the loss of batch size samples of the buffer, and train the current model network with that loss
        :param samples: tuple of samples. Samples must have the format (state, action, reward, next_state, done)
        :return:
        float. Loss computed at this learning step
        """

        # Transform them into torch variables, for being used on GPU during the training
        state = Variable(torch.FloatTensor(frames), requires_grad=True)

        loss = (state - self.current_model.forward(state)).pow(2).mean()

        # Backpropagates the loss
        self.optimizer.zero_grad()
        loss.backward()
        # Learn
        self.optimizer.step()

        # Return the loss of this step
        return loss

    def train_convolutional_part(self, env, n_frames, print_state_every=100):
        self.current_model.mode_enc_dec = True
        # Take a random action
        action = self.current_model.act(state=None, epsilon=1.)
        state = env.reset()
        states_buffer = ReplayBuffer(capacity=1000)
        losses = []
        for i in range(n_frames):
            next_state, reward, done, _ = env.step(action)
            states_buffer.push(state, action, reward, next_state, done)

            if n_frames%4 == 0:
                action = self.current_model.act(state=None, epsilon=1.)

            if done:
                print("Episode done during Encoder Decoder Training")
                state = env.reset()
            if len(states_buffer)>self.batch_size:
                # Train
                loss = self.compute_conv_loss(states_buffer.state_sample(batch_size=self.batch_size))
                # Save the loss
                losses.append(loss.item())
            if i%print_state_every==0 and len(losses)>1:
                print("Training Encoder Decoder. Step:"+str(i)+"/"+str(n_frames)+". "
                     "Mean Loss: "+str(np.round(np.mean(losses[-10:]), decimals=5)))
        for param in self.current_model.encoder.parameters():
            param.requires_grad = False
        self.current_model.mode_enc_dec = False
        self.update_target()




    def epsilon_by_frame(self, frame_idx, epsilon_start = EPSILON_START, epsilon_final = EPSILON_FINAL, epsilon_decay = EPSILON_DECAY):
        """
        Gets the epsilon of the current frame for the given parameters
        :param frame_idx: int. Index of the frame
        :param epsilon_start: float. Epsilon at frame 1
        :param epsilon_final: float. Minimum epsilon for maintaining exploration
        :param epsilon_decay: int. Manages how fast the epsilon decays
        :return:
        Epsilon for the frame frame_idx
        """
        return epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

    def train(self, env, num_frames = DEFAULT_NUM_FRAMES, DQN_update_ratio=DEFAULT_DQN_UPDATE_RATIO, plotting_path = None,
              verbose=True, videos_to_save=DEFAULT_VIDEOS_TO_SAVE, train_conv_first = True, show=False):
        """
        Train the network in the given environment for an amount of frames
        :param env:
        :param num_frames:
        :return:
        """
        if train_conv_first:
            self.train_convolutional_part(env=env, n_frames=CONV_TRAINING_FRAMES)
        # Save the losses of the network and the rewards of each episode
        losses, all_rewards = [], []
        episode_reward = 0

        # Reset the game for starting the game from 0
        state = env.reset()
        actions_taken = []
        for i in range(MIN_RANDOM_ACTIONS):
            action = self.current_model.act(state, epsilon=1.)
            next_state, reward, done, _ = env.step(action)
            self.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            if done:
                env.reset()
        for frame_idx in range(1, num_frames + 1):
            # Gets an action for the current state having in account the current epsilon
            action = self.current_model.act(state, epsilon=self.epsilon_by_frame(frame_idx=frame_idx))
            actions_taken.append(action)
            if show:
                env.render()
            # Execute the action, capturing the new state, the reward and if the game is ended or not
            next_state, reward, done, _ = env.step(action)
            # Save the action at the replay buffer
            self.replay_buffer.push(state, action, reward, next_state, done)
            # Update the current state and the actual episode reward
            state = next_state
            episode_reward += reward

            # If a game is finished save the results of that game and restart the game
            if done:
                print("Episode Reward: "+str(episode_reward)+".  "
                    "Std of actions: "+str(np.round(np.std(actions_taken), decimals=4))+". " 
                    "Epsilon "+str(np.round(self.epsilon_by_frame(frame_idx=frame_idx), decimals=3)))
                actions_taken = []
                all_rewards.append(episode_reward)
                state, episode_reward = env.reset(), 0

            # If there are enough actions in the buffer for learning, start to learn a policy
            if frame_idx% ACTIONS_PER_TRAIN_STEP ==0:
                # Train
                loss = self.compute_td_loss(self.replay_buffer.sample(self.batch_size))
                # Save the loss
                losses.append(loss.item())

            if plotting_path is not None and frame_idx % PLOT_EVERY == 0:
                save_plot(frame_idx, all_rewards, losses, path_to_save=plotting_path)

            if frame_idx % DQN_update_ratio == 0:
                self.update_target()

                if verbose and frame_idx% DQN_update_ratio == 0:
                    print(env.unwrapped.spec.id+' Training: '+str(frame_idx)+'/'+str(num_frames)+'. '
                            'Mean Rewards: '+str(np.round(np.mean(all_rewards[-10:]), decimals=2)))

            if frame_idx%(num_frames//videos_to_save) == 0:
                save_video(env=env, policy=self,  path=os.path.join(plotting_path,VIDEOS_DIR_NAME,'During Training',
                                                                    str(len(all_rewards))+' Games'))
                env.reset()

    def save(self, path):
        if not os.path.isdir(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        self.current_model.cpu()
        torch.save(self.current_model.state_dict(), path)
        if USE_CUDA:
            self.current_model.cuda()

    def load_weights(self, path):
        if not os.path.isfile(path):
            warnings.warn("Trying to charge non existent weights. Skipping")
        else:
            self.current_model.cpu()
            output_state_dict = torch.load(path)
            new_dict = {key : (output_state_dict[key] if key in output_state_dict else value) for key, value in self.current_model.state_dict().items()}
            self.current_model.load_state_dict(new_dict)

            for param in self.current_model.parameters():
                param.requires_grad = False

            for param in self.current_model.pre_output.parameters():
                param.requires_grad = True
            for param in self.current_model.output.parameters():
                param.requires_grad = True

            if USE_CUDA:
                self.current_model.cuda()
        return self.current_model

class DQN(nn.Module):
    def __init__(self, input_size, num_actions):
        super(DQN, self).__init__()

        self.build_network(input_size=input_size[0]*input_size[1], num_actions=num_actions)
        self.num_actions = num_actions
        self.input_size = input_size

    def build_network(self, input_size, num_actions):
        self.linear = nn.Sequential(nn.Linear(input_size, 1024),
                                    nn.ReLU(),
                                    nn.Linear(1024, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, num_actions))

    def forward(self, x):
        return self.linear(x.view(x.size(0), -1)/255)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(state).unsqueeze(0), requires_grad=False)
            q_value = self.forward(state)
            action = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.num_actions)
        return action


class CNN_DQN(nn.Module):
    def __init__(self, input_size, num_actions, mode_enc_dec=False, prepare_decoder=False):
        super(CNN_DQN, self).__init__()

        self.prepare_decoder = prepare_decoder
        self.build_network(input_size=input_size, num_actions=num_actions)
        self.num_actions = num_actions
        self.input_size = input_size
        self.mode_enc_dec = mode_enc_dec

    def build_network(self, input_size, num_actions):
            c, h, w = input_size
            # Encoder
            self.encoder = nn.Sequential(nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8,stride=4, bias=False),
                                          nn.ReLU(),
                                          nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, bias=False),
                                          nn.ReLU(),
                                          nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=1, bias=False),
                                          nn.ReLU())
            for layer, weights in self.encoder.state_dict().items():
                if 'weight' in layer:
                    torch.nn.init.kaiming_normal_(weights,nonlinearity='relu')

            if self.prepare_decoder:
                self.decoder = nn.Sequential(
                                             nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2,stride=1, bias=False),
                                             nn.ReLU(),
                                             nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, bias=False),
                                             nn.ReLU(),
                                             nn.ConvTranspose2d(in_channels=32, out_channels=c, kernel_size=8, stride=4, bias=False),
                                             nn.ReLU())
                for layer, weights in self.decoder.state_dict().items():
                    if 'weight' in layer:
                        torch.nn.init.kaiming_normal_(weights,nonlinearity='relu')

            # Linear
            self.linear = nn.Sequential(nn.Linear(64*8*8, 512),
                                        nn.ReLU(),
                                        nn.Linear(512, num_actions))

            for layer, weights in self.linear.state_dict().items():
                if 'weight' in layer:
                    torch.nn.init.kaiming_normal_(weights, nonlinearity='relu')

    def forward(self, x):
        x = self.encoder(x)
        if self.mode_enc_dec:
            x = self.decoder(x)
        else:
            x = x.view(x.size(0), -1)
            x = self.linear(x)
        return x

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(state).unsqueeze(0), requires_grad=False)
            q_value = self.forward(state)
            action = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.num_actions)
        return action

