import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
import datetime
import tensorflow.summary as summary_writer
from air_raid_utils import get_stacked_frames

class DQN_Agent:
    #
    # Initializes attributes and constructs CNN model and target_model
    #
    def __init__(self, env, state_size, action_size,num_episodes,num_steps,batch_size, gamma:float = 0.985, memory_size:int = 10000,update_rate:int = 10):
        self.state_size = state_size
        #print(state_size)
        #print(batch_size)
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.env = env
        self.episodes = num_episodes
        self.max_steps = num_steps 
        self.stack_size = 4
        self.batch_size = batch_size
        # Hyperparameters
        self.gamma = gamma            # Discount rate
        self.epsilon = 1.0          # Exploration rate
        self.epsilon_min = 0.05      # Minimal exploration rate (epsilon-greedy)
        self.epsilon_decay = 0.995  # Decay rate for epsilon
        self.update_rate = update_rate # Number of steps until updating the target network
        self.skip_start = 90
        self.model = self._build_model() # Q Network
        self.target_model = self._build_model() # Target Network
        self.target_model.set_weights(self.model.get_weights())
        self.model.summary()

    #
    # Constructs CNN
    #
    def _build_model(self):
        model = Sequential()
        
        # Conv Layers
        model.add(Conv2D(32, (8, 8), strides=4, padding='same', input_shape=self.state_size))
        model.add(Activation('relu'))
        
        model.add(Conv2D(64, (4, 4), strides=2, padding='same'))
        model.add(Activation('relu'))
        
        model.add(Conv2D(64, (3, 3), strides=1, padding='same'))
        model.add(Activation('relu'))
        model.add(Flatten())

        # FC Layers
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        
        model.compile(loss='huber', optimizer=Adam())
        return model

    #
    # Stores experience in replay memory
    #
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    #
    # Chooses action based on epsilon-greedy policy
    #
    def act(self, state):
        # Random exploration
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state, verbose=0)
        
        return np.argmax(act_values[0])  # Returns action using policy
    
    def greedy_act(self, state):
        act_values = self.model.predict(state, verbose=0)
        
        return np.argmax(act_values[0])  # Returns action using policy
    
    

    #
    # Trains the model using randomly selected experiences in the replay memory
    #
    def replay(self, batch_size,total_time,file_writer):
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            #print("inside replay state shape ",state.shape)
            #print("inside replay next state shape ",next_state.shape)
            if not done:
                target = (reward + self.gamma * np.max(self.target_model.predict(next_state, verbose=0)))
            else:
                target = reward
                
            # Construct the target vector as follows:
            # 1. Use the current model to output the Q-value predictions
            target_f = self.model.predict(state, verbose=0)
            #print(target)
            
            # 2. Rewrite the chosen action value with the computed target
            target_f[0][action] = target
            #print(target_f)
            #print(target_f[0][action])
            
            # 3. Use vectors in the objective computation
            self.model.fit(state, target_f, epochs=1, verbose=0)
            # Log loss to TensorBoard
            with file_writer.as_default():
                #mse_loss = np.mean(np.square(target_f[0][action] - target))
                huber_loss = np.mean(tf.keras.losses.huber(tf.expand_dims(target_f[0][action], 0), tf.expand_dims(target_f, 0)))
                #summary_writer.scalar('MSE Loss', mse_loss, step=total_time)
                tf.summary.scalar('Huber Loss', huber_loss, step=total_time)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    #
    # Sets the target model parameters to the current model parameters
    #
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
            
    #
    # Loads a saved model
    #
    def load(self, name):
        self.model.load_weights(name)
        self.target_model.load_weights(name)

    #
    # Saves parameters of a trained model
    #
    def save(self, name):
        self.model.save_weights(name)
    
    def train(self):
        total_time = 0
        total_reward = 0
        all_rewards = 0
        sum_5_games = 0
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        train_log_dir = './logs/' + current_time + '/train'
        file_writer = tf.summary.create_file_writer(train_log_dir)
        for e in range(self.episodes):
            frames = deque(maxlen=4)  # Initialize frames deque for stacking
            state = self.env.reset()
            # Initialize the frame stack
            state, frames = get_stacked_frames(frames, state[0], is_new_episode=True)

            for _ in range(self.skip_start):
                self.env.step(0)

            game_score = 0

            for time in range(self.max_steps):
                total_time += 1
                
                # Transition Dynamics
                action = self.act(state)
                next_state, reward, done, _, _ = self.env.step(action)
                
                # Get the stacked frames for next state
                next_state, frames = get_stacked_frames(frames, next_state, is_new_episode=False)
                
                # Store sequence in replay memory
                self.remember(state, action, reward, next_state, done)
                state = next_state
                game_score += reward
                total_reward += reward

                if done:
                    all_rewards += game_score
                    sum_5_games += game_score
                    avg_5_games = sum_5_games / (e % 5 + 1)

                    print(f"episode: {e+1}/{self.episodes}, game score: {game_score}, avg_10: {avg_5_games}, reward: {total_reward}, avg reward: {all_rewards / (e+1)}, time: {time}, total time: {total_time}")

                    break

                if len(self.memory) > self.batch_size * 2 and total_time % 25 == 0:
                    self.replay(self.batch_size,total_time,file_writer)

            if total_time % self.update_rate == 0:
                self.update_target_model()
            with file_writer.as_default():
                tf.summary.scalar('game_score', game_score, step=e)
                tf.summary.scalar('Average Reward', all_rewards / (e + 1), step=e)
                tf.summary.scalar('Exploration Rate', self.epsilon, step=e)
            if e % 10 == 0:
                fname = f'models/10k-memory_{e}-games.weights.h5'
                print(f'Saving: {fname}')
                self.save(fname)
                sum_10_games = 0

            if game_score > 15:
                break