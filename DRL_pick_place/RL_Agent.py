import torch
import random
import numpy as np
from collections import deque
from rl_env import GripperGame, Direction, Point
from rl_model import Linear_QNet, QTrainer
from plotter import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 5000
# BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        gripper = game.gripper
        point_l = Point(gripper.x - 20, gripper.y)
        point_r = Point(gripper.x + 20, gripper.y)
        point_u = Point(gripper.x, gripper.y - 20)
        point_d = Point(gripper.x, gripper.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(gripper = point_r)) or 
            (dir_l and game.is_collision(gripper = point_l)) or 
            (dir_u and game.is_collision(gripper = point_u)) or 
            (dir_d and game.is_collision(gripper = point_d)),

            # Danger right
            (dir_u and game.is_collision(gripper = point_r)) or 
            (dir_d and game.is_collision(gripper = point_l)) or 
            (dir_l and game.is_collision(gripper = point_u)) or 
            (dir_r and game.is_collision(gripper = point_d)),

            # Danger left
            (dir_d and game.is_collision(gripper = point_r)) or 
            (dir_u and game.is_collision(gripper = point_l)) or 
            (dir_r and game.is_collision(gripper = point_u)) or 
            (dir_l and game.is_collision(gripper = point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # packet center location 
            # game.packet[2].x < game.gripper.x,
            # game.packet[2].x > game.gripper.x,
            # game.packet[2].y < game.gripper.y,
            # game.packet[2].y > game.gripper.y 
            (game.packet[2].x < game.gripper.x) or
            (game.packet[3].x < game.gripper.x),  # packet center left
            (game.packet[2].x > game.gripper.x) or
            (game.packet[3].x > game.gripper.x),  # packet center right
            (game.packet[2].y < game.gripper.y) or
            (game.packet[3].y < game.gripper.y),  # packet center up
            (game.packet[2].y > game.gripper.y) or
            (game.packet[3].y > game.gripper.y)  # packet center down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        # self.epsilon = 80 - self.n_games
        self.epsilon = 200 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = GripperGame()
    # game = GripperGame(w=960)
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state = state_old, 
                                action = final_move,
                                reward = reward, 
                                next_state = state_new,
                                done = done)
        # remember
        agent.remember(state = state_old,
                    action = final_move,
                    reward = reward,
                    next_state = state_new,
                    done = done)

        if done:
            # train long memory, plot result
            game.reset_env()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()