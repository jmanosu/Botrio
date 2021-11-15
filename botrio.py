import os

import numpy as np

from argparse import ArgumentParser

import gym
import gym_super_mario_bros
from gym.spaces import Box, MultiBinary
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

from nes_py.wrappers import JoypadSpace

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
 
class MarioWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

        self.action_space = MultiBinary(8)
        self.observation_space = Box(low=0, high=3, shape=(16,16), dtype=np.uint8)

    def step(self, action):
        actionInt = 0 # action comes in as a MultiBinary we need to convert to int
        for i in range(8):
            actionInt = actionInt | (int(action[i]) << i)

        _, reward, done, info = self.env.step(actionInt)

        self.was_real_done = done

        lives = self.env.unwrapped._life
        if self.lives > lives > 0:
            done = True
            
        self.lives = lives

        tiles = self.getTiles()

        return tiles, reward, done, info

    def reset(self, **kwargs):
        if self.was_real_done:
            self.env.reset(**kwargs)
        else:
            self.env.step(0)

        self.lives = self.env.unwrapped._life

        obs = self.getTiles()
        return obs

    def getEnemyLocations(self):
        enemyLocations = []

        for enemyNum in range(5):
            enemy = self.env.unwrapped.ram[0xF + enemyNum]

            if enemy:
                xPosLevel  = self.env.unwrapped.ram[0x6E + enemyNum]
                xPosScreen = self.env.unwrapped.ram[0x87 + enemyNum]

                enemyLocX = (xPosLevel * 0x100) + xPosScreen
                enemyLocY = self.env.unwrapped.ram[0xCF + enemyNum]

                loc = {'x' : enemyLocX, 'y' : enemyLocY}
                enemyLocations.append(loc)
    
        return enemyLocations

    def getTile(self, x, y):
        page = (x // 256) % 2
        subX = (x % 256) // 16
        subY = (y - 32) // 16

        if subY not in range(13):
            return 0

        addr = 0x500 + page*208 + subY*16 + subX

        if self.env.unwrapped.ram[addr] != 0:
            return 1

        return 0

    def getMarioPos(self):
        marioX = self.env.unwrapped.ram[0x06D] * 0x100 + self.env.unwrapped.ram[0x086]
        marioY = self.env.unwrapped.ram[0x3B8]
        return {'x' : marioX, 'y' : marioY}

    def getMarioScreenPos(self):
        marioX = self.env.unwrapped.ram[0x3AD]
        marioY = self.env.unwrapped.ram[0xCE] * self.env.unwrapped.ram[0xB5] + 16
        return {'x' : marioX, 'y' : marioY}

    def getTiles(self):
        tiles = np.zeros(shape=(16,16), dtype=np.uint8)

        row = 0
        col = 0

        marioWorldPos = self.getMarioPos()
        marioScreenPos = self.getMarioScreenPos()

        startX = ((marioWorldPos['x'] - marioScreenPos['x']) // 16) * 16
        startY = 0

        # add tile locations to map
        for yPos in range(startY, 250, 16):
            for xPos in range(startX, startX + 256, 16):
                tile = self.getTile(xPos, yPos)

                if row < 2:
                    tiles[row][col] = 0
                else:
                    tiles[row][col] = tile
                col += 1
            col = 0
            row += 1

        # add enemy location to map
        for loc in self.getEnemyLocations():
            xPos = loc['x'] - startX
            yPos = loc['y']

            row = (yPos + 16) // 16
            col = (xPos + 8) // 16

            if row >= 0 and row < 16 and col >= 0 and col < 16:
                tiles[row][col] = 2

        # add mario location to map
        xPos = marioWorldPos['x'] - startX + 8
        yPos = marioScreenPos['y']

        row = yPos // 16
        col = xPos // 16

        if row >= 0 and row < 16 and col >= 0 and col < 16:
            tiles[row][col] = 3

        return tiles


class CustomReward(gym.Wrapper):
    def __init__(self, env):
        super(CustomReward, self).__init__(env)
        self._current_score = 0
        self._last_x = 0

    def step(self, action):
        state, reward, done, info = self.env.step(action)       
        if done:
            if info['flag_get']:
                reward += 350.0
            else:
                reward -= 50.0

        return state, reward, done, info


def env_creator(env_name):
    env = gym_super_mario_bros.make(env_name)
    env = CustomReward(env)
    env = MarioWrapper(env)
    env = DummyVecEnv([lambda: env])
    return env

def model_creator(env):
    log_path = os.path.join('Training','Logs')
    return PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

def train(model):
    model.learn(total_timesteps=200000)

def trainUntil(env, model):
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)
    eval_callback = EvalCallback(env,
        callback_on_new_best=stop_callback,
        eval_freq=10000,
        verbose=1)
    model.learn(total_timesteps=60000, callback=eval_callback)

def save(model):
    model_path = os.path.join('Training','Models','PPO_botrio')
    model.save(model_path)

def loadModel(env):
    model_path = os.path.join('Training','Models','PPO_botrio')
    return PPO.load(model_path, env=env)

def play(env, model, episodes):
    for episode in range(0, episodes):
        state = env.reset()
        done = False
        score = 0

        while not done:
            env.render()
            action = model.predict(state)
            state, reward, done, info = env.step(action[0])
            score += reward
        print('Episode:{} Score:{}'.format(episode, score))

def parseArgs():
    parser = ArgumentParser(description="Gym Super Mario Bros machine learning trainer.")

    parser.add_argument('--new', type=bool, default=False)
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--evals', type=int, default=10)
    parser.add_argument('--play', type=int, default=5)

    return parser.parse_args()

def main():
    env = env_creator('SuperMarioBros-1-1-v0')

    args = parseArgs()

    model = model_creator(env) if args.new else loadModel(env)

    if args.train:
       train(model)
       save(model)

    if args.play > 0:
       play(env, model, args.play)

    env.close()

if __name__ == "__main__":
    main()