import gym
import gym_cityflow
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.env_checker import check_env
from stable_baselines import DQN
from datetime import datetime

if __name__ == "__main__":
    env = gym.make('gym_cityflow:CityFlow-1x1-LowTraffic-v0')
    #check_env(env)
    model = DQN(MlpPolicy, env, verbose=1)
    log_interval = 10
    total_episodes = 200
    model.learn(total_timesteps=env.steps_per_episode * total_episodes, log_interval=log_interval)
    now = datetime.now()
    fileName = now.strftime("%m%d%Y_%H%M%S")
    model.save(rf"cityFlowModels/{fileName}")