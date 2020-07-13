from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from testCase_prioritization.PointWiseEnv import CIPointWiseEnv
import time
from wrapt_timeout_decorator import *


class TPPointWisePPO2Agent:

    def train_agent(self, env: CIPointWiseEnv, steps: int, path_to_save_agent: None, base_model=None):
        env.reset()
        if not base_model:
            base_model = PPO2(MlpPolicy, env, gamma=0.90, learning_rate=0.0005, n_steps=steps)
        else:
            env = DummyVecEnv([lambda: env])
            base_model.set_env(env)
        # check_env(env)
        base_model = base_model.learn(total_timesteps=steps, reset_num_timesteps=False)
        if path_to_save_agent:
            base_model.save(path_to_save_agent)
        return base_model

    # @timeout(500)
    def test_agent(self, env: CIPointWiseEnv, model_path: str, model):
        agent_actions = []
        print("Evaluation of an agent from " + model_path)
        if not model:
            model = PPO2.load(model_path)
            print("Agent is loaded")
        if model:
            test_cases = env.cycle_logs.test_cases
            env = DummyVecEnv([lambda: env])
            model.set_env(env)
            obs = env.reset()
            done = False
            index = 0
            test_cases_vector_prob = []
            for index in range(0, len(test_cases)):
                action, _states = model.predict(obs, deterministic=False)
                print(action)
                obs, rewards, done, info = env.step(action)
                test_cases_vector_prob.append({'index': index, 'prob': action})
                if done:
                    assert len(test_cases) == index+1, "Evaluation is finished without iterating all " \
                                                                    "test cases "
                    break
        return test_cases_vector_prob
