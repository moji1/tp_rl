from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import A2C
from stable_baselines.common.vec_env import DummyVecEnv

from testCase_prioritization.PairWiseEnv import CIPairWiseEnv
import time
from wrapt_timeout_decorator import *


class TPPairWiseA2CAgent:

    def train_agent(self, env: CIPairWiseEnv, steps: int, path_to_save_agent: None, base_model=None,
                    callback_class=None):
        env.reset()
        if not base_model:
            base_model = A2C(MlpPolicy, env, gamma=0.90, learning_rate=0.0005,
                             n_cpu_tf_sess=None, verbose=0,
                             tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
                             full_tensorboard_log=False, seed=None)
        else:
            env = DummyVecEnv([lambda: env])
            base_model.set_env(env)
        # check_env(env)
        base_model = base_model.learn(total_timesteps=steps, reset_num_timesteps=False, callback=callback_class)
        if path_to_save_agent:
            base_model.save(path_to_save_agent)
        return base_model

    # @timeout(500)
    def test_agent(self, env: CIPairWiseEnv, model_path: str, model):
        agent_actions = []
        print("Evaluation of an agent from " + model_path)
        if not model:
            model = A2C.load(model_path)
            print("Agent is loaded")
        if model:
            env = DummyVecEnv([lambda: env])
            model.set_env(env)
            obs = env.reset()
            env.get_attr("test_cases_vector")
            done = False
            while True:
                action, _states = model.predict(obs, deterministic=False)
                print(action)
                obs, rewards, done, info = env.step(action)
                if done:
                    break
            return env.get_attr("sorted_test_cases_vector")[0]

        def __test_agent(self, CIPairWiseEnv, model):

