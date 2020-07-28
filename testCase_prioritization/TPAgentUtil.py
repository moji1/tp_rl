#
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import DummyVecEnv

from PairWiseEnv import CIPairWiseEnv


class TPAgentUtil:
    #supported_algo = ['DQN', 'PPO2', "A2C", "ACER", "ACKTR", "HER", "PPO1", "TRPO"]
    supported_algo = ['DQN', 'PPO2', "A2C", "ACKTR", "DDPG", "ACER", "GAIL", "HER", "PPO1", "SAC", "TD3", "TRPO"]

    def create_model(algo, env):
        assert TPAgentUtil.supported_algo.count(algo.upper()) == 1, "The algorithms  is not supported for" \
                                                                    " pairwise formalization"
        if algo.upper() == "DQN":
            from stable_baselines import DQN
            from stable_baselines.deepq.policies import MlpPolicy
            model = DQN(MlpPolicy, env, gamma=0.90, learning_rate=0.0005, buffer_size=10000,
                        exploration_fraction=1, exploration_final_eps=0.02, exploration_initial_eps=1.0,
                        train_freq=1, batch_size=32, double_q=True, learning_starts=1000,
                        target_network_update_freq=500, prioritized_replay=False, prioritized_replay_alpha=0.6,
                        prioritized_replay_beta0=0.4, prioritized_replay_beta_iters=None,
                        prioritized_replay_eps=1e-06, param_noise=False, n_cpu_tf_sess=None, verbose=0,
                        tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
                        full_tensorboard_log=False, seed=None)
        elif algo.upper() == "PPO2":
            from stable_baselines.common.policies import MlpPolicy
            from stable_baselines.ppo2 import PPO2
            model = PPO2(MlpPolicy, env, verbose=1)
        elif algo.upper() == "A2C":
            from stable_baselines.common.policies import MlpPolicy
            from stable_baselines.a2c import A2C
            env = DummyVecEnv([lambda: env])
            model = A2C(MlpPolicy, env, gamma=0.90, learning_rate=0.0005,
                        n_cpu_tf_sess=None, verbose=0,
                        tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
                        full_tensorboard_log=False, seed=None)
        elif algo.upper() == "ACER":
            from stable_baselines.common.policies import MlpPolicy
            from stable_baselines.acer import ACER
            env = DummyVecEnv([lambda: env])
            model = ACER(MlpPolicy, env, replay_ratio=0, verbose=0)
            #model = ACER(MlpPolicy, env,  verbose=1)
        elif algo.upper() == "ACKTR":
            from stable_baselines.common.policies import MlpPolicy
            from stable_baselines.acktr import ACKTR
            env = DummyVecEnv([lambda: env])
            model = ACKTR(MlpPolicy, env, verbose=0)
        elif algo.upper() == "GAIL":
            assert False , "GAIL is not implemented"
            from stable_baselines.common.policies import MlpPolicy
            import stable_baselines.gail
        elif algo.upper() == "HER":
            assert False , "HER is not implemented"
            import stable_baselines.her
            pass
        elif algo.upper() == "PPO1":
            from stable_baselines.common.policies import MlpPolicy
            from stable_baselines.ppo1 import PPO1
            env = DummyVecEnv([lambda: env])
            model = PPO1(MlpPolicy, env, verbose=0)
        elif algo.upper() == "TRPO":
            from stable_baselines.common.policies import MlpPolicy
            from stable_baselines.trpo_mpi import TRPO
            env = DummyVecEnv([lambda: env])
            model = TRPO(MlpPolicy, env, verbose=1)

        return model

    def load_model(algo, env, path):
        if algo.upper() == "DQN":
            from stable_baselines.deepq import DQN
            model = DQN.load(path)
            model.set_env(env)
        elif algo.upper() == "PPO2":
            from stable_baselines.ppo2 import PPO2
            model = PPO2.load(path)
            env = DummyVecEnv([lambda: env])
            model.set_env(env)
        elif algo.upper() == "A2C":
            from stable_baselines.a2c import A2C
            model = A2C.load(path)
            env = DummyVecEnv([lambda: env])
            model.set_env(env)
        elif algo.upper() == "ACER":
            from stable_baselines.acer import ACER
            model = ACER.load(path)
            env = DummyVecEnv([lambda: env])
            model.set_env(env)
        elif algo.upper() == "ACKTR":
            from stable_baselines.acktr import ACKTR
            model = ACKTR.load(path)
            env = DummyVecEnv([lambda: env])
            model.set_env(env)
        elif algo.upper() == "GAIL":
            import stable_baselines.gail
        elif algo.upper() == "HER":
            import stable_baselines.her
            pass
        elif algo.upper() == "PPO1":
            from stable_baselines.ppo1 import PPO1
            model = PPO1.load(path)
            env = DummyVecEnv([lambda: env])
            model.set_env(env)
        elif algo.upper() == "TRPO":
            from stable_baselines.trpo_mpi import TRPO
            model = TRPO.load(path)
            env = DummyVecEnv([lambda: env])
            model.set_env(env)
        else:
            return None
        return model

    def test_agent(env: CIPairWiseEnv, model_path: str, algo, mode):
        agent_actions = []
        print("Evaluation of an agent from " + model_path)
        model = TPAgentUtil.load_model(path=model_path, algo=algo, env=env)
        if model:
            if mode.upper() == "PAIRWISE" and algo.upper() != "DQN" :
                env = model.get_env()
                obs = env.reset()
                done = False
                while True:
                    action, _states = model.predict(obs, deterministic=True)
                    #print(action)
                    obs, rewards, done, info = env.step(action)
                    if done:
                        break
                return env.get_attr("sorted_test_cases_vector")[0]
            elif mode.upper() == "PAIRWISE" and algo.upper() == "DQN":
                env = model.get_env()
                obs = env.reset()
                done = False
                while True:
                    action, _states = model.predict(obs, deterministic=True)
                    obs, rewards, done, info = env.step(action)
                    if done:
                        break
                return env.sorted_test_cases_vector
            elif mode.upper() == "LISTWISE":
                pass
            elif mode.upper() == "POINTWISE":
                pass
