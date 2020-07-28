from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN
from PairWiseEnv import CIPairWiseEnv


class TPPairWiseDQNAgent:

    def train_agent(self, env: CIPairWiseEnv, steps: int, path_to_save_agent: None, base_model=None,
                    callback_class=None):
        env.reset()
        if not base_model:
            base_model = DQN(MlpPolicy, env, gamma=0.90, learning_rate=0.0005, buffer_size=10000,
                        exploration_fraction=1, exploration_final_eps=0.02, exploration_initial_eps=1.0,
                        train_freq=1, batch_size=32, double_q=True, learning_starts=1000,
                        target_network_update_freq=500, prioritized_replay=False, prioritized_replay_alpha=0.6,
                        prioritized_replay_beta0=0.4, prioritized_replay_beta_iters=None,
                        prioritized_replay_eps=1e-06, param_noise=False, n_cpu_tf_sess=None, verbose=0,
                        tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
                        full_tensorboard_log=False, seed=None)
            base_model.set_env(env)
        # check_env(env)
        base_model = base_model.learn(total_timesteps=steps, reset_num_timesteps=False, callback=callback_class)
        if path_to_save_agent:
            base_model.save(path_to_save_agent)
        return base_model

    #@timeout(500)
    def test_agent(self, env: CIPairWiseEnv, model_path: str, model):
        agent_actions = []
        print("Evaluation of an agent from " + model_path)
        if not model:
            model = DQN.load(model_path)
            print("Agent is loaded")
        if model:
            model.set_env(env)
            obs = env.reset()
            done = False
            while True:
                action, _states = model.predict(obs, deterministic=False)
                print(action)
                obs, rewards, done, info = env.step(action)
                if done:
                    break
        return env.test_cases_vector

