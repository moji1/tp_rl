import os

from stable_baselines.common.callbacks import BaseCallback
import numpy as np
from stable_baselines.results_plotter import load_results, ts2xy


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, svae_path: str, check_freq: int, log_dir: str, verbose=1):
        super(CustomCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = svae_path
        self.best_mean_reward = -np.inf
        self.plateau_cnt = 0

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        #print(self.n_calls)
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print(
                        "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward,
                                                                                                 mean_reward))

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.plateau_cnt = 0
                    self.best_mean_reward = mean_reward
                    self.model.save(self.save_path)
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                    else:
                        self.plateau_cnt = self.plateau_cnt + 1
                        episodes = int(self.n_calls / self.check_freq)
                    if self.plateau_cnt>=50 or self.n_calls >= 1000000:
                        print("Training is stopped due to the  plateau at step " + str(self.n_calls), flush=True)
                        return False

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

    def zero_slope(self, x, y, chunksize=10, max_slope=.01):
        """return the 'first' data point with zero slope

        data --> numpy ndarray - 2d [[x0,y0],[x1,y1],...]
        chunksize --> odd int
        returns numpy ndarray
        """
        midindex = chunksize / 2
        cnt = 0
        for index in range(len(x) - chunksize, len(x)-1):
            # chunk = data_points[index: index + chunksize]
            # subtract the endpoints of the chunk
            # if not sufficient, maybe use a linear fit
            dx = abs(x[index + 1] - x[index])
            dy = abs(y[index + 1] - y[index])
            print(dy, dx, dy / dx)
            if 0 <= dy / dx < max_slope:
                cnt = cnt + 1
        if cnt == chunksize-1:
            return True
        else:
            print(cnt)
            return False
