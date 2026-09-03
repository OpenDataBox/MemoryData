# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ray
import gym
import numpy as np

# -----------------------------------------------------------------------------
# Ray remote worker actor -----------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopWorker:
    """Ray remote actor that replaces the worker function.
    Each actor hosts a *WebAgentTextEnv* instance.
    """
    
    def __init__(self, seed, env_kwargs, assigned_goal_idx=None):
        """
        Args:
            seed: Random seed for the worker
            env_kwargs: Environment configuration kwargs
            assigned_goal_idx: If provided (for validation), this worker will only use this specific goal.
                              If None (for training), worker can access all goals from goal_idxs pool.
        """
        # Lazy import avoids CUDA initialisation issues
        import sys
        import os
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'webshop'))
        sys.path.append(project_root)
        from web_agent_site.envs import WebAgentTextEnv  # noqa: WPS433 (runtime import)
        
        env_kwargs['seed'] = seed
        self.env = gym.make('WebAgentTextEnv-v0', **env_kwargs)
        self.assigned_goal_idx = assigned_goal_idx
    
    def step(self, action):
        """Execute a step in the environment"""
        obs, reward, done, info = self.env.step(action)
        info = dict(info or {})  # make a *copy* so we can mutate safely
        info['available_actions'] = self.env.get_available_actions()
        info['task_score'] = reward

        # Redefine reward. We only use rule-based reward - win for 10, lose for 0.
        if done and reward == 1.0:
            info['won'] = True
            reward = 10.0
        else:
            info['won'] = False
            reward = 0

        return obs, reward, done, info
    
    def reset(self, idx=None):
        """Reset the environment with given session index
        
        Args:
            idx: Goal index to use. If None and assigned_goal_idx is set, uses assigned_goal_idx.
        """
        # Use assigned goal if available and idx not explicitly provided
        if idx is None and self.assigned_goal_idx is not None:
            idx = self.assigned_goal_idx
            
        obs, info = self.env.reset(session=idx)
        info = dict(info or {})
        info['available_actions'] = self.env.get_available_actions()
        info['won'] = False
        return obs, info
    
    def render(self, mode_for_render):
        """Render the environment"""
        rendered = self.env.render(mode=mode_for_render)
        return rendered
    
    def get_available_actions(self):
        """Get available actions"""
        return self.env.get_available_actions()
    
    def get_goals(self):
        """Get environment goals"""
        return self.env.server.goals
    
    def close(self):
        """Close the environment"""
        self.env.close()


# -----------------------------------------------------------------------------
# Vectorised Ray environment --------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopMultiProcessEnv(gym.Env):
    """A vectorised, Ray-based wrapper around *WebAgentTextEnv*.

    ``info`` dictionaries returned by :py:meth:`step` **and** :py:meth:`reset`
    automatically contain the key ``'available_actions'`` so downstream RL code
    can obtain the *legal* action set without extra IPC overhead.
    """
    def __init__(
        self,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: dict,
        is_train: bool = True,
        env_kwargs: dict = None,
        batch_start_idx: int = 0,
    ) -> None:
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train
        if not is_train: assert group_n == 1

        self._rng = np.random.RandomState(seed)

        self._env_kwargs = env_kwargs if env_kwargs is not None else {'observation_mode': 'text', 'num_products': None}

        # -------------------------- Load one env to get goal count --------------------------
        import sys
        import os
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'webshop'))
        sys.path.append(project_root)
        from web_agent_site.envs import WebAgentTextEnv  # noqa: WPS433 (runtime import)
        
        # Create a temporary env just to get the number of goals
        temp_env_kwargs = self._env_kwargs.copy()
        temp_env_kwargs['seed'] = seed
        temp_env = gym.make('WebAgentTextEnv-v0', **temp_env_kwargs)
        num_total_goals = len(temp_env.server.goals)
        temp_env.close()
        
        print(f"Total goals available: {num_total_goals}")

        # -------------------------- Determine goal indices based on split -------
        split = env_kwargs.get('split', 'train')
        if split == 'test':
            # Test: use first 500 goals
            self.goal_idxs = list(range(500))
            num_available_goals = len(self.goal_idxs)
        elif split == 'val':
            # Validation: use next 500 goals (500-999)
            self.goal_idxs = list(range(500, 1000))
            num_available_goals = len(self.goal_idxs)
        else:
            # Training: use goals from 1000 onwards
            self.goal_idxs = list(range(1000, num_total_goals))
            num_available_goals = len(self.goal_idxs)
            
        print(f"Goal indices: {self.goal_idxs[:10]}...{self.goal_idxs[-10:]} (total: {num_available_goals})")

        # -------------------------- Assign goals to workers --------------------------
        if is_train:
            # Training: multiple workers can sample from all goals
            assigned_goals = [None] * self.num_processes  # None means access all goals
        else:
            # Validation: one worker per goal, deterministic assignment
            assert group_n == 1, "group_n should be 1 for validation/test"
            
            # For batched evaluation, assign goals starting from batch_start_idx
            # Each batch processes a consecutive slice of eval goals
            batch_end_idx = batch_start_idx + self.num_processes
            assert batch_end_idx <= num_available_goals, \
                f"ERROR: batch_end_idx ({batch_end_idx}) exceeds available eval goals ({num_available_goals})."
            
            # Assign consecutive goals from batch_start_idx
            assigned_goals = [self.goal_idxs[batch_start_idx + i] for i in range(self.num_processes)]
            print(f"Batch goals assigned: {assigned_goals[:5]}...{assigned_goals[-5:]} (start_idx={batch_start_idx}, count={self.num_processes})")

        # -------------------------- Ray actors setup --------------------------
        # Each worker will load its own environment (unavoidable due to Java objects in SimServer)
        env_worker_class = ray.remote(**resources_per_worker)(WebshopWorker)
        self._workers = []
        for i in range(self.num_processes):
            worker = env_worker_class.remote(
                seed + (i // self.group_n), 
                self._env_kwargs,
                assigned_goal_idx=assigned_goals[i]  # None for training, specific goal for validation
            )
            self._workers.append(worker)

    # ------------------------------------------------------------------
    # Base API ----------------------------------------------------------
    # ------------------------------------------------------------------

    def step(self, actions: list[str]):
        if len(actions) != self.num_processes:
            raise ValueError(
                f'Expected {self.num_processes} actions, got {len(actions)}',
            )

        # Send step commands to all workers
        futures = []
        for worker, action in zip(self._workers, actions):
            future = worker.step.remote(action)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self):
        """Reset all workers.
        
        For training: randomly sample goals for each worker.
        For validation: each worker uses its pre-assigned goal.
        """
        if self.is_train:
            # Training: randomly sample goals
            idx = self._rng.choice(self.goal_idxs, size=self.env_num, replace=False)
            idx = np.repeat(idx, self.group_n).tolist()
        else:
            # Validation: use pre-assigned goals (pass None to let worker use assigned_goal_idx)
            idx = [None] * self.num_processes

        # Send reset commands to all workers
        futures = []
        for worker, i in zip(self._workers, idx):
            future = worker.reset.remote(i)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)

        return obs_list, info_list

    # ------------------------------------------------------------------
    # Convenience helpers ----------------------------------------------
    # ------------------------------------------------------------------

    def render(self, mode: str = 'text', env_idx: int = None):
        if env_idx is not None:
            future = self._workers[env_idx].render.remote(mode)
            return ray.get(future)

        futures = []
        for worker in self._workers:
            future = worker.render.remote(mode)
            futures.append(future)
        
        return ray.get(futures)

    # ------------------------------------------------------------------
    # Clean‑up ----------------------------------------------------------
    # ------------------------------------------------------------------

    def close(self):
        if getattr(self, '_closed', False):
            return

        # Check if _workers attribute exists (in case __init__ failed)
        if not hasattr(self, '_workers'):
            return

        try:
            # Close all workers and kill Ray actors
            close_futures = []
            for worker in self._workers:
                future = worker.close.remote()
                close_futures.append(future)
            
            # Wait for all workers to close
            ray.get(close_futures)
            
            # Kill all Ray actors
            for worker in self._workers:
                ray.kill(worker)
                
            self._closed = True
        except (ImportError, RuntimeError):
            # Ignore errors during Python/Ray shutdown
            pass

    def __del__(self):  # noqa: D401
        self.close()


# -----------------------------------------------------------------------------
# Factory helper --------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_webshop_envs(
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: dict,
    is_train: bool = True,
    env_kwargs: dict = None,
    batch_start_idx: int = 0,
):
    """Mirror *build_sokoban_envs* so higher‑level code can swap seamlessly."""
    return WebshopMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_kwargs=env_kwargs,
        batch_start_idx=batch_start_idx,
    )