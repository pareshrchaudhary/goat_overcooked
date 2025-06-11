import os

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import pygame

from .core import Agent
from .multi_discrete import MultiDiscrete

# update bounds to center around agent
cam_range = 2

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, scenario, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, post_step_callback=None,
                 shared_viewer=True, discrete_action=True):

        self.scenario = scenario
        
        self.world = world
        self.world_length = self.world.world_length
        self.current_step = 0
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback

        self.post_step_callback = post_step_callback

        # environment parameters
        # self.discrete_action_space = True
        self.discrete_action_space = discrete_action

        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(
            world, 'discrete_action') else False
        # in this env, force_discrete_action == False��because world do not have discrete_action

        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(
            world, 'collaborative') else False
        #self.shared_reward = False
        self.time = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        share_obs_dim = 0
        for agent in self.agents:
            total_action_space = []
            
            # physical action space
            if agent.movable:
                if self.discrete_action_space:
                    u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
                else:
                    u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,), dtype=np.float32)  # [-1,1]
                total_action_space.append(u_action_space)

            # communication action space
            if not agent.silent:
                if self.discrete_action_space:
                    c_action_space = spaces.Discrete(world.dim_c)
                else:
                    c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)  # [0,1]
                total_action_space.append(c_action_space)
            
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete(
                        [[0, act_space.n-1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            
            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            share_obs_dim += obs_dim
            self.observation_space.append(spaces.Box(
                low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))  # [-inf,inf]
            agent.action.c = np.zeros(self.world.dim_c)
        
        self.share_observation_space = [spaces.Box(
            low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32) for _ in range(self.n)]
        
        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def seed(self, seed=None):
        if hasattr(self.scenario, "_seed"):
            self.scenario._seed(seed)
        else:
            np.random.seed(seed)

    # step  this is  env.step()
    def step(self, action_n):
        self.current_step += 1
        obs_n = []
        reward_n = []
        done_n = []
        info_n = []
        self.agents = self.world.policy_agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
        # advance world state
        self.world.step()  # core.step()
        # record observation for each agent
        for i, agent in enumerate(self.agents):
            obs_n.append(self._get_obs(agent))
            reward_n.append([self._get_reward(agent)])
            done_n.append(self._get_done(agent))
            info = {'individual_reward': self._get_reward(agent)}
            env_info = self._get_info(agent)
            if 'fail' in env_info.keys():
                info['fail'] = env_info['fail']
            info_n.append(info)

        # all agents get total reward in cooperative case, if shared reward, all agents have the same reward, and reward is sum
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [[reward]] * self.n

        if self.post_step_callback is not None:
            self.post_step_callback(self.world)

        return obs_n, reward_n, done_n, info_n

    def reset(self):
        self.current_step = 0
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents

        for agent in self.agents:
            obs_n.append(self._get_obs(agent))

        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            if self.current_step >= self.world_length:
                return True
            else:
                return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index+s)])
                index += s
            action = act
        else:
            action = [action]
        if agent.movable:
            # physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete action
                if action[0] == 1:
                    agent.action.u[0] = -1.0
                if action[0] == 2:
                    agent.action.u[0] = +1.0
                if action[0] == 3:
                    agent.action.u[1] = -1.0
                if action[0] == 4:
                    agent.action.u[1] = +1.0
                d = self.world.dim_p
            else:
                if self.discrete_action_space:
                    agent.action.u[0] += action[0][1] - action[0][2]
                    agent.action.u[1] += action[0][3] - action[0][4]
                    d = 5
                else:
                    if self.force_discrete_action:
                        p = np.argmax(action[0][0:self.world.dim_p])
                        action[0][:] = 0.0
                        action[0][p] = 1.0
                    agent.action.u = action[0][0:self.world.dim_p]
                    d = self.world.dim_p

            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity

            if (not agent.silent) and (not isinstance(action_space, MultiDiscrete)):
                action[0] = action[0][d:]
            else:
                action = action[1:]

        if not agent.silent:
            # communication action
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]

            action = action[1:]

        # make sure we used all elements of action
        assert len(action) == 0

    # for renderring
    def _reset_render(self):
        pygame.init()
        self.game_font = pygame.freetype.Font(
            os.path.join(os.path.dirname(__file__), "secrcode.ttf"), 24
        )
        self.width, self.height = 700, 700
        self.screen = pygame.Surface([self.width, self.height])
    
    # use petting-zoo implementations: https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/mpe/_mpe_utils/simple_env.py
    def enable_render(self):
        if getattr(self, "renderOn", False) is False:
            self.screen = pygame.display.set_mode(self.screen.get_size())
            self.clock = pygame.time.Clock()
            self.renderOn = True

    def render(self, mode="human"):
        self.enable_render()

        self.draw()
        if mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        observation = np.array(pygame.surfarray.pixels3d(self.screen))
        return [np.transpose(observation, axes=(1, 0, 2))]

    def draw(self):
        # clear screen
        self.screen.fill((255, 255, 255))

        # update bounds to center around agent
        all_poses = [entity.state.p_pos for entity in self.world.entities]
        cam_range = np.max(np.abs(np.array(all_poses)))

        # update geometry and text positions
        text_line = 0
        for e, entity in enumerate(self.world.entities):
            # geometry
            x, y = entity.state.p_pos
            y *= (
                -1
            )  # this makes the display mimic the old pyglet setup (ie. flips image)
            x = (
                (x / cam_range) * self.width // 2 * 0.9
            )  # the .9 is just to keep entities from appearing "too" out-of-bounds
            y = (y / cam_range) * self.height // 2 * 0.9
            x += self.width // 2
            y += self.height // 2
            pygame.draw.circle(
                self.screen, entity.color * 200, (x, y), entity.size * 350
            )  # 350 is an arbitrary scale factor to get pygame to render similar sizes as pyglet
            pygame.draw.circle(
                self.screen, (0, 0, 0), (x, y), entity.size * 350, 1
            )  # borders
            assert (
                0 < x < self.width and 0 < y < self.height
            ), f"Coordinates {(x, y)} are out of bounds."

            # text
            if isinstance(entity, Agent):
                if entity.silent:
                    continue
                if np.all(entity.state.c == 0):
                    word = "_"
                elif not self.discrete_action_space:
                    word = (
                        "[" + ",".join([f"{comm:.2f}" for comm in entity.state.c]) + "]"
                    )
                else:
                    word = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[np.argmax(entity.state.c)]

                message = entity.name + " sends " + word + "   "
                message_x_pos = self.width * 0.05
                message_y_pos = self.height * 0.95 - (self.height * 0.05 * text_line)
                self.game_font.render_to(
                    self.screen, (message_x_pos, message_y_pos), message, (0, 0, 0)
                )
                text_line += 1

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
