import numpy as np
import gym
from gym import spaces
import random
random.seed(1)
np.random.seed(1)

class PF(gym.Env):

	def __init__(self):
		self.env = gym.make('FetchPickAndPlace-v1')
		self.action_space = self.env.action_space
		#self.observation_space = self.env.observation_space["observation"]
		self.observation_space = spaces.Box(low=np.array([-np.inf]*28) , high=np.array([np.inf]*28), dtype=np.float32)
		self.episodes = 0
		self.mode_freq = [0.0, 0.0, 0.0, 0.0]
		self.total_steps = 0
		self._max_episode_steps = self.env._max_episode_steps


	def render(self):
		self.env.render()
		
		
	def reset(self):
		self.ep_rew = 0.0
		obs = self.env.reset()
		self.env_obs = obs
		obs = np.concatenate((obs["observation"], obs["desired_goal"]))
		self.timeStep = 0
		self.mode = 0
		return obs


	def step(self, actions):
		#if(self.episodes%5 == 0):
		#self.render()
		obs, rew, done, info = self.env.step(actions)
		rew += 1
		'''for i in range(3):
			obs, rew, done, info = self.env.step(actions)
			self.render()'''
		self.env_obs = obs
		obs = np.concatenate((obs["observation"], obs["desired_goal"]))
		self.ep_rew += rew
		if(done):
			self.episodes += 1
			'''f = open("rewards.txt", "a+")
			f.write(str(np.mean(self.ep_rew)) + "\n")
			f.close()'''
		self.timeStep += 1
		self.total_steps += 1
		return [obs, rew, done, info]
		

	def goToGoal_0(self):
	
		lastObs = self.env_obs
		goal = lastObs['desired_goal']

		#objectPosition
		objectPos = lastObs['observation'][3:6]
		gripperPos = lastObs['observation'][:3]
		gripperState = lastObs['observation'][9:11]
		object_rel_pos = lastObs['observation'][6:9]

		object_oriented_goal = object_rel_pos.copy()
		object_oriented_goal[2] += 0.03
		print(lastObs['observation'], self.mode)
		self.mode_freq[self.mode] += 1
		if(self.total_steps % 1024*4 == 0):
			print(self.mode_freq)
			print(self.mode)
			self.mode_freq = [0.0, 0.0, 0.0, 0.0]
		#print(lastObs["observation"], self.mode, self.timeStep)
		if(np.linalg.norm(object_oriented_goal) >= 0.035 and self.timeStep <= self.env._max_episode_steps and self.mode == 0):
			action = [0, 0, 0, 0]
			for i in range(len(object_oriented_goal)):
				action[i] = object_oriented_goal[i]*6

			action[len(action)-1] = 1.0#0.05


		elif(np.linalg.norm(object_rel_pos) >= 0.025 and self.timeStep <= self.env._max_episode_steps and self.mode <= 1):
			action = [0, 0, 0, 0]
			for i in range(len(object_rel_pos)):
				action[i] = object_rel_pos[i]*6

			action[len(action)-1] = -1.0#-0.005
			self.mode = 1


		elif(np.linalg.norm(goal - objectPos) >= 0.015 and self.timeStep <= self.env._max_episode_steps and self.mode <= 2):
			action = [0, 0, 0, 0]
			for i in range(len(goal - objectPos)):
				action[i] = (goal - objectPos)[i]*6

			action[len(action)-1] = -1.0#-0.005
			self.mode = 2

		
		else:
			action = [0, 0, 0, 0]

			action[len(action)-1] = -1.0#-0.005
			self.mode = 3


		return action
		
		
	def goToGoal(self, o):
	
		lastObs = self.env_obs
		goal = lastObs['desired_goal']

		#objectPosition
		objectPos = lastObs['observation'][3:6]
		gripperPos = lastObs['observation'][:3]
		gripperState = lastObs['observation'][9:11]
		object_rel_pos = lastObs['observation'][6:9]
		

		object_oriented_goal = object_rel_pos.copy()
		object_oriented_goal[2] += 0.03

		self.mode_freq[self.mode] += 1
		if(self.total_steps % 1024 == 0):
			print(self.mode_freq)
			self.mode_freq = [0.0, 0.0, 0.0, 0.0]

		action = [0, 0, 0, 0]

		obj_dist = self.get_dist(gripperPos[0], gripperPos[1], gripperPos[2], objectPos[0], objectPos[1], objectPos[2])
		#print(obj_dist, self.mode)
		#if(self.mode == 2):
		#print(lastObs['observation'], self.mode, gripperState)

		if(obj_dist > 0.03):
			action = [0, 0, 0, 0]
			for i in range(len(object_oriented_goal)):
				action[i] = object_oriented_goal[i]*6
			action[len(action)-1] = 0.3
			self.mode = 0
		else:
			if(gripperState[0] >= 0.03):
				action = [0, 0, 0, 0]
				for i in range(len(object_rel_pos)):
					action[i] = object_rel_pos[i]*6
				self.mode = 1
				action[len(action)-1] = -0.3
			else:#if(self.mode == 1):
				action = [0, 0, 0, 0]
				for i in range(len(goal - objectPos)):
					action[i] = (goal - objectPos)[i]*6
				self.mode = 2
				action[len(action)-1] = -0.3

		return np.array(action)
		
		
	def get_dist(self, s_x, s_y, s_z, t_x, t_y, t_z):
		return np.sqrt(pow(s_x-t_x, 2) + pow(s_y-t_y, 2) + pow(s_z-t_z, 2))
