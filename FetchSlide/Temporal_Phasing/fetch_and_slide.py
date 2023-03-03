import numpy as np
import gym
from gym import spaces
import random
random.seed(1)
np.random.seed(1)
import math

class FS(gym.Env):

	def __init__(self):
		self.env = gym.make('FetchSlide-v1')
		self.action_space = self.env.action_space
		#self.observation_space = self.env.observation_space["observation"]
		self.observation_space = spaces.Box(low=np.array([-np.inf]*28) , high=np.array([np.inf]*28), dtype=np.float32)
		self.episodes = 0
		self.mode_freq = [0.0, 0.0, 0.0, 0.0]
		self.total_steps = 0
		print(self.action_space)
		print(self.observation_space)


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
		#actions=self.goToGoal()
		obs, rew, done, info = self.env.step(actions)
		rew += 1.0
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
		
		
	def goToGoal(self):
	
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
		if(self.total_steps % 1024*4 == 0):
			print(self.mode_freq)
			self.mode_freq = [0.0, 0.0, 0.0, 0.0]

		action = [0, 0, 0, 0]
		
		opp_goal_ang = (self.get_ang(objectPos[0], objectPos[1], goal[0], goal[1])+180.0)*np.pi/180.0
		x_disp = 0.2*np.cos(opp_goal_ang)
		y_disp = 0.2*np.sin(opp_goal_ang)
		proximity_obj_pos = objectPos.copy()
		proximity_obj_pos[0] += x_disp
		proximity_obj_pos[1] += y_disp
		proximity_obj_pos[2] = 0.52
		
		proximity_obj_dist = self.get_dist(gripperPos[0], gripperPos[1], gripperPos[2], proximity_obj_pos[0], proximity_obj_pos[1], proximity_obj_pos[2])
		obj_dist = self.get_dist(gripperPos[0], gripperPos[1], gripperPos[2], objectPos[0], objectPos[1], objectPos[2])
		goal_dist = self.get_dist(gripperPos[0], gripperPos[1], gripperPos[2], goal[0], goal[1], goal[2])

		if(proximity_obj_dist > 0.08):
			action = [0, 0, 0, 0]
			for i in range(len(object_oriented_goal)):
				if(i==2):
					action[i] = (proximity_obj_pos-gripperPos)[i]*20
				else:
					action[i] = (proximity_obj_pos-gripperPos)[i]*10 #(2.0-object_oriented_goal[i])*6
				
			#action[len(action)-1] = 0.0
		else:	
			self.mode = 1
		if(self.mode>=1):
			'''action = [0, 0, 0, 0]
			for i in range(len(object_rel_pos)):
				action[i] = object_rel_pos[i]*6
			self.mode = 1'''
			
			if(obj_dist>0.06 and self.mode==1):
				action = [0, 0, 0, 0]
				for i in range(len(goal - objectPos)):
					action[i] = (objectPos-gripperPos)[i]*10
				action[len(action)-1] = 0.0
				action[2] = -1.0
				
			else:
				action = [0, 0, 0, 0]
				for i in range(len(goal - objectPos)):
					action[i] = (goal-gripperPos)[i]*2
				action[len(action)-1] = 0.0
				self.mode = 2
		#if(obj_dist>goal_dist):
			#self.mode = 0

		return action
		
		
	def get_dist(self, s_x, s_y, s_z, t_x, t_y, t_z):
		return np.sqrt(pow(s_x-t_x, 2) + pow(s_y-t_y, 2) + pow(s_z-t_z, 2))
		
	def get_ang(self, source_x, source_y, neighbor_x, neighbor_y):
		if((source_x - neighbor_x) == 0.0 and (source_y - neighbor_y) != 0.0):
			if((neighbor_y - source_y) > 0.0):
				return 90.0
			else:
				return 270.0

		if((source_x - neighbor_x) == 0.0 and (source_y - neighbor_y) == 0.0):
			return 0.0

		angle = math.atan((neighbor_y - source_y)/(neighbor_x - source_x)) * 180.0/math.pi
		if(neighbor_x < source_x):
			angle = 180.0 + angle
		elif((neighbor_y < source_y) and (neighbor_x > source_x)):
			angle = 360.0 + angle
		return angle

