#################################### INNER ##########################################

import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 1 - 0.003*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        delta_z = self.sim.pose[2] - self.target_pos[2]
        if self.sim.time > self.sim.runtime: # penalize the agent if it runs out of time
            reward -= 20
        reward += 0.003*self.sim.v[2] # reward positive z velocity to encourage the takeoff task
        
        if abs(delta_z) < 3: # if the agent is within a small distance of the target height, reward it generously and end episode
            reward +=100
            self.sim.done = True
        if self.sim.time < self.sim.runtime and self.sim.done: # penalize the agent if it crashes
            reward -= 10
            
#         delta_x = self.sim.pose[0] - self.target_pos[0]
#         delta_y = self.sim.pose[1] - self.target_pos[1]
#         euclidean = (np.sqrt((delta_x**2) + (delta_y**2) + (delta_z**2)))
#         reward = 10 - 0.7*euclidean

        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state