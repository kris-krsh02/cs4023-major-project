#!/usr/bin/env python3
import matplotlib
import matplotlib.pyplot as plt
from dqn import DQNAgent
import gymnasium as gym
import rospy
import touring_env

def main():
    env = gym.make("Turtlebot3Touring")
    rospy.loginfo("Gym environment created")
    agent = DQNAgent(env)
    agent.inference("\models\model_0.9_200.pth")

    # Close the environment
    env.close()

if __name__ == "__main__":
    main()
