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
    
    # Iniialize master node
    rospy.init_node("inference_node", anonymous=True)

    agent = DQNAgent(env)
    agent.inference("/home/kars0009/catkin_ws/src/cs4023-major-project/cs4023-major-project/src/models/models_300_win30/model_0.9_200.pth")

    # Close the environment
    env.close()

if __name__ == "__main__":
    main()
