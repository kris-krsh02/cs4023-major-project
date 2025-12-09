#!/usr/bin/env python3

from dqn import DQNAgent
import gymnasium as gym
import rospy
import touring_env


def main():
    rospy.init_node("master_node", anonymous=True)

    # Create environment
    env = gym.make("Turtlebot3Touring")
    rospy.loginfo("Gym environment created")

    agent = DQNAgent(env)
    agent.train(300)
    env.close()

if __name__ == "__main__":
    main()
