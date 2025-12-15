#!/usr/bin/env python3

from dqn import DQNAgent
import gymnasium as gym
import rospy
import touring_env


def main():
    # Training parameters
    gammas = [0.9, 0.5, 0]
    epsilons = [50, 200]
    num_episodes = 20

    # Iniialize master node
    rospy.init_node("master_node", anonymous=True)

    # Create environment
    env = gym.make("Turtlebot3Touring")
    rospy.loginfo("Gym environment created")

    # Run trainings over parameters
    for gamma in gammas:
        for eps in epsilons:
            agent = DQNAgent(env, gamma, eps)
            agent.train(num_episodes)
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    main()
