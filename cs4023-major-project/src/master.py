#!/usr/bin/env python3
import matplotlib
import matplotlib.pyplot as plt
from dqn import DQNAgent
import gymnasium as gym
import rospy
import touring_env

def plot_param_avgs(metric, metric_name):
        # Iterate through all averages:
        print(metric)
        for key, values in metric.items():
            print(key, values)
            plt.plot(values, label = key)
            plt.xlabel = "Episodes"
            plt.ylabel = metric_name

        plt.legend(loc = 'upper left')
        plt.title(f"Comparison of Average {metric_name}")
        plt.savefig(f"plots/{metric_name}_comparison.png")
        plt.close()


def main():
    # Training parameters
    gammas = [0.9, 0.5, 0]
    epsilons = [50, 200]
    num_episodes = 300

    # Result variables
    all_avg_reward = {}
    all_avg_steps = {}

    # Iniialize master node
    rospy.init_node("master_node", anonymous=True)

    # Create environment
    env = gym.make("Turtlebot3Touring")
    rospy.loginfo("Gym environment created")

    # Run trainings over parameters
    for gamma in gammas:
        for eps in epsilons:
            print(f"Model \u03B3 = {gamma}, \u03B5_decay = {eps}")
            avg_key = f"\u03B3 = {gamma}, \u03B5_decay = {eps}"
            agent = DQNAgent(env, gamma, eps)
            all_avg_reward[avg_key], all_avg_steps[avg_key] =  agent.train(num_episodes)
    
    # Plot the moving averages
    plot_param_avgs(all_avg_reward, 'Average Reward')
    plot_param_avgs(all_avg_steps, 'Average Num of Steps')

    # Close the environment
    env.close()

if __name__ == "__main__":
    main()
