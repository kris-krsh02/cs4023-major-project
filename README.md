# Double Deep Q Networks for Robot Local Navigation

TODO:

- [] Create OpenAI Gym Environment
- [] Create DDQN Agent
- [] Create master node to process tasks

## Layers

- Global Planner - takes the sequence of landmarks. Passes one by one to local planner
- Local Planner (Navigator)
  - Takes in sensor input
  - Gives input to DDQN
  - Gets actions back
  - Executes actions
- DDQN
  - Implements 2 DQNs
  - Produces and sends predefined actions
  - Updates itself (training)

## Modes

- Training
- Testing

## Experiment Setup

Since we are learning a predefined route to be traversed, how should training be done?

- Option 1: Intermediary goals with final landmark as final goal
  - Will store a list of goals. Progressively reach each goal with a big reward, and change the goal, do not terminate. Final landmark has big big reward.
- Option 2: Let the robot randomly go to each landmark from a random starting position?
  - Prefer the first one; this doesn't sound that efficient.

### Sources for setting up the environment:

https://bitbucket.org/theconstructcore/drone_training/src/master/drone_training/src/myquadcopter_env.py

https://www.theconstruct.ai/using-openai-ros/
