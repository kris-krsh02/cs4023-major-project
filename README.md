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
