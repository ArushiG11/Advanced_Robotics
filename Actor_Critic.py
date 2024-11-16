import mujoco as mj
import numpy as np
import random
import matplotlib.pyplot as plt
import time

# Load MuJoCo environment from XML file
model = mj.MjModel.from_xml_path("nav1.xml")
sim = mj.MjData(model)


# Parameters for the environment and planning
goal_region = (0.9, 0.0)  # Goal position as per the XML setup
Tmax = 30  # Termination time threshold in seconds
delta_t = 0.1  # Time step for each control action
goal_threshold = 0.1  # Distance threshold to consider goal reached
max_control = 0.5  # Max control input

# Helper functions for Kinodynamic RRT

def sample_configuration():
    # Sample a random configuration within defined bounds
    return (random.uniform(-1, 1), random.uniform(-1, 1))

def nearest_neighbor(tree, x_rand):
    # Find the nearest point in the tree to x_rand
    return min(tree, key=lambda x: np.linalg.norm(np.array(x) - np.array(x_rand)))

def sample_control():
    # Sample a random control input within max bounds
    return (random.uniform(-max_control, max_control), random.uniform(-max_control, max_control))

def simulate(x_near, u):
    # Set the control, simulate the robot, and return the resulting position
    sim.data.qpos[:2] = x_near
    sim.data.ctrl[:] = u
    sim.step()
    return (sim.data.qpos[0], sim.data.qpos[1])

def is_collision_free(x):
    # Check if x is in collision-free space (simplified here)
    return sim.data.ncon == 0  # No contacts, adjust logic based on actual collision checks

def goal_check(x, goal_region):
    # Check if x is within the goal threshold distance
    return np.linalg.norm(np.array(x) - np.array(goal_region)) < goal_threshold

# Kinodynamic RRT Algorithm
def kinodynamic_rrt(start, goal_region, Tmax):
    tree = [start]
    parent_map = {start: None}  # Keeps track of tree structure
    start_time = time.time()
    
    while time.time() - start_time < Tmax:
        x_rand = sample_configuration()
        x_near = nearest_neighbor(tree, x_rand)
        u = sample_control()
        x_new = simulate(x_near, u)

        if is_collision_free(x_new):
            tree.append(x_new)
            parent_map[x_new] = x_near  # Link new node to the tree

            if goal_check(x_new, goal_region):
                print("Goal reached!")
                return tree, parent_map

    print("Reached Tmax without finding a solution.")
    return tree, parent_map  # Return the tree even if goal was not reached

# Function to retrieve path from tree
def get_path(tree, parent_map, start, goal):
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = parent_map.get(current, start)
    path.append(start)
    path.reverse()
    return path

# Plotting Function
def plot_tree(tree, path=None):
    for node in tree:
        parent = parent_map.get(node)
        if parent is not None:
            plt.plot([parent[0], node[0]], [parent[1], node[1]], 'b-')
    if path:
        path_x, path_y = zip(*path)
        plt.plot(path_x, path_y, 'g-', linewidth=2)  # Highlight the path in green
    plt.scatter(*goal_region, color='red', label="Goal Region")
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Kinodynamic RRT Tree")
    plt.show()

# Run multiple trials with different seeds and Tmax values
def run_trials(start, goal_region, Tmax_values, num_trials=30):
    success_rates = {}
    for Tmax in Tmax_values:
        success_count = 0
        for i in range(num_trials):
            print(f"Trial {i+1} with Tmax = {Tmax}")
            random.seed(i)  # Set a different seed for each trial
            tree, parent_map = kinodynamic_rrt(start, goal_region, Tmax)
            
            if any(goal_check(node, goal_region) for node in tree):
                success_count += 1
                goal_node = next(node for node in tree if goal_check(node, goal_region))
                path = get_path(tree, parent_map, start, goal_node)
                plot_tree(tree, path=path)  # Visualize tree with successful path
            
        success_rate = (success_count / num_trials) * 100
        success_rates[Tmax] = success_rate
        print(f"Success rate for Tmax = {Tmax}: {success_rate}%")

    return success_rates

# Define start point and goal
start = (0.0, 0.0)  # Start position as per assignment

# Execute the trials with Tmax values
Tmax_values = [5, 10, 20, 30]
success_rates = run_trials(start, goal_region, Tmax_values)

# Print success rates summary
for Tmax, rate in success_rates.items():
    print(f"Tmax = {Tmax} seconds: Success Rate = {rate}%")
