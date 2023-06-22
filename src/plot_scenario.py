import argparse

import numpy as np
import matplotlib.pyplot as plt

from data_generation.data_generator import DataGenerator
from util.load_config_files import load_yaml_into_dotdict, dotdict



def parse_input_args():
    # Load CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-tp",
        "--task_params",
        help="filepath to configuration yaml file defining the task",
        required=True,
    )

    args = parser.parse_args()
    params = load_yaml_into_dotdict(args.task_params)
    params.training = dotdict()

    # Now you can set attributes on params.training
    params.training.device = "cpu"
    params.training.batch_size = 1
    print(f"Task configuration file: {args.task_params}")

    return params


def polar_to_cartesian(r, theta):
    return r * np.cos(theta), r * np.sin(theta) 

def plot_estimates_and_measurements(true_measurements, false_measurements, trajectories):

    fig, ax = plt.subplots(figsize=(48, 48))
    
    marker_size = 40
    false_measurements = false_measurements[0]
    r = false_measurements[:, 0]
    theta = false_measurements[:, 1]
    x, y = polar_to_cartesian(r, theta)

    alpha = 0.8 #- (false_measurements[:, 2] / false_measurements[:, 2].max())
    ax.scatter(x, y, facecolors=(0.5, 0.5, 0.5, 0.5), edgecolors='k', s=marker_size)
    
    true_measurements = true_measurements[0]
    r = true_measurements[:, 0]
    theta = true_measurements[:, 1]
    x, y = polar_to_cartesian(r, theta)

    alpha = 0.8 #- (false_measurements[:, 2] / false_measurements[:, 2].max())
    ax.scatter(x, y, facecolors=(1.0, 0.0, 0.0, 0.5), edgecolors='r', s=marker_size)
    
    # plt.show()
    
    
    # # Plot trajectories
    for id, traj in trajectories[0].items():
        traj_x, traj_y, traj_vx, traj_vy, traj_time = zip(*traj)
        ax.plot(
            traj_x,
            traj_y,
            color="g",
            alpha=1.0,
            linewidth=2.0
            #label=f"Ground truth for target {obj_id}",
        )
        
    for theta in np.linspace(0, 2*np.pi, 8, endpoint=False):
        x = 250 * np.cos(theta)
        y = 250 * np.sin(theta)
        ax.plot([0, x], [0, y], color='gray', linestyle='dashed', linewidth=1)

    step = 31.25
    for i in range(1, 6):
        circle = plt.Circle((0, 0), i*step, color='gray', fill=False, linestyle='dashed', linewidth=1)
        ax.add_artist(circle)


    font_size = 40
    ax.legend()
    ax.set_xlabel("East", fontsize=font_size)
    ax.set_ylabel("North", fontsize=font_size)
    ax.set_xlim([-125, 125])
    ax.set_ylim([-125, 125])
    ax.set_title("Automatically generated scenario", fontsize=font_size)
    ax.set_aspect("equal")
    plt.xticks([])
    plt.yticks([])

    plt.show()
    
    
def main():

    params = parse_input_args()

    data_generator = DataGenerator(params)


    # Generate data
    (
        training_nested_tensor,
        labels,
        unique_measurement_ids,
        unique_label_ids,
        trajectories,
        true_measurements,
        false_measurements,
        object_events,
    ) = data_generator.get_single_training_scenario()
    
    plot_estimates_and_measurements(true_measurements, false_measurements, trajectories)
    


if __name__ == "__main__":
    main()