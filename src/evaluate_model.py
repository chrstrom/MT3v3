import os
import csv
import time
import argparse


import torch
import numpy as np
from scipy.stats import t
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

from data_generation.data_generator import DataGenerator
from util.load_config_files import load_yaml_into_dotdict
from util.misc import NestedTensor

from modules.models.mt3v3.mt3v3 import MOTT as MT3V3

DT = 0.1


def calculate_gospa(targets, tracks, c, p, alpha=2, assignment_cost_function=None):
    """GOSPA metric for multitarget tracking filters.

    The algorithm is of course symmetric and can be used for any pair of
    comparable sets, but the labels 'targets' and tracks' are used to denote
    them as this is one of the most common applications.
    Returns the total GOSPA, the target-to-track assignment and the decomposed
    contribution of assignment errors, missed targets and false tracks.
    Parameters
    ----------
    targets : List[np.ndarray] iterable of elements
        Contains the elements of the first set.
    tracks : List[np.ndarray] iterable of elements
        Contains the elements of the second set.
    c : float
        The maximum allowable localization error, and also determines the cost
        of a cardinality mismatch.
    p : float
        Order parameter. A high value of p penalizes outliers more.
    alpha : float, optional
        Defines the cost of a missing target or false track along with c. The
        default value is 2, which is the most suited value for tracking
        algorithms.
    assignment_cost_function : function, optional
        This is the metric for comparing tracks and targets, referred to as
        d(x,y) in the reference. If no parameter is given, euclidian distance
        between x and y is used.
    Returns
    -------
    gospa : float
        Total gospa.
    assignment : dictionary
        Contains the assignments on the form {target_idx : track_idx}.
    gospa_localization : float
        Localization error contribution.
    gospa_missed : float
        Number of missed target contribution.
    gospa_false : float
        Number of false tracks contribution.
    References
    ----------
    - A. S. Rahmathullah, A. F. Garcia-Fernandez and L. Svensson, Generalized
      optimal sub-pattern assignment metric, 20th International Conference on
      Information Fusion, 2017.
    - L. Svensson, Generalized optimal sub-pattern assignment metric (GOSPA),
      presentation, 2017. Available online: https://youtu.be/M79GTTytvCM

    License for implementation of calculate_gospa:

    MIT License

    Copyright (c) 2019 Erik Wilthil

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """
    if alpha <= 0 or alpha > 2:
        raise ValueError("The value of alpha is outside the range (0, 2]")
    if c <= 0:
        raise ValueError("The cutoff distance c is outside the range (0, inf)")
    if p < 1:
        raise ValueError("The order p is outside the range [1, inf)")

    if assignment_cost_function is None:
        assignment_cost_function = lambda x, y: np.linalg.norm(x - y)

    num_targets = len(targets)
    num_tracks = len(tracks)
    miss_cost = c**p / alpha
    
    if num_targets == 0 and num_tracks == 0:
        return 0, dict(), 0, 0, 0

    if num_targets == 0:  # All the tracks are false tracks
        gospa_false = miss_cost * num_tracks
        return gospa_false ** (1 / p), dict(), 0, 0, gospa_false

    if num_tracks == 0:  # All the targets are missed
        gospa_missed = miss_cost * num_targets
        return gospa_missed ** (1 / p), dict(), 0, gospa_missed, 0

    # There are elements in both sets. Compute cost matrix
    cost_matrix = np.zeros((num_targets, num_tracks))
    for n_target in range(num_targets):
        for n_track in range(num_tracks):
            current_cost = (
                assignment_cost_function(targets[n_target], tracks[n_track]) ** p
            )
            cost_matrix[n_target, n_track] = np.min([current_cost, alpha * miss_cost])

    target_assignment, track_assignment = linear_sum_assignment(cost_matrix)
    gospa_localization = 0

    target_to_track_assigments = dict()
    for target_idx, track_idx in zip(target_assignment, track_assignment):
        if cost_matrix[target_idx, track_idx] < alpha * miss_cost:
            gospa_localization += cost_matrix[target_idx, track_idx]
            target_to_track_assigments[target_idx] = track_idx

    num_assignments = len(target_to_track_assigments)
    num_missed = num_targets - num_assignments
    num_false = num_tracks - num_assignments

    gospa_missed = miss_cost * num_missed
    gospa_false = miss_cost * num_false
    gospa = (gospa_localization + gospa_missed + gospa_false) ** (1 / p)

    return (
        gospa,
        target_to_track_assigments,
        gospa_localization,
        gospa_missed,
        gospa_false,
    )

     
    
def parse_input_args():
    # Load CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-tp",
        "--task_params",
        help="filepath to configuration yaml file defining the task",
        required=True,
    )
    parser.add_argument(
        "-mp",
        "--model_params",
        help="filepath to configuration yaml file defining the model",
        required=True,
    )
    parser.add_argument(
        "-pt",
        "--pretrained_model",
        help="filepath and name of pretrained model",
        required=True,
    )
    parser.add_argument(
        "--continue_training_from",
        help="filepath to folder of an experiment to continue training from",
    )
    args = parser.parse_args()

    # Load hyperparameters
    params = load_yaml_into_dotdict(args.task_params)
    params.update(load_yaml_into_dotdict(args.model_params))
    eval_params = load_yaml_into_dotdict(args.task_params)
    eval_params.update(load_yaml_into_dotdict(args.model_params))
    eval_params.recursive_update(load_yaml_into_dotdict("configs/params/eval.yaml"))
    eval_params.data_generation.seed += 1  # make sure we don't evaluate with same seed as final evaluation after training

    # Generate 32-bit random seed, or use user-specified one
    if params.general.pytorch_and_numpy_seed is None:
        random_data = os.urandom(4)
        params.general.pytorch_and_numpy_seed = int.from_bytes(
            random_data, byteorder="big"
        )

    if params.training.device == "auto":
        params.training.device = "cuda"
    if eval_params.training.device == "auto":
        eval_params.training.device = "cuda"

    # Seed pytorch and numpy for reproducibility
    np.random.seed(params.general.pytorch_and_numpy_seed)

    print(f"Task configuration file: {args.task_params}")
    print(f"Model configuration file: {args.model_params}")
    print(f"Using seed: {params.general.pytorch_and_numpy_seed}")
    print(
        f"Training device: {params.training.device}, Eval device: {eval_params.training.device}"
    )

    return args, params, eval_params



def get_max_time_from_trajectory(trajectory):
    max_time = 0.0
    for entry in trajectory[0].items():
        max_time_for_entry = np.round(entry[1][-1][-1], 3)
        if max_time_for_entry > max_time:
            max_time = max_time_for_entry

    return max_time


def plot_estimates_and_measurements(
    trajectories, mt3_position_estimates, true_measurements, false_measurements
):

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot true measurements
    true_measurements = true_measurements[0]
    r = true_measurements[:, 0]
    theta = true_measurements[:, 1]
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    alpha = 1 #- (true_measurements[:, 2] / true_measurements[:, 2].max())
    ax.scatter(x, y, alpha=alpha, color="k", marker="x", s=20)

    # Plot false measurements
    false_measurements = false_measurements[0]
    r = false_measurements[:, 0]
    theta = false_measurements[:, 1]
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    alpha = 1 #- (false_measurements[:, 2] / false_measurements[:, 2].max())
    ax.scatter(x, y, alpha=alpha, color="k", marker=".", s=20)
    
    for id, trajectory in trajectories[0].items():
        x = trajectory[:, 0]
        y = trajectory[:, 1]
        plt.plot(x, y, 'x-', markersize=3)
    

    for single_step_estimates in mt3_position_estimates:
        for id, estimate in single_step_estimates:
            ax.scatter(estimate[0], estimate[1], c="r")
    

    ax.legend()
    ax.set_xlabel("East")
    ax.set_ylabel("North")
    ax.set_xlim([-125, 125])
    ax.set_ylim([-125, 125])
    ax.set_title("Estimates and Measurements for all targets")
    ax.set_aspect("equal")
    ax.grid(True)

    plt.show()

def get_high_confidence_predictions(predictions, threshold):
    high_conf_predictions = []

    max_pred = 0
    for i, prediction in enumerate(predictions):
        if prediction[4] > threshold:
            high_conf_predictions.append((i, prediction))
            
        if prediction[4] > max_pred:
            max_pred = prediction[4]
            
    print(max_pred)
            
    return high_conf_predictions

def filter_by_timestamp(nested_tensor, timestamp):
    data = nested_tensor.tensors

    # Extract the third element of each row, which contains the timestamp
    timestamps = data[..., 2]

    # Create a mask for the rows with timestamp less than or equal to the given timestamp
    mask = timestamps <= timestamp
    # Use the mask to filter the rows
    filtered_data = data[mask]
    # Reshape the filtered data to have shape (num_rows, num_features)
    filtered_data = filtered_data.view(-1, data.shape[-1])

    total_elements = filtered_data.shape[0]
    tensor_mask = nested_tensor.mask[:, :total_elements]

    if total_elements < 20:
        num_padding_rows = 20 - total_elements
        padding = torch.empty(
            (num_padding_rows, filtered_data.shape[-1]),
            device=data.device,
            dtype=torch.float,
        ).uniform_(-30, 30)
        filtered_data = torch.cat([filtered_data, padding], dim=0)

        mask_padding = torch.tensor([False] * (num_padding_rows)).cuda()
        tensor_mask = torch.cat([tensor_mask, mask_padding.unsqueeze(0)], dim=1)

    filtered_nested_tensor = NestedTensor(filtered_data.unsqueeze(0), tensor_mask)
    return filtered_nested_tensor

def filter_by_timestamp_sliding_window(nested_tensor, timestamp, window_size=10):
    data = nested_tensor.tensors

    # Extract the third element of each row, which contains the timestamp
    timestamps = data[..., 2]

    # Create a mask for the rows with timestamp less than or equal to the given timestamp
    window_size_time = np.round(window_size * DT, 3)

    mask = (timestamp - window_size_time <= timestamps) & (timestamps <= timestamp)
    # Use the mask to filter the rows
    filtered_data = data[mask]
    # Reshape the filtered data to have shape (num_rows, num_features)
    filtered_data = filtered_data.view(-1, data.shape[-1])
    
    total_elements = filtered_data.shape[0]
    tensor_mask = nested_tensor.mask[:, :total_elements]

    if total_elements < 20:
        num_padding_rows = 20 - total_elements
        padding = torch.empty(
            (num_padding_rows, filtered_data.shape[-1]),
            device=data.device,
            dtype=torch.float,
        ).uniform_(-30, 30)
        padding[:, 2] = 0.0
        filtered_data = torch.cat([filtered_data, padding], dim=0)

        mask_padding = torch.tensor([False] * (num_padding_rows)).cuda()
        tensor_mask = torch.cat([tensor_mask, mask_padding.unsqueeze(0)], dim=1)
        
    min_time = torch.min(filtered_data[:, 2])
    filtered_data[:, 2] -= min_time

    filtered_nested_tensor = NestedTensor(filtered_data.unsqueeze(0), tensor_mask)
    return filtered_nested_tensor

def filter_data_by_time(data, time):
    filtered_data = []

    for d in data:
        for object_id, trajectory in d.items():
            for state in trajectory:
                if np.round(state[4], 3) == np.round(time, 3):
                    filtered_data.append(state[:2])

    return filtered_data

def ensemble_mean(list_of_lists):
    # Get the length of the internal lists
    length = len(list_of_lists[0])
    # Initialize an empty list to hold the mean values
    mean_list = [0] * length
    # Loop through the internal lists and sum the values at each index
    for lst in list_of_lists:
        for i in range(length):
            mean_list[i] += lst[i]
    # Divide each sum by the number of internal lists to get the mean value
    mean_list = [x / len(list_of_lists) for x in mean_list]
    return mean_list
    

def main():
    args, params, eval_params = parse_input_args()

    data_generator = DataGenerator(params)

    # MT3v3 setup
    model = MT3V3(params)
    checkpoint = torch.load(
        args.pretrained_model, map_location=params.training.device
    )  # Load pretrained model
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(torch.device(params.training.device))

    model.eval()
    model = None

    n_monte_carlo = 1
    mt3_gospa_scores_all = []
    mt3_gospa_loc_all = []
    mt3_gospa_false_all = []
    mt3_gospa_missed_all = []
    

    for nmc in range(n_monte_carlo):
        
        start_time = time.time()
        
        
        # Generate data
        (
            training_nested_tensor,
            _,
            _,
            _,
            trajectories,
            true_measurements,
            false_measurements,
            _,
        ) = data_generator.get_single_training_scenario()
        
        max_time = get_max_time_from_trajectory(trajectories)

        prediction_alive_threshold = 0.5
        mt3_position_estimates = []
        mt3_gospa_scores = []  # Array elements: tuple(total, loc, loc_norm, false, mis)
        mt3_gospa_loc = []
        mt3_gospa_false = []
        mt3_gospa_missed = []
        
        start = 0
        stop = int(max_time / DT)
        print(f"Running scenario from timestep {start} to {stop}")
        for timestep in range(start, stop):
            t = np.round(timestep * DT, 3)

            measurement_tensor_up_to_now = filter_by_timestamp(training_nested_tensor, t) # Can add sliding window if want
            
            (
                prediction,
                _,
                _,
                _,
                _,
            ) = model.forward(measurement_tensor_up_to_now)
            predictions = prediction.get_predictions()

            estimated_alive_targets = get_high_confidence_predictions(
                predictions, threshold=prediction_alive_threshold
            )
            current_target_position_estimates = []
            current_target_position_estimates_with_id = []
            for id, target in estimated_alive_targets:
                current_target_position_estimates.append(target[:2])
                current_target_position_estimates_with_id.append((id, target[:2]))

            mt3_position_estimates.append(current_target_position_estimates_with_id)

            # GOSPA scores
            ground_truths = filter_data_by_time(trajectories, t)
            
            (
                gospa,
                _,
                gospa_localization,
                gospa_missed,
                gospa_false,
            ) = calculate_gospa(
                ground_truths,
                current_target_position_estimates,
                c=10,
                p=1,
            )
            mt3_gospa_scores.append(gospa)
            mt3_gospa_loc.append(gospa_localization)
            mt3_gospa_false.append(gospa_false)
            mt3_gospa_missed.append(gospa_missed)
            
        mt3_gospa_scores_all.append(mt3_gospa_scores)
        mt3_gospa_loc_all.append(mt3_gospa_loc)
        mt3_gospa_false_all.append(mt3_gospa_false)
        mt3_gospa_missed_all.append(mt3_gospa_missed)

        
        print(f"Monte Carlo run number {nmc + 1} took {time.time() - start_time:.3f} seconds")
        print(f"MT3v3: \t GOSPA: {sum(mt3_gospa_scores) / len(mt3_gospa_scores):.3f}\tLoc: {sum(mt3_gospa_loc) / len(mt3_gospa_loc):.3f}\tFalse: {sum(mt3_gospa_false) / len(mt3_gospa_false):.3f}\tMissed: {sum(mt3_gospa_missed) / len(mt3_gospa_missed):.3f}")

    
    # End of monte carlo
    mt3_ensemble_mean = ensemble_mean(mt3_gospa_scores_all)
    mt3_loc_ensemble_mean = ensemble_mean(mt3_gospa_loc_all)
    mt3_false_ensemble_mean = ensemble_mean(mt3_gospa_false_all)
    mt3_missed_ensemble_mean = ensemble_mean(mt3_gospa_missed_all)
    
    ensemble_means = {
        'mt3': {
            'gospa_scores': mt3_ensemble_mean,
            'gospa_loc': mt3_loc_ensemble_mean,
            'gospa_false': mt3_false_ensemble_mean,
            'gospa_missed': mt3_missed_ensemble_mean
        },
    }

    for algo, means in ensemble_means.items():
        with open(f"{algo}_ensemble_means.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Variable', 'Value'])
            for key, value in means.items():
                writer.writerow([key, value])
    print("Monte Carlo averages")

    def calc_confidence_interval(data, confidence=0.95):
        n = len(data)
        m, se = np.mean(data), np.std(data, ddof=1) / np.sqrt(n)
        h = se * t.ppf((1 + confidence) / 2, n - 1)
        return m, h


    print("Monte Carlo averages and 95% confidence intervals (CI)")
    print(f"MT3v3: \t GOSPA: {sum(mt3_ensemble_mean) / len(mt3_ensemble_mean):.3f} ± {calc_confidence_interval(mt3_ensemble_mean)[1]:.3f} Loc: {sum(mt3_loc_ensemble_mean) / len(mt3_loc_ensemble_mean):.3f}\tFalse: {sum(mt3_false_ensemble_mean) / len(mt3_false_ensemble_mean):.3f}\tMissed: {sum(mt3_missed_ensemble_mean) / len(mt3_missed_ensemble_mean):.3f}")
        
    # plot the average scores across all Monte Carlo simulations
    # GOSPA plot
    fig, axs = plt.subplots(4, 1, figsize=(12, 12))
        
    mt3_ensemble_mean = ensemble_mean(mt3_gospa_scores_all)

    title_mt3 = "MT3v3"
    axs[0].plot(mt3_ensemble_mean, color='red', label=title_mt3)
    axs[0].set_title('Total GOSPA')
    axs[0].set_ylabel('Score')

    # Localization plot
    axs[1].plot(mt3_loc_ensemble_mean, color='red', label=title_mt3)
    axs[1].set_title('Localization')
    axs[1].set_ylabel('Score')

    # False plot
    axs[2].plot(mt3_false_ensemble_mean, color='red', label=title_mt3)
    axs[2].set_title('False')
    axs[2].set_ylabel('Score')

    # Missed plot
    axs[3].plot(mt3_missed_ensemble_mean, color='red', label=title_mt3)
    axs[3].set_title('Missed')
    axs[3].set_ylabel('Score')

    # Add legend to all subplots
    for ax in axs.flat:
        ax.legend()

    plt.tight_layout()
    plt.show()

    plot_estimates_and_measurements(trajectories, mt3_position_estimates, true_measurements, false_measurements)



if __name__ == "__main__":

    main()
