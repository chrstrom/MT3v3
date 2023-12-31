import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.patches as patches
import matplotlib.colors as colors
import matplotlib.cm as cmx
from scipy.stats import chi2

@torch.no_grad()
def output_truth_plot(
    ax, prediction, labels, matched_idx, batch, params, training_example_to_plot=0
):

    assert hasattr(
        prediction, "positions"
    ), "Positions should have been predicted for plotting."
    assert hasattr(
        prediction, "logits"
    ), "Logits should have been predicted for plotting."
    if params.data_generation.prediction_target == "position_and_velocity":
        assert hasattr(
            prediction, "velocities"
        ), "Velocities should have been predicted for plotting."

    bs, num_queries = prediction.positions.shape[:2]
    assert (
        training_example_to_plot <= bs
    ), "'training_example_to_plot' should be less than batch_size"

    if params.data_generation.prediction_target == "position_and_shape":
        raise NotImplementedError("Plotting not working yet for shape predictions.")

    # Get ground-truth, predicted state, and logits for chosen training example
    truth = labels[training_example_to_plot].cpu().numpy()
    indices = tuple(
        [t.cpu().detach().numpy() for t in matched_idx[training_example_to_plot]]
    )
    if params.data_generation.prediction_target == "position":
        out = prediction.positions[training_example_to_plot].cpu().detach().numpy()
    elif params.data_generation.prediction_target == "position_and_velocity":
        pos = prediction.positions[training_example_to_plot]
        vel = prediction.velocities[training_example_to_plot]
        out = torch.cat((pos, vel), dim=1).cpu().detach().numpy()
    out_prob = (
        prediction.logits[training_example_to_plot]
        .cpu()
        .sigmoid()
        .detach()
        .numpy()
        .flatten()
    )

    # Optionally get uncertainties for chosen training example
    if hasattr(prediction, "uncertainties"):
        uncertainties = (
            prediction.uncertainties[training_example_to_plot].cpu().detach().numpy()
        )
    else:
        uncertainties = None

    # Plot xy position of measurements, alpha-coded by time
    measurements = batch.tensors[training_example_to_plot][
        ~batch.mask[training_example_to_plot]
    ]
    colors = np.zeros((measurements.shape[0], 4))
    unique_time_values = np.array(sorted(list(set(measurements[:, 2].tolist()))))

    def f(t):
        """Exponential decay for alpha in time"""
        idx = (np.abs(unique_time_values - t)).argmin()
        return 1 / 1.2 ** (len(unique_time_values) - idx)

    colors[:, 3] = [f(t) for t in measurements[:, 2].tolist()]

    measurements_cpu = measurements.cpu()
    ax.scatter(
        measurements_cpu[:, 0]*np.cos(measurements_cpu[:, 1]),
        measurements_cpu[:, 0]*np.sin(measurements_cpu[:, 1]),
        marker="x",
        c=colors,
        zorder=np.inf,
    )

    for i in range(len(out)):
        if i in indices[0]:
            tmp_idx = np.where(indices[training_example_to_plot] == i)[0][0]
            truth_idx = indices[1][tmp_idx]

            # Plot predicted positions
            p = ax.plot(
                out[i, 0],
                out[i, 1],
                marker="o",
                label="Matched Predicted Object",
                markersize=5,
            )
            color = p[0].get_color()

            # Plot ground-truth
            truth_to_plot = truth[truth_idx]
            ax.plot(
                truth_to_plot[0],
                truth_to_plot[1],
                marker="D",
                color=color,
                label="Matched Predicted Object",
                markersize=5,
            )

            # Plot velocity
            if params.data_generation.prediction_target == "position_and_velocity":
                ax.arrow(
                    out[i, 0],
                    out[i, 1],
                    out[i, 2],
                    out[i, 3],
                    color=color,
                    head_width=0.2,
                    linestyle="--",
                    length_includes_head=True,
                )
                ax.arrow(
                    truth_to_plot[0],
                    truth_to_plot[1],
                    truth_to_plot[2],
                    truth_to_plot[3],
                    color=p[0].get_color(),
                    head_width=0.2,
                    length_includes_head=True,
                )

            # Plot uncertainties (2-sigma ellipse)
            if uncertainties is not None:
                ell_position = Ellipse(
                    xy=(out[i, 0], out[i, 1]),
                    width=uncertainties[i, 0] * 4,
                    height=uncertainties[i, 1] * 4,
                    color=color,
                    alpha=0.4,
                )
                ell_velocity = Ellipse(
                    xy=(out[i, 0] + out[i, 2], out[i, 1] + out[i, 3]),
                    width=uncertainties[i, 2] * 4,
                    height=uncertainties[i, 3] * 4,
                    edgecolor=color,
                    linestyle="--",
                    facecolor="none",
                )
                ax.add_patch(ell_position)
                ax.add_patch(ell_velocity)
        else:
            # Plot missed predictions
            p = ax.plot(
                out[i, 0],
                out[i, 1],
                marker="*",
                color="k",
                label="Unmatched Predicted Object",
                markersize=5,
            )
            if params.data_generation.prediction_target == "position_and_velocity":
                ax.arrow(
                    out[i, 0],
                    out[i, 1],
                    out[i, 2],
                    out[i, 3],
                    color="k",
                    head_width=0.2,
                    linestyle="--",
                    length_includes_head=True,
                )

        label = "{:.2f}".format(out_prob[i])
        ax.annotate(
            label,  # this is the text
            (out[i, 0], out[i, 1]),  # this is the point to label
            textcoords="offset points",  # how to position the text
            xytext=(0, 10),  # distance from text to points (x,y)
            ha="center",
            color=p[0].get_color(),
        )

@torch.no_grad()
def contrastive_classifications_plot(ax, batch, object_ids, contrastive_classifications):

    measurements = batch.tensors[0]
    n_measurements = measurements.shape[0]
    classifications = contrastive_classifications[0]
    is_there_pads = np.min(object_ids[0].numpy()) == -2
    if is_there_pads:
        n_measurements_to_use = np.argmin(object_ids[0]).item()
    else:
        n_measurements_to_use = n_measurements
    object_ids = object_ids[0][:n_measurements_to_use].int().tolist()

    # Choose random measurement (not including padding measurements)
    chosen_measurement_idx = np.random.choice(n_measurements_to_use)
    chosen_object_id = int(object_ids[chosen_measurement_idx])

    # Assign a different color (if possible) to each of the objects in the scene, but the chosen object is always blue
    # and false measurements always red
    available_colors = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive',
                        'tab:cyan', 'sandybrown', 'goldenrod', 'lime', 'cyan']
    unique_ids = set(object_ids)
    color_dict = {i: available_colors[i % len(available_colors)] for i in unique_ids if i not in [chosen_object_id, -1]}
    color_dict[chosen_object_id] = 'tab:blue'
    color_dict[-1] = 'tab:red'
    bar_colors = list(map(color_dict.get, object_ids))

    # Plot color-coded predicted pmf for the chosen measurement
    chosen_classifications = classifications[chosen_measurement_idx].exp().detach()
    ax.bar(range(n_measurements_to_use), chosen_classifications.numpy()[:n_measurements_to_use], color=bar_colors)

@torch.no_grad()
def compute_avg_certainty(outputs_history, matched_idx_history):
    matched_certainties = []
    unmatched_certainties = []
    for outputs, matched_idx in zip(outputs_history, matched_idx_history):
        idx = _get_src_permutation_idx(matched_idx)
        output_logits = outputs['logits']

        mask = torch.zeros_like(output_logits).bool()
        mask[idx] = True
        matched_certainties.extend(output_logits[mask].sigmoid().cpu().tolist())
        unmatched_certainties.extend(output_logits[~mask].sigmoid().cpu().tolist())

    if len(matched_certainties) > 0:
        matched_quants = np.quantile(matched_certainties, [0.0, 0.25, 0.5, 0.75, 1.0])
    else:
        matched_quants = [-1, -1, -1, -1, -1]

    if len(unmatched_certainties) > 0:
        unmatched_quants = np.quantile(unmatched_certainties, [0.0, 0.25, 0.5, 0.75, 1.0])
    else:
        unmatched_quants = [-1, -1, -1, -1, -1]

    return tuple(matched_quants), tuple(unmatched_quants)


def _get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i)
                        for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx
    

def get_constrastive_ax():
    fig, ax = plt.subplots()
    ax.grid('on')
    line, = ax.plot([1], 'r', label='Contrastive loss')
    ax.set_ylabel('Contrastive loss')

    return fig, ax, line

def get_state_uncertainties_ax():
    fig, ax = plt.subplots()
    ax.grid('on')
    line_x, = ax.plot([1], label='x', c='C0')
    line_y, = ax.plot([1], label='y', c='C1')
    line_vx, = ax.plot([1], label='v_x', c='C2')
    line_vy, = ax.plot([1], label='v_y', c='C3')
    ax.set_ylabel('Average standard deviation of predictions')

    return fig, ax, (line_x, line_y, line_vx, line_vy)

def get_false_ax():
    fig, ax = plt.subplots()
    ax.grid('on')
    line, = ax.plot([1], 'r', label='False loss')
    ax.set_ylabel('False loss')

    return fig, ax, line

def get_total_loss_ax():
    fig, ax = plt.subplots()
    ax.grid('on')
    line, = ax.plot([1], 'r', label='Total loss')
    ax.set_ylabel('Total loss')
    ax.set_yscale('log')
    
    return fig, ax, line

def get_new_ax(log=False, ylabel='Loss'):
    fig, ax = plt.subplots()
    ax.grid('on')
    line, = ax.plot([1], 'r', label='Loss')
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale('log')

    return fig, ax, line
