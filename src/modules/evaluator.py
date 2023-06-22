import time

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from modules.loss import MotLoss

from util.misc import NestedTensor


def compute_losses(outputs, labels, contrastive_classifications, unique_ids, mot_loss, contrastive_loss):
    c_loss = contrastive_loss.forward(contrastive_classifications, unique_ids)
    d_loss,_ = detr_loss.forward(outputs, labels, loss_type = 'detr')
    return d_loss, c_loss


def compute_gospa(outputs, labels, mot_loss, existence_threshold=0.9):
    loss, indices, decomposition = mot_loss.gospa_forward(outputs, labels, False, existence_threshold)
    return loss, indices, decomposition


def compute_nees(outputs, labels, params):
    all_nees_samples = []
    for batch_idx in range(len(outputs['state'])):
        alive_idx = outputs['logits'][batch_idx].sigmoid().squeeze(-1) > params.loss.existence_prob_cutoff
        alive_output = outputs['state'][batch_idx, alive_idx]
        alive_covariances = outputs['state_covariances'][batch_idx][alive_idx]
        n_predictions = len(alive_output)
        n_ground_truths = len(labels[batch_idx])
        if len(alive_covariances.shape) == 2:
            distribution_type = Normal
            scale_params = alive_covariances.sqrt()
        else:
            distribution_type = MultivariateNormal
            scale_params = alive_covariances

        # Compute cost matrix as the NLL of the ground-truth samples from the predicted distributions
        cost_matrix = torch.zeros(n_predictions, n_ground_truths)
        for i in range(n_predictions):
            prediction = alive_output[i]
            predicted_distribution = distribution_type(prediction, scale_params[i])
            for j in range(n_ground_truths):
                ground_truth = labels[batch_idx][j]
                cost_matrix[i, j] = -predicted_distribution.log_prob(ground_truth).sum().item()
        cost_matrix = cost_matrix.clamp(max=params.loss.nees.cutoff_distance)
        prediction_idxs, ground_truth_idxs = linear_sum_assignment(cost_matrix)

        # For each match, compute the NEES score and add it to running sum
        for prediction_idx, ground_truth_idx in zip(*(prediction_idxs, ground_truth_idxs)):

            # Skip bad matches
            if cost_matrix[prediction_idx, ground_truth_idx] >= params.loss.nees.cutoff_distance:
                continue

            predicted_state = alive_output[prediction_idx]
            covariance = alive_covariances[prediction_idx]
            target = labels[batch_idx][ground_truth_idx]

            error = (predicted_state-target)[:, None]
            if len(covariance.shape) == 1:
                p_inv = torch.diag(covariance).inverse()
            else:
                p_inv = covariance.inverse()

            nees = (error.t() @ p_inv @ error).item()
            all_nees_samples.append(nees)

    return all_nees_samples


def compute_error_histogram(outputs, labels, params):
    errors_x = []
    errors_vx = []
    stds_x = []
    stds_vx = []

    for batch_idx in range(len(outputs['state'])):
        alive_idx = outputs['logits'][batch_idx].sigmoid().squeeze(-1) > params.loss.existence_prob_cutoff
        alive_output = outputs['state'][batch_idx, alive_idx]
        alive_covariances = outputs['state_covariances'][batch_idx][alive_idx]
        n_predictions = len(alive_output)
        n_ground_truths = len(labels[batch_idx])
        if len(alive_covariances.shape) == 2:
            distribution_type = Normal
            alive_covariances = alive_covariances.sqrt()
        else:
            distribution_type = MultivariateNormal

        # Compute cost matrix as the NLL of the ground-truth samples from the predicted distributions
        cost_matrix = torch.zeros(n_predictions, n_ground_truths)
        for i in range(n_predictions):
            prediction = alive_output[i]
            covariance = alive_covariances[i]
            predicted_distribution = distribution_type(prediction, covariance)
            for j in range(n_ground_truths):
                ground_truth = labels[batch_idx][j]
                cost_matrix[i, j] = -predicted_distribution.log_prob(ground_truth).sum().item()
        prediction_idxs, ground_truth_idxs = linear_sum_assignment(cost_matrix)

        # For each match, save corresponding error and standard deviations
        for prediction_idx, ground_truth_idx in zip(*(prediction_idxs, ground_truth_idxs)):

            predicted_state = alive_output[prediction_idx]
            covariance = alive_covariances[prediction_idx]
            target = labels[batch_idx][ground_truth_idx]

            error = (predicted_state-target).abs()
            errors_x.append(error[0].item())
            errors_vx.append(error[2].item())
            if len(covariance.shape) != 1:
                covariance = covariance.sqrt()
            stds_x.append(covariance[0, 0].item())
            stds_vx.append(covariance[2, 2].item())

    errors = (errors_x, errors_vx)
    stds = (stds_x, stds_vx)
    return errors, stds


def create_random_baseline(params):
    bs = params.training.batch_size
    nq = params.arch.num_queries
    if params.data_generation.prediction_target == 'position':
        d_det = params.arch.d_detections
    elif params.data_generation.prediction_target == 'position_and_velocity':
        d_det = params.arch.d_detections * 2
    else:
        raise NotImplementedError
    lb = params.data_generation.field_of_view_lb
    ub = params.data_generation.field_of_view_ub
    out = {}
    state = np.random.uniform(low=lb, high=ub, size=(bs, nq, d_det))
    if params.arch.d_prediction==5:
        V = params.data_generation.V0
        v = params.data_generation.v0
        ext = np.diag(V)/(v-6)
        state_ext = np.expand_dims(np.repeat(np.expand_dims(ext,0),nq,axis=0),axis=0)
        tmp = np.zeros((bs, nq, params.arch.d_prediction))
        tmp[:,:,:2] = state
        tmp[:,:,2:-1] = state_ext
        state = tmp

    logits = np.log(np.ones(shape=(bs,nq,1))) 

    out['state'] = torch.Tensor(state).to(torch.device(params.training.device))
    out['logits'] = torch.Tensor(logits).to(torch.device(params.training.device))

    return out


def create_all_measurement_baseline(batch, params):
    bs = params.training.batch_size
    nq = params.model.num_queries
    d_det = params.model.d_detections
    out = {}
    state = np.zeros(shape=(bs, nq, d_det))
    logits = np.log(np.ones(shape=(bs, nq, 1)) * 0.01)

    for i in range(bs):
        idx = batch[i,:,-1] == (params.general.n_timesteps - 1 - params.general.n_prediction_lag)*params.general.dt
        b = batch[i,idx] 
        num_meas = b.shape[0]
        state[i,:num_meas,:] = b[:,:d_det].cpu()
        logits[i,:num_meas,:] = np.log(0.99/(1-0.99))  

    out['state'] = torch.Tensor(state).to(torch.device(params.training.device))
    out['logits'] = torch.Tensor(logits).to(torch.device(params.training.device))

    return out


def create_true_measurement_baseline(batch, unique_ids, params):
    bs = params.training.batch_size
    nq = params.arch.num_queries
    d_det = params.arch.d_detections
    out = {}
    state = np.zeros(shape=(bs, nq, d_det))
    logits = np.log(np.ones(shape=(bs, nq, 1)) * 0.01)

    for i in range(bs):
        time_idx = batch[i,:,-1] == (params.data_generation.n_timesteps - 1 - params.data_generation.n_prediction_lag)*params.data_generation.dt
        true_idx = torch.logical_not(torch.logical_or(unique_ids[i,:]==-1, unique_ids[i,:]==-2))
        idx = torch.logical_and(time_idx, true_idx)
        if not torch.any(idx):
            continue
            
        b = batch[i,idx].cpu()
        ids = unique_ids[i,idx] 
        for obj_i, j in enumerate(torch.unique(ids)):
            tmp = ids==j
            state[i,obj_i,:] = b[tmp].mean(axis=0)[:d_det]

        logits[i,:obj_i+1,:] = np.log(0.99/(1-0.99))

    if params.arch.d_prediction==5:
        V = params.data_generation.V0
        v = params.data_generation.v0
        ext = np.diag(V)/(v-6)
        state_ext = np.expand_dims(np.repeat(np.expand_dims(ext,0),nq,axis=0),axis=0)
        tmp = np.zeros((bs, nq, params.arch.d_prediction))
        tmp[:,:,:2] = state
        tmp[:,:,2:-1] = state_ext
        state = tmp

    out['state'] = torch.Tensor(state).to(torch.device(params.training.device))
    out['logits'] = torch.Tensor(logits).to(torch.device(params.training.device))

    return out




def visualize_attn_maps(batch, outputs, attn_maps, ax, object_to_visualize=0, layer_to_visualize=-1):
    if type(attn_maps) == list:
        attn_weights = attn_maps[layer_to_visualize][0,object_to_visualize,:].detach().numpy()
        nq = attn_weights.shape[0]
        number_decoder_layers = len(attn_maps)
    else:
        nq = attn_maps.shape[2]
        number_decoder_layers = attn_maps.shape[1]
        attn_maps = attn_maps.detach()
        attn_weights = attn_maps[0,layer_to_visualize,object_to_visualize, :].numpy()

    assert 0 <= object_to_visualize < nq, f"object to visualize should be in range of {0}-{nq}"
    if 'aux_outputs' in outputs and layer_to_visualize < number_decoder_layers - 1:
        outputs_state = outputs['aux_outputs'][layer_to_visualize]['state'].detach()
        outputs_prob = outputs['aux_outputs'][layer_to_visualize]['logits'].sigmoid().detach()
    else:
        outputs_state = outputs['state'].detach()
        outputs_prob = outputs['logits'].sigmoid().detach()

    if type(attn_maps) == list:
        if layer_to_visualize > 0:
            measurements = outputs['aux_outputs'][layer_to_visualize-1]['state'].detach().numpy()[0]
        else:
            measurements = outputs['enc_outputs']['state'].detach().numpy()[0]
    else:
        measurements = batch[0,:,:-1].numpy()

    object_pos = outputs_state[0,object_to_visualize,:].numpy()
    object_prob = outputs_prob[0,object_to_visualize,:].numpy()

    label = "{:.2f}".format(object_prob[0])
    ax.annotate(label, # this is the text
                (object_pos[0], object_pos[1]), # this is the point to label
                textcoords="offset points", # how to position the text
                xytext=(0,10), # distance from text to points (x,y)
                ha='center',
                color='g')
    ax.plot(object_pos[0], object_pos[1], marker='o', color='g', label='Predicted object position', markersize=10)
    colors = np.zeros((measurements.shape[0], 4))
    colors[:, 3] = attn_weights/np.linalg.norm(attn_weights)
    ax.scatter(measurements[:,0], measurements[:,1], marker='x', color=colors, s=64)
    ax.legend()


def evaluate_gospa(data_generator, model, eval_params):
    with torch.no_grad():
        model.eval()
        mot_loss = MotLoss(eval_params)
        gospa_total = 0
        gospa_loc = 0
        gospa_norm_loc = 0
        gospa_miss = 0
        gospa_false = 0

        for i in range(eval_params.n_samples):
            # Get batch from data generator and feed it to trained model
            batch, labels, unique_ids, _, trajectories = data_generator.get_batch()
            prediction, _, _, _, _ = model.forward(batch)

            # Compute GOSPA score
            prediction_in_format_for_loss = {'state': torch.cat((prediction.positions, prediction.velocities), dim=2),
                                             'logits': prediction.logits,
                                             'state_covariances': prediction.uncertainties ** 2}
            loss, _, decomposition = mot_loss.compute_orig_gospa_matching(prediction_in_format_for_loss, labels,
                                                                          eval_params.loss.existence_prob_cutoff)
            gospa_total += loss.item()
            gospa_loc += decomposition['localization']
            gospa_norm_loc += decomposition['localization'] / decomposition['n_matched_objs'] if \
                decomposition['n_matched_objs'] != 0 else 0.0
            gospa_miss += decomposition['missed']
            gospa_false += decomposition['false']

        model.train()
        gospa_total /= eval_params.n_samples
        gospa_loc /= eval_params.n_samples
        gospa_norm_loc /= eval_params.n_samples
        gospa_miss /= eval_params.n_samples
        gospa_false /= eval_params.n_samples
    return gospa_total, gospa_loc, gospa_norm_loc, gospa_miss, gospa_false