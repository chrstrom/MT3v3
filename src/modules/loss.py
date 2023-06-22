import warnings
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from scipy.optimize import linear_sum_assignment
from torch.autograd import Variable

# Sqrt of matrix from https://github.com/msubhransu/matrix-sqrt/blob/cc2289a3ed7042b8dbacd53ce8a34da1f814ed2f/matrix_sqrt.py#L54
def compute_error(A, sA):
  normA = torch.sqrt(torch.sum(torch.sum(A * A, dim=1),dim=1))
  error = A - torch.bmm(sA, sA)
  error = torch.sqrt((error * error).sum(dim=1).sum(dim=1)) / normA
  return torch.mean(error)

# Forward via Newton-Schulz iterations
# Backward via autograd
def sqrt_newton_schulz_autograd(A, numIters, dtype, tol=1e-4):
    batchSize = A.data.shape[0]
    dim = A.data.shape[1]
    normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
    Y = A.div(normA.view(batchSize, 1, 1).expand_as(A))
    I = Variable(torch.eye(dim,dim).view(1, dim, dim).
                repeat(batchSize,1,1).type(dtype),requires_grad=False).to(A.device)
    Z = Variable(torch.eye(dim,dim).view(1, dim, dim).
                repeat(batchSize,1,1).type(dtype),requires_grad=False).to(A.device)

    for i in range(numIters):
        T = 0.5*(3.0*I - Z.bmm(Y))
        Y = Y.bmm(T)
        Z = T.bmm(Z)
        if i % 10:
            sA = Y*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
            error = compute_error(A, sA)
            if abs(error) < tol:
                return sA, error
                
    sA = Y*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
    error = compute_error(A, sA)
    return sA, error

def check_gospa_parameters(c, p, alpha):
    """ Check parameter bounds.

    If the parameter values are outside the allowable range specified in the
    definition of GOSPA, a ValueError is raised.
    """
    if alpha <= 0 or alpha > 2:
        raise ValueError("The value of alpha is outside the range (0, 2]")
    if c <= 0:
        raise ValueError("The cutoff distance c is outside the range (0, inf)")
    if p < 1:
        raise ValueError("The order p is outside the range [1, inf)")

def compute_wasserstein(src, target, dim=2, scaling=1):
    num_cov_params = int((dim**2 - dim)/2 + dim)
    src_pos = src[:,:dim]*scaling
    src_cov_flat = src[:,-num_cov_params:]

    # Fill first diagonal of prediction
    src_cov = torch.diag_embed(src_cov_flat[:,:dim])
    
    if dim == 2:
        # P is rotation matrix, last prediction is angle of ellipse
        cosines = torch.cos(src_cov_flat[:,-1]).reshape(-1,1,1)
        sinuses = torch.sin(src_cov_flat[:,-1]).reshape(-1,1,1)
        row1 = torch.cat((cosines, -sinuses), dim=-1)
        row2 = torch.cat((sinuses, cosines), dim=-1)
        P = torch.cat((row1,row2), dim=-2)
        # Apply rotation
        src_cov = P@src_cov@P.permute(0,2,1)*scaling
    elif dim >= 2:
        raise Exception('Wasserstein metric not implemented for 3D and above')

    target_pos = target[:,:dim]*scaling
    tgt_cov_flat = target[:,-dim*dim:]*scaling
    # Reshape target to dim x dim matrices
    tgt_cov = tgt_cov_flat.reshape(-1,dim,dim)
    tgt_cov_sqrt, _ = sqrt_newton_schulz_autograd(tgt_cov, 100, tgt_cov.dtype)
    prod = tgt_cov_sqrt@src_cov@tgt_cov_sqrt
    prod_sqrt, _ = sqrt_newton_schulz_autograd(prod, 100, prod.dtype)
    trace_part = src_cov + tgt_cov - 2*prod_sqrt
    trace = torch.diagonal(trace_part, dim1=-2, dim2=-1).sum(-1)

    loss = F.mse_loss(src_pos, target_pos, reduction='none')
    loss = torch.sum(loss, dim=-1) + trace
    loss = torch.mean(torch.sqrt(loss))
    
    return loss

#order, cutoff_distance, alpha=2, num_classes=1, device='cpu'
class MotLoss(nn.Module):
    def __init__(self, params):
        super().__init__()
        if params.loss.type == 'gospa':
            check_gospa_parameters(params.loss.cutoff_distance, params.loss.order, params.loss.alpha)
            self.order = params.loss.order
            self.cutoff_distance = params.loss.cutoff_distance
            self.alpha = params.loss.alpha
            self.miss_cost = self.cutoff_distance ** self.order
        self.params = params
        self.device = torch.device(params.training.device)
        self.to(self.device)

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i)
                              for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i)
                              for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def compute_hungarian_matching(self, predicted_states, predicted_logits, targets, distance='detr', scaling=1):
        """ Performs the matching

        Params:
            outputs: dictionary with 'state' and 'logits'
                state: Tensor of dim [batch_size, num_queries, d_label]
                logits: Tensor of dim [batch_size, num_queries, number_of_classes]

            targets: This is a list of targets (len(targets) = batch_size), where each target is a
                    tensor of dim [num_objects, d_label] (where num_objects is the number of ground-truth
                    objects in the target)

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        bs, num_queries = predicted_states.shape[:2]
        predicted_probabilities = predicted_logits.sigmoid()

        indices = []
        for i in range(bs):
            # Compute cost matrix for this batch position
            cost = torch.cdist(predicted_states[i], targets[i], p=2)
            cost -= predicted_probabilities[i].log()

            # Compute minimum cost assignment and save it
            with torch.no_grad():
                indices.append(linear_sum_assignment(cost.cpu()))

        permutation_idx = [(torch.as_tensor(i, dtype=torch.int64).to(self.device),
                            torch.as_tensor(j, dtype=torch.int64).to(self.device)) for i, j in indices]

        return permutation_idx, cost.to(self.device)

    def compute_orig_gospa_matching(self, outputs, targets, existence_threshold):
        """ Performs the matching. Note that this can NOT be used as a loss function

        Params:
            outputs: dictionary with 'state' and 'logits'
                state: Tensor of dim [batch_size, num_queries, d_label]
                logits: Tensor of dim [batch_size, num_queries, number_of_classes]

            targets: This is a list of targets (len(targets) = batch_size), where each target is a
                    tensor of dim [num_objects, d_label] (where num_objects is the number of ground-truth
                    objects in the target)

            existence_threshold: Float in range (0,1) that decides which object are considered alive and which are not.

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        assert 'state' in outputs, "'state' should be in dict"
        assert 'logits' in outputs, "'logits' should be in dict"
        assert self.order == 1, 'This code does not work for loss.order != 1'
        assert self.alpha == 2, 'The permutation -> assignment relation used to decompose GOSPA might require that loss.alpha == 2'

        output_state = outputs['state'].detach()
        output_existence_probabilities = outputs['logits'].sigmoid().detach()

        bs, num_queries = output_state.shape[:2]
        dim_predictions = output_state.shape[2]
        dim_targets = targets[0].shape[1]
        assert dim_predictions == dim_targets

        loss = torch.zeros(size=(1,))
        localization_cost = 0
        missed_target_cost = 0
        false_target_cost = 0
        indices = []

        for i in range(bs):
            alive_idx = output_existence_probabilities[i, :].squeeze(-1) > existence_threshold
            alive_output = output_state[i, alive_idx, :]
            current_targets = targets[i]
            permutation_length = 0

            if len(current_targets) == 0:
                indices.append(([], []))
                loss += torch.Tensor([self.miss_cost/self.alpha * len(alive_output)])
                false_target_cost = self.miss_cost/self.alpha * len(alive_output)
            elif len(alive_output) == 0:
                indices.append(([], []))
                loss += torch.Tensor([self.miss_cost/self.alpha * len(current_targets)])
                missed_target_cost = self.miss_cost / self.alpha * len(current_targets)
            else:
                dist = torch.cdist(alive_output, current_targets, p=2)
                dist = dist.clamp_max(self.cutoff_distance)
                c = torch.pow(input=dist, exponent=self.order)
                c = c.cpu()
                output_idx, target_idx = linear_sum_assignment(c)
                indices.append((output_idx, target_idx))

                for t, o in zip(output_idx, target_idx):
                    loss += c[t,o]
                    if c[t, o] < self.cutoff_distance:
                        localization_cost += c[t, o].item()
                        permutation_length += 1
                
                cardinality_error = abs(len(alive_output) - len(current_targets))
                loss += self.miss_cost/self.alpha * cardinality_error

                missed_target_cost += (len(current_targets) - permutation_length) * (self.miss_cost/self.alpha)
                false_target_cost += (len(alive_output) - permutation_length) * (self.miss_cost/self.alpha)

        decomposition = {'localization': localization_cost, 'missed': missed_target_cost, 'false': false_target_cost,
                         'n_matched_objs': permutation_length}
        return loss, indices, decomposition

    def compute_orig_gospa_matching_with_uncertainties(self, predictions, targets, existence_threshold):

        assert self.order == 1, 'This code does not work for loss.order != 1'
        assert self.alpha == 2, 'The permutation -> assignment relation used to decompose GOSPA might require that loss.alpha == 2'

        batch_size, _, dim_predictions = predictions['state'].shape
        n_targets, dim_targets = targets[0].shape
        assert dim_predictions == dim_targets
        assert batch_size == 1, 'GOSPA matching with uncertainties currently only works with batch size = 1'

        existence_probabilities = predictions['logits'][0].sigmoid().detach()
        alive_idx = existence_probabilities.squeeze(-1) > existence_threshold

        predicted_distributions = {'states': predictions['state'][0, alive_idx].detach(),
                                   'state_covariances': predictions['state_covariances'][0, alive_idx].detach()}
        targets = targets[0]
        n_predictions = len(predicted_distributions['states'])

        loss = torch.zeros(size=(1,))
        localization_cost = 0
        missed_target_cost = 0
        false_target_cost = 0
        indices = []
        permutation_length = 0

        if n_targets == 0:
            indices.append(([], []))
            loss += torch.Tensor([self.miss_cost / self.alpha * n_predictions])
            false_target_cost = self.miss_cost / self.alpha * n_predictions
        elif n_predictions == 0:
            indices.append(([], []))
            loss += torch.Tensor([self.miss_cost / self.alpha * n_targets])
            missed_target_cost = self.miss_cost / self.alpha * n_targets
        else:
            dist = compute_pairwise_crossentropy(predicted_distributions, targets)
            dist = dist.clamp_max(self.cutoff_distance)
            c = torch.pow(input=dist, exponent=self.order)
            c = c.cpu()
            target_idx, output_idx = linear_sum_assignment(c)
            indices.append((target_idx, output_idx))

            for t, o in zip(target_idx, output_idx):
                loss += c[t, o]
                if c[t, o] < self.cutoff_distance:
                    localization_cost += c[t, o].item()
                    permutation_length += 1

            cardinality_error = abs(n_predictions - n_targets)
            loss += self.miss_cost / self.alpha * cardinality_error

            missed_target_cost += (n_targets - permutation_length) * (self.miss_cost / self.alpha)
            false_target_cost += (n_predictions - permutation_length) * (self.miss_cost / self.alpha)

        decomposition = {'localization': localization_cost, 'missed': missed_target_cost, 'false': false_target_cost,
                         'n_matched_objs': permutation_length}
        return loss, indices, decomposition

    def compute_prob_gospa_matching(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: dictionary with 'state' and 'logits'
                state: Tensor of dim [batch_size, num_queries, d_label]
                logits: Tensor of dim [batch_size, num_queries, number_of_classes]

            targets: This is a list of targets (len(targets) = batch_size), where each target is a
                    tensor of dim [num_objects, d_label] (where num_objects is the number of ground-truth
                    objects in the target)

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        output_state = outputs['state']
        output_logits = outputs['logits'].sigmoid()

        bs, num_queries = output_state.shape[:2]

        # We flatten to compute the cost matrices in a batch
        # [batch_size * num_queries, d_label]
        out = output_state.flatten(0, 1)
        probs = output_logits.flatten(0, 1)
        # Also concat the target labels
        # [sum(num_objects), d_labels]
        tgt = torch.cat(targets)

        # Compute the L2 cost
        # [batch_size * num_queries, sum(num_objects)]
        assert probs.shape[0] == bs * num_queries
        assert probs.shape[1] == 1

        dist = torch.cdist(out, tgt, p=2)
        dist = dist.clamp_max(self.cutoff_distance)

        cost = torch.pow(input=dist, exponent=self.order) * probs
        cost += (1-probs) * (self.miss_cost) / 2.0

        assert cost.shape[0] == bs * num_queries
        assert cost.shape[1] == tgt.shape[0]

        # Clamp according to GOSPA
        # cost = cost.clamp_max(self.miss_cost)

        # Reshape
        # [batch_size, num_queries, sum(num_objects)]
        #cost = cost.view(bs, num_queries, -1)
        cost = cost.view(bs, num_queries, -1).cpu()

        # List with num_objects for each training-example
        sizes = [len(v) for v in targets]

        # Perform hungarian matching using scipy linear_sum_assignment
        with torch.no_grad():
            cost_split = cost.split(sizes, -1)
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_split)]
            
            permutation_idx = []
            unmatched_x = []
            for i, perm_idx in enumerate(indices):
                pred_idx, ground_truth_idx = perm_idx
                pred_unmatched = list(set(i for i in range(num_queries)) - set(pred_idx))

                permutation_idx.append((torch.as_tensor(pred_idx, dtype=torch.int64).to(self.device), torch.as_tensor(ground_truth_idx, dtype=torch.int64)))
                unmatched_x.append(torch.as_tensor(pred_unmatched, dtype=torch.int64).to(self.device))
               
        return permutation_idx, cost.to(self.device), unmatched_x
    
    def gospa_forward(self, outputs, targets, probabilistic=True, existence_threshold=0.75):

        assert 'state' in outputs, "'state' should be in dict"
        assert 'logits' in outputs, "'logits' should be in dict"

        output_state = outputs['state']
        output_logits = outputs['logits'].sigmoid()
        # List with num_objects for each training-example
        sizes = [len(v) for v in targets]

        bs = output_state.shape[0]
        if probabilistic:
            indices, cost_matrix, unmatched_x = self.compute_prob_gospa_matching(outputs, targets)
            cost_matrix = cost_matrix.split(sizes, -1)
            loss = 0
            for i in range(bs):
                batch_idx = indices[i]
                batch_cost = cost_matrix[i][i][batch_idx].sum()
                batch_cost = batch_cost + output_logits[i][unmatched_x[i]].sum() * self.miss_cost/2.0
                loss = loss + batch_cost
            loss = loss/sum(sizes)
            return loss, indices
        else:
            assert 0 <= existence_threshold < 1, "'existance_threshold' should be in range (0,1)"
            if 'state_covariances' in outputs and False:  # Not ready for use yet
                loss, indices, decomposition = self.compute_orig_gospa_matching_with_uncertainties(outputs,
                                                                                                   targets,
                                                                                                   existence_threshold)
            else:
                loss, indices, decomposition = self.compute_orig_gospa_matching(outputs, targets, existence_threshold)
            loss = loss / bs
            return loss, indices, decomposition

    def state_loss(self, predicted_states, targets, indices, uncertainties=None):
        idx = self._get_src_permutation_idx(indices)
        matched_predicted_states = predicted_states[idx]
        target = torch.cat([t[i] for t, (_, i) in zip(targets, indices)], dim=0)
        if uncertainties is not None:
            matched_uncertainties = uncertainties[idx]
            prediction_distribution = torch.distributions.normal.Normal(matched_predicted_states, matched_uncertainties)
            loss = -prediction_distribution.log_prob(target).mean()
        else:
            loss = F.l1_loss(matched_predicted_states, target, reduction='none').sum(-1).mean()

        return loss

    def logits_loss(self, predicted_logits, targets, indices):
        idx = self._get_src_permutation_idx(indices)
        
        target_classes = torch.zeros_like(predicted_logits, device=predicted_logits.device)
        target_classes[idx] = 1.0  # this is representation of an object

        gt_objects = torch.Tensor([t.shape[0] for t in targets]).to(predicted_logits.device) # mean (over batch) number of ground truth objects
        loss = F.binary_cross_entropy_with_logits(predicted_logits.squeeze(-1).permute(1,0), target_classes.squeeze(-1).permute(1,0))

        return loss

    def wasserstein_loss(self, outputs, targets, indices, scaling=1):
        dim = len(self.params.data_generation.std_x0[0])
        target = torch.cat([t[i] for t, (_, i) in zip(targets, indices)], dim=0)
        idx = self._get_src_permutation_idx(indices)
        src = outputs['state'][idx]

        loss = compute_wasserstein(src, target, dim=dim, scaling=scaling)
        
        return loss

    def get_loss(self, prediction, targets, loss_type, existance_threshold=None):
        # Create state vectors for the predictions, based on prediction target specified by user
        if self.params.data_generation.prediction_target == 'position':
            predicted_states = prediction.positions
        elif self.params.data_generation.prediction_target == 'position_and_velocity':
            predicted_states = torch.cat((prediction.positions, prediction.velocities), dim=2)
        else:
            raise NotImplementedError(f'Hungarian matching not implemented for prediction target '
                                      f'{self.params.data_generation.prediction_target}')

        if loss_type == 'gospa':
            loss, indices = self.gospa_forward(prediction, targets, probabilistic=True)
            loss = {f'{loss_type}_state': loss, f'{loss_type}_logits': 0}
        elif loss_type == 'gospa_eval':
            loss,_ = self.gospa_forward(prediction, targets, probabilistic=False, existence_threshold=existance_threshold)
            indices = None
            loss = {f'{loss_type}_state': loss, f'{loss_type}_logits': 0}
        elif loss_type == 'detr':
            indices, _ = self.compute_hungarian_matching(predicted_states, prediction.logits, targets)
            log_loss = self.logits_loss(prediction.logits, targets, indices)
            if hasattr(prediction, 'uncertainties'):
                state_loss = self.state_loss(predicted_states, targets, indices, uncertainties=prediction.uncertainties)
            else:
                state_loss = self.state_loss(predicted_states, targets, indices)
            loss = {f'{loss_type}_state': state_loss, f'{loss_type}_logits': log_loss}
        elif loss_type == 'wasserstein':
            scale = 1
            indices, _ = self.compute_hungarian_matching(prediction, targets, distance='wasserstein', scaling=scale)
            loss = self.wasserstein_loss(prediction, targets, indices, scaling=scale)
            log_loss = self.logits_loss(prediction, targets, indices)
            loss = loss = {f'{loss_type}_state': loss, f'{loss_type}_logits': log_loss}
        
        return loss, indices
    
    def forward(self, targets, prediction, intermediate_predictions=None, encoder_prediction=None, loss_type='detr',
                existence_threshold=0.75):
        if loss_type not in ['gospa', 'gospa_eval', 'detr', 'wasserstein']:
            raise NotImplementedError(f"The loss type '{loss_type}' was not implemented.'")

        losses = {}
        loss, indices = self.get_loss(prediction, targets, loss_type, existence_threshold)
        losses.update(loss)

        if intermediate_predictions is not None:
            for i, intermediate_prediction in enumerate(intermediate_predictions):
                aux_loss, _ = self.get_loss(intermediate_prediction, targets, loss_type, existence_threshold)
                aux_loss = {f'{k}_{i}': v for k, v in aux_loss.items()}
                losses.update(aux_loss)

        if encoder_prediction is not None:
            enc_loss, _ = self.get_loss(encoder_prediction, targets, loss_type, existence_threshold)
            enc_loss = {f'{k}_enc': v for k, v in enc_loss.items()}
            losses.update(enc_loss)

        return losses, indices


class FalseMeasurementLoss(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.device = torch.device(params.training.device)
        self.to(self.device)

    def forward(self, log_classifications, unique_ids):
        output = log_classifications.squeeze(axis=-1).permute(1,0).flatten()
        i = unique_ids.flatten().long()
        output = output[i != -2]
        i = i[i != -2]
        tgt = torch.zeros_like(output, device=output.device)
        tgt[i == -1] = 1
        

        weight = (tgt.numel()-tgt.sum()) / tgt.sum() if tgt.sum() > 0 else None
        loss = torch.nn.functional.binary_cross_entropy_with_logits(output, tgt, pos_weight=weight)
        return loss


class EncoLoss(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.device = torch.device(params.training.device)
        self.to(self.device)

    def forward(self, src, trajectories):
        # bs x n
        time_idx = src['time']
        # bs x n x 2
        state = src['state'][...,:2]
        # bs x n x 2
        tgt = torch.zeros_like(state)
        unique_ids = src['unique_ids']
        bs = len(trajectories)

        for i in range(bs):
            for k,v in trajectories[i].items():
                tgt_states = torch.from_numpy(v).float().to(state.device)
                for j in range(len(v)):
                    tgt_state = tgt_states[j,:2]
                    t = torch.round(tgt_states[j,4]*10)
                    obj_id = unique_ids[i] == k
                    time_id = time_idx[i] == t
                    idx = torch.logical_and(time_id, obj_id) 
                    tgt[i,idx,:] = tgt_state
        

        state = state.flatten(0,1)
        tgt = tgt.flatten(0,1)
        valid = unique_ids.flatten() >= 0
        loss = torch.nn.functional.l1_loss(state[valid,:], tgt[valid,:],reduction='none').sum(-1).mean()

        return loss


class DencoLoss(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.device = torch.device(params.training.device)
        self.to(self.device)
    
    def forward(self, src, trajectories, only_l1=False):
        bs, ts, d_pred = src['state'].shape
        tgt_probs = torch.zeros((bs,ts), device=src['state'].device)
        d_label = d_pred if d_pred == 2 else d_pred + 1
        tgt_state = torch.zeros((bs, ts, d_label), device=src['state'].device)

        for i in range(bs):
            tmp = None
            for k,v in trajectories[i].items():
                if d_pred > 2:
                    t = np.concatenate((v[:, :2], v[:, 5:]), axis=1)
                else:
                    t = v[:,:2]

                time = np.round(v[:,4]/self.params.data_generation.dt).astype(int)
                tgt_state[i,time,:] = torch.from_numpy(t).float().to(src['state'].device)
                tgt_probs[i,time] = 1

        valid_states_idx = tgt_probs == 1
        valid_states_idx = valid_states_idx.flatten()
        src_state = src['state'].flatten(0,1)[valid_states_idx,:]
        tgt_state = tgt_state.flatten(0,1)[valid_states_idx,:]
        if d_pred == 2:
            state_loss = torch.nn.functional.l1_loss(src_state, tgt_state, reduction='none').sum(-1).mean()
        else:
            if only_l1:
                a = tgt_state[:,2]
                b = tgt_state[:,3]
                c = tgt_state[:,5]
                lamba_1 = (a+c)/2 + torch.sqrt(((a-c)/2)**2 + b**2)
                lamba_2 = (a+c)/2 - torch.sqrt(((a-c)/2)**2 + b**2)
                x_radii = torch.sqrt(lamba_1)
                y_radii = torch.sqrt(lamba_2)
                angle = torch.atan2(lamba_1-a,b)
                cond1 = torch.logical_and(b==0, a>=c)
                cond2 = torch.logical_and(b==0, a<c)
                angle[cond1] = 0
                angle[cond2] = np.pi/2

                state_loss = torch.nn.functional.l1_loss(src_state[:,:2], tgt_state[:,:2], reduction='none').sum(-1).mean()
                tgt_extent = torch.cat((x_radii.unsqueeze(-1), y_radii.unsqueeze(-1), angle.unsqueeze(-1)),axis=-1)
                extent_loss = torch.nn.functional.l1_loss(src_state[:,2:], tgt_extent)
                state_loss = state_loss + extent_loss

            else:
                state_loss = compute_wasserstein(src_state, tgt_state)

        src_logits = src['logits'].squeeze(-1).flatten()
        tgt_probs = tgt_probs.flatten()
        weight = (tgt_probs.numel()-tgt_probs.sum()) / tgt_probs.sum() if tgt_probs.sum() > 0 else None
        weight = weight if weight > 0 else None
        logit_loss = torch.nn.functional.binary_cross_entropy_with_logits(src_logits, tgt_probs, pos_weight=weight)

        return {'state': state_loss, 'logits': logit_loss}
        

class DataAssociationLoss(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.device = torch.device(params.training.device)
        self.params = params
        self.sim = torch.nn.CosineSimilarity()
        # Use BCE or CE? i.e. allow one measurement to be from one or mutliple objects
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.cosine_loss = torch.nn.CosineEmbeddingLoss(margin=0.5)
        self.to(self.device)

    def forward(self, first, second, missed_variable):
        first_embed = first['embed']
        first_ids = first['ids']

        second_embed = second['embed']
        second_ids = second['ids']

        bs = len(first_embed)

        loss = {}
        if self.params.loss.cosine_loss:
            loss['cosine'] = 0
        if self.params.loss.binary_cross_entropy_loss:
            loss['binary_cross_entropy'] = 0
        if self.params.loss.cross_entropy_loss:
            loss['cross_entropy'] = 0
        
        aff_matrix = []
        for i in range(bs):
            # filter out all false measurements
            true_objects = first_ids[i] != -1
            x = first_embed[i][true_objects]
            y = second_embed[i]
            if x.shape[0] == 0:
                continue

            # create target ids
            id_y = second_ids[i].cpu().numpy()
            id_x = first_ids[i][true_objects].cpu().numpy()
            missed_detection_embedding_idx = len(id_y)
            target = Tensor([np.where(id_y == s)[0][0] if (s in id_y) else missed_detection_embedding_idx for s in id_x]).to(self.device)

            if self.params.loss.cosine_loss:
                for (t, t_id) in zip(x,target):
                    cosine_target = -torch.ones((y.shape[0])).to(missed_variable.device)  
                    if t_id.long() != missed_detection_embedding_idx:          
                        cosine_target[t_id.long()] = 1

                    loss['cosine'] = loss['cosine'] + self.cosine_loss(t.unsqueeze(0), y, cosine_target)

            # reshape to fit torch.nn.CosineSimilarity
            x = x.unsqueeze(dim=2)
            y = y.unsqueeze(dim=0).permute(0,2,1)
            # compute affinity matrix
            aff = self.sim(x, y)
            delta = torch.ones((x.shape[0],1)).to(missed_variable.device)*missed_variable
            aff = torch.cat((aff, delta), dim=1)
            aff_matrix.append(aff)

            # binaryCrossEntropy
            if self.params.loss.binary_cross_entropy_loss:
                target_one_hot = torch.nn.functional.one_hot(target.to(torch.int64), missed_detection_embedding_idx+1).type_as(aff)
                loss['binary_cross_entropy'] = loss['binary_cross_entropy'] + self.bce_loss(input=aff, target=target_one_hot)
            
            # crossEntropyLoss
            if self.params.loss.cross_entropy_loss:
                loss['cross_entropy'] = loss['cross_entropy'] + self.ce_loss(input=aff, target=target.long())
                
            loss = dict([(k, v/bs) for k,v in loss.items()])
        return loss, aff_matrix


class DhnLoss(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.order = params.loss.order
        self.cutoff_distance = params.loss.cutoff_distance
        self.alpha = params.loss.alpha
        self.miss_cost = self.cutoff_distance ** self.order
        self.device = torch.device(params.training.device)
        self.distance_metric_matching = params.loss.distance_metric_matching
        self.distance_metric_loss = params.loss.distance_metric_loss
        self.delta_value = params.loss.delta_value

        assert self.delta_value is not None, "'loss.delta_value' has to be set if loss.type = 'dhn'. It should specify the delta value used in row/col softmax"
        assert self.distance_metric_matching is not None, "'loss.distance_metric_matching' has to be set if loss.type = 'dhn'. It should specify what distance metric was used when training the DHN"
        assert self.distance_metric_loss is not None, "'loss.distance_metric_loss' has to be set if loss.type = 'dhn'. It should specify what distance metric is to be used for loss computation"
        assert params.loss.saved_weights_file is not None, "'loss.saved_weights' has to be set if loss.type = 'dhn'. It should hold filepath to saved dhn-weights"

        is_cuda = params.training.device == 'cuda'
        self.matcher = Munkrs(  element_dim=1,
                                hidden_dim=256,
                                target_size=1,
                                bidirectional=True,
                                minibatch=params.training.batch_size,
                                is_cuda=is_cuda,
                                is_train=False,
                                sigmoid=True,
                                trainable_delta=False)

        try:
            weights = torch.load(params.loss.saved_weights_file, map_location=torch.device(params.training.device))
            self.matcher.load_state_dict(weights)
            # freeze all weights
            for param in self.matcher.parameters():
                param.requires_grad = False

        except FileNotFoundError:
            print(f'Path specified to load dhn weights from does not exist: {params.loss.saved_weights_file}')
            exit()

    def compute_loss(self, association_matrix, distance_matrix, logits_tensor, scale=100.0):
        """
        association_matrix: num_examples_minibatch x num_queries x num_targets_in_minibatch
        distance_matrix:    num_examples_minibatch x num_queries x num_targets_in_minibatch  
        """
        nm, nq, nt = association_matrix.shape

        row_clutter = (torch.ones(nm, nq, 1) * self.delta_value).to(self.device)
        col_clutter = (torch.ones(nm, 1, nt) * self.delta_value).to(self.device)
        row_softmax = F.softmax(torch.cat([association_matrix, row_clutter], dim=2)*scale,  dim=2)
        col_softmax = F.softmax(torch.cat([association_matrix * logits_tensor.sigmoid(), col_clutter], dim=1)*scale,  dim=1)

        fp = torch.sum(row_softmax[:, :, -1].unsqueeze(2) * logits_tensor.sigmoid()) * self.miss_cost/self.alpha
        fn = torch.sum(col_softmax[:, -1, :]) * self.miss_cost/self.alpha
        
        amax = torch.argmax(col_softmax, dim=1)
        ha_mask = torch.zeros_like(col_softmax).detach()
        for batch in range(amax.shape[0]):
            for width in range(ha_mask.shape[2]):
                ha_mask[batch, amax[batch, width], width] = 1.0
        ha_mask = ha_mask[0:, 0:-1, 0:]

        loc = torch.sum(distance_matrix * ha_mask)

        return {'fp': fp, 'fn': fn, 'loc': loc}, ha_mask, torch.argmax(col_softmax[0:, 0:-1, 0:], dim=1)

    def forward(self,outputs, labels, loss_type=None, existance_threshold=None):
        bs = len(labels)
        decomposition = {k:0 for k in ['fp', 'fn', 'loc']}
        indicies = []
        for i_bs in range(bs):
            target = labels[i_bs]
            state_tensor = outputs['state'][i_bs].unsqueeze(0)
            logits_tensor = outputs['logits'][i_bs].unsqueeze(0)

            if 'aux_outputs' in outputs:
                for i, aux_outputs in enumerate(outputs['aux_outputs']): 
                    state_tensor = torch.cat((state_tensor, aux_outputs['state'][i_bs].unsqueeze(0)), dim=0)
                    logits_tensor = torch.cat((logits_tensor, aux_outputs['logits'][i_bs].unsqueeze(0)), dim=0)

            target = target.repeat(state_tensor.shape[0],1,1)

            src = {'state': state_tensor, 'logits': logits_tensor}
            # Calculate the distance-metric used for matching
            D = calculate_distance_matrix(src, target, self.distance_metric_matching, miss_cost=self.miss_cost/self.alpha)
            Dt = torch.stack(D)
            At = self.matcher(Dt)
            # Calculate the distance-matrix used for loss-computation
            cost_matrix = calculate_distance_matrix(src, target, self.distance_metric_loss, miss_cost=self.miss_cost/self.alpha)
            cost_matrix = torch.stack(cost_matrix)
            # calculate the decomposite loss and association_matrix
            dec, association_matrix, amax = self.compute_loss(At, cost_matrix, logits_tensor)

            # use matching from original output = first in list
            row_idx = amax[0]
            col_idx = torch.as_tensor(range(len(amax[0])), dtype=torch.int64).to(self.device)
            # only those that are not false negative is considered to be a match
            valid = row_idx != self.params.arch.num_queries
            row_idx = row_idx[valid]
            col_idx = col_idx[valid]

            indicies.append((row_idx, col_idx))
            for k in decomposition:
                decomposition[k] = decomposition[k] + dec[k]
            
        return {k:v/bs for k,v in decomposition.items()}, indicies


def compute_pairwise_crossentropy(predictions, targets):
    predicted_states = predictions['states']
    predicted_state_covariances = predictions['state_covariances']

    # TODO: this needs to be updated. 'state_covariances' is now always the variances, so when calling Normal we should
    #  take sqrt()

    # take the square root
    # Check if predicted state covariances are diagonal or entire matrices
    # if len(predicted_state_covariances.shape) == 2:
    #     distribution = Normal
    # elif len(predicted_state_covariances.shape) == 3:
    #     distribution = MultivariateNormal
    # else:
    #     raise NotImplementedError
    #
    # n_predictions, d_predictions = predicted_states.shape
    # n_targets = targets.shape[0]
    # cost = torch.zeros((n_predictions, n_targets))
    #
    # for i in range(n_predictions):
    #     for j in range(n_targets):
    #         predicted_dist = distribution(predicted_states[i], predicted_state_covariances[i])
    #         cost[i, j] = - predicted_dist.log_prob(targets[j]).sum()
    #
    # return cost
