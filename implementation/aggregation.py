from typing import Dict, List, Optional
import torch

def decentralized_fedavg(
    local_weights: Dict[str, torch.Tensor],
    neighbor_weights_list: List[Dict[str, torch.Tensor]],
    local_weight_factor: Optional[float] = None
) -> Dict[str, torch.Tensor]:
    """
    Performs decentralized FedAvg aggregation for a client.
    Aggregates the client's own updated weights with weights received from its neighbors:
    
    θ_i^{(t+1)} = (1 / (|N_i| + 1)) * [ θ_i^{(t)} + \sum_{j \in N_i} θ_j^{(t)} ]

    Args:
        local_weights: state_dict of the client's local model.
        neighbor_weights_list: list of state_dicts from active neighbors in this round.
        local_weight_factor: optional custom weight for the local model (defaults to equal 1 / (N+1)).

    Returns:
        aggregated_weights: new state_dict for the client model.
    """
    num_models = 1 + len(neighbor_weights_list)
    if num_models == 1:
        # No neighbors received; retain local weights
        return {k: v.clone() for k, v in local_weights.items()}

    aggregated = {}

    if local_weight_factor is None:
        # Standard unweighted/equal FedAvg across the local model + neighbors
        inv_total = 1.0 / float(num_models)
        for key, tensor in local_weights.items():
            if tensor.is_floating_point():
                sum_tensor = tensor.clone().float()
                for n_dict in neighbor_weights_list:
                    sum_tensor += n_dict[key].float()
                aggregated[key] = (sum_tensor * inv_total).to(tensor.dtype)
            else:
                # Non-floating point buffers (e.g. integer tracking counters)
                aggregated[key] = tensor.clone()
    else:
        # Weighted version (e.g., higher local retention)
        assert 0.0 < local_weight_factor < 1.0, "local_weight_factor must be between 0 and 1"
        neighbor_factor = (1.0 - local_weight_factor) / len(neighbor_weights_list)
        for key, tensor in local_weights.items():
            if tensor.is_floating_point():
                sum_tensor = tensor.clone().float() * local_weight_factor
                for n_dict in neighbor_weights_list:
                    sum_tensor += n_dict[key].float() * neighbor_factor
                aggregated[key] = sum_tensor.to(tensor.dtype)
            else:
                aggregated[key] = tensor.clone()

    return aggregated
