import re
from functools import reduce
from typing import Dict, List

import tensorly as tl
import torch
from flex.model import FlexModel
from flex.pool import aggregate_weights, collect_clients_weights
from flexclash.pool import bulyan, multikrum


@aggregate_weights
def layerwise_krum(weights: List[List[tl.tensor]]):
    return [
        multikrum.__wrapped__([[weights[j][i]] for j in range(len(weights))], m=1)[0]
        for i in range(len(weights[0]))
    ]


@aggregate_weights
def layerwise_bulyan(weights: List[List[tl.tensor]]):
    return [
        bulyan.__wrapped__([[weights[j][i]] for j in range(len(weights))], m=1)[0]
        for i in range(len(weights[0]))
    ]


def _group_by_block_efficient(state_dict: Dict[str, torch.Tensor]):
    group_regexes = [r"features\.\d\.\d\.block\.*"]
    group_regex = re.compile(group_regexes[0])
    groups = {}
    before = []
    after = []
    for name in state_dict:
        match = group_regex.match(name)
        if match is None:
            if len(groups) == 0:
                before.append(state_dict[name])
            else:
                after.append(state_dict[name])
        else:
            group = match.string[match.start() : match.end()]
            if group not in groups:
                groups[group] = []
            groups[group].append(state_dict[name])
    groups["before"] = before
    groups["after"] = after
    return groups


@collect_clients_weights
def get_clients_weights_celeba_blockwise(client_flex_model: FlexModel):
    weight_dict = client_flex_model["model"].state_dict()
    server_dict = client_flex_model["server_model"].state_dict()
    dev = [weight_dict[name] for name in weight_dict][0].get_device()
    dev = "cpu" if dev == -1 else "cuda"
    state = {
        name: (weight_dict[name] - server_dict[name].to(dev)).type(torch.float)
        for name in weight_dict
    }

    return _group_by_block_efficient(state)


@aggregate_weights
def blockwise_krum(weights: List[Dict[str, tl.tensor]]):
    before = [weight["before"] for weight in weights]
    after = [weight["after"] for weight in weights]
    ids = [name for name in weights[0] if name != "before" and name != "after"]

    before: List[tl.tensor] = multikrum.__wrapped__(before, m=1)
    after: List[tl.tensor] = multikrum.__wrapped__(after, m=1)
    the_rest: List[List[tl.tensor]] = [
        multikrum.__wrapped__([weight[name] for weight in weights], m=1) for name in ids
    ]
    the_rest = list(reduce(lambda x, y: x + y, the_rest, []))
    return before + the_rest + after


def _krum_cosine_similarity(
    list_of_updates: List[torch.Tensor], f: int = 1, k: int = 1
) -> torch.Tensor:
    """
    Implements the Krum operator using cosine similarity in PyTorch.

    Args:
        updates (torch.Tensor): Tensor of shape (n_clients, update_size).
        f (int): Maximum number of Byzantine clients to tolerate.
        k (int): The ammount of updates to select

    Returns:
        torch.Tensor: The k indexes of the selected updates of the Krum operator.
    """
    updates = torch.stack(list_of_updates)
    n_clients = updates.size(0)
    # Normalize the updates to unit vectors
    norm_updates = updates / updates.norm(dim=1, keepdim=True)
    # Compute the cosine similarity matrix
    similarities = torch.mm(norm_updates, norm_updates.t())
    # Convert similarities to cosine distances
    distances = 1 - similarities
    # Exclude self-distances by setting the diagonal to infinity
    distances.fill_diagonal_(float("inf"))
    # Sort distances in ascending order
    sorted_distances, _ = distances.sort(dim=1)
    # Sum the smallest n - f - 2 distances for each client
    m = n_clients - f - 2
    scores = sorted_distances[:, :m].sum(dim=1)
    # Select the k client with the minimum score
    min_score_index = torch.topk(scores, k, largest=False).indices
    # Return the selected update
    return min_score_index


@aggregate_weights
def krum_cosine_similarity(list_of_weights: List[List[torch.Tensor]], f: int = 1):
    # Flatten and stack the weights from each model
    flattened_weights = [
        torch.cat([torch.flatten(param) for param in model_weights])
        for model_weights in list_of_weights
    ]

    # Call the original Krum function
    selected_update = list_of_weights[
        _krum_cosine_similarity(flattened_weights, f).item()
    ]

    # Reconstruct the weights into the original parameter shapes
    param_shapes = [param.shape for param in list_of_weights[0]]
    offset = 0
    selected_weights = []
    for shape in param_shapes:
        num_params = int(torch.tensor(shape).prod().item())
        param = selected_update[slice(offset, offset + num_params)].view(shape)
        selected_weights.append(param)
        offset += num_params

    return selected_weights


@aggregate_weights
def krum_cosine_similarity_layerwise(
    list_of_weights: List[List[torch.Tensor]], f: int = 1
):
    selected_update = [
        list_of_weights[
            _krum_cosine_similarity(
                [
                    torch.flatten(list_of_weights[j][i])
                    for j in range(len(list_of_weights))
                ],
                f,
            ).item()
        ][i]
        for i in range(len(list_of_weights[0]))
    ]

    # Reconstruct the weights into the original parameter shapes
    param_shapes = [param.shape for param in list_of_weights[0]]
    selected_weights = []
    for param, shape in zip(selected_update, param_shapes):
        assert type(param) == torch.Tensor, f"param is not a tensor: {param}"
        param = param.view(shape)
        selected_weights.append(param)

    return selected_weights


def _bulyan_cosine_similarity(
    list_of_updates: List[torch.Tensor], f: int = 1, m: int = 5
) -> torch.Tensor:
    """
    Implements the Bulyan operator using cosine similarity Krum in PyTorch.

    Args:
        list_of_updates (List[torch.Tensor]): List of client updates.
        f (int): Maximum number of Byzantine clients to tolerate.

    Returns:
        torch.Tensor: The aggregated update computed by Bulyan.
    """
    updates = list_of_updates.copy()
    n = len(updates)
    if n < 4 * f + 3:
        raise ValueError(
            "Number of clients is too small for the given number of Byzantine clients."
        )

    k = m
    selected_updates_indexes = _krum_cosine_similarity(updates, f, k)
    selected_updates = [updates[i] for i in selected_updates_indexes]

    # Stack the selected updates
    stacked_updates = torch.stack(
        selected_updates
    )  # Shape: (num_selected, update_size)

    # Compute the Bulyan aggregation
    bulyan_update = torch.zeros_like(stacked_updates[0])
    num_coordinates = stacked_updates.size(1)

    for i in range(num_coordinates):
        # Collect the i-th coordinate from all selected updates
        coordinate_values = stacked_updates[:, i]
        # Sort the coordinate values
        sorted_vals, _ = torch.sort(coordinate_values)
        # Remove the smallest and largest 'f' values
        trimmed_vals = sorted_vals[f:-f]
        # Compute the mean of the remaining values
        bulyan_update[i] = trimmed_vals.mean()

    return bulyan_update


@aggregate_weights
def bulyan_cosine_similarity(
    list_of_weights: List[List[torch.Tensor]], f: int = 1, m: int = 5
) -> List[torch.Tensor]:
    """
    Aggregates model weights using the Bulyan operator based on cosine similarity Krum.

    Args:
        list_of_weights (List[List[torch.Tensor]]): List of model weights from clients.
        f (int): Maximum number of Byzantine clients to tolerate.

    Returns:
        List[torch.Tensor]: Aggregated model weights.
    """
    # Flatten the weights from each model
    flattened_weights = [
        torch.cat([param.view(-1) for param in model_weights])
        for model_weights in list_of_weights
    ]

    # Call the Bulyan function
    aggregated_update = _bulyan_cosine_similarity(flattened_weights, f, m)

    # Reconstruct the weights into the original parameter shapes
    param_shapes = [param.shape for param in list_of_weights[0]]
    offset = 0
    aggregated_weights = []
    for shape in param_shapes:
        num_params = int(torch.tensor(shape).prod().item())
        param = aggregated_update[offset : offset + num_params].view(shape)
        aggregated_weights.append(param)
        offset += num_params

    return aggregated_weights


@aggregate_weights
def bulyan_cosine_similarity_layerwise(
    list_of_updates: List[List[torch.Tensor]], f: int = 1, m: int = 5
) -> List[torch.Tensor]:
    """
    Implements the layer-wise Bulyan operator using cosine similarity Krum.

    Args:
        list_of_updates (List[List[torch.Tensor]]): List of client updates, each is a list of tensors per layer.
        f (int): Maximum number of Byzantine clients to tolerate.

    Returns:
        List[torch.Tensor]: The aggregated update for each layer.
    """
    n = len(list_of_updates)
    if n < 4 * f + 3:
        raise ValueError(
            "Number of clients is too small for the given number of Byzantine clients."
        )

    num_layers = len(list_of_updates[0])  # Number of layers
    aggregated_update = []

    for layer_idx in range(num_layers):
        # Collect the layer parameters from all clients
        layer_updates = [
            client_updates[layer_idx].flatten() for client_updates in list_of_updates
        ]
        # Apply Bulyan using cosine similarity Krum to the layer
        aggregated_layer = _bulyan_cosine_similarity(layer_updates, f, m)
        # Reshape to original shape
        aggregated_layer = aggregated_layer.view(list_of_updates[0][layer_idx].shape)
        aggregated_update.append(aggregated_layer)

    return aggregated_update


def clip_by_norm(aggregator_func):
    @aggregate_weights
    def _func(list_of_weights: List[List[torch.Tensor]], *args, **kwargs):
        # Transpose list_of_weights to group weights by parameter
        list_of_weights_per_param = list(zip(*list_of_weights))  # Length: n_params

        # Compute norms of each parameter across all clients
        norms_per_param = []
        for param_weights in list_of_weights_per_param:
            param_weights_tensor = torch.stack(param_weights)  # Shape: (n_clients, ...)
            param_weights_flat = param_weights_tensor.view(
                param_weights_tensor.size(0), -1
            )
            norms = torch.norm(param_weights_flat, dim=1)  # Shape: (n_clients,)
            norms_per_param.append(norms)

        # Stack norms to get a tensor of shape (n_params, n_clients)
        norms_tensor = torch.stack(norms_per_param)  # Shape: (n_params, n_clients)

        # Compute median norms for each parameter across clients
        median_norms = torch.median(norms_tensor, dim=1).values  # Shape: (n_params,)

        # Call the aggregator function to get the selected update
        selected_update = aggregator_func.__wrapped__(list_of_weights, *args, **kwargs)

        # Compute norms of the selected update's parameters
        selected_norms = [torch.norm(param) for param in selected_update]
        selected_norms_tensor = torch.stack(selected_norms)  # Shape: (n_params,)

        # Compute scaling factors to clip the norms
        scaling_factors = torch.where(
            selected_norms_tensor > median_norms,
            median_norms / selected_norms_tensor,
            torch.ones_like(selected_norms_tensor),
        )

        # Apply scaling factors to clip the selected update's parameters
        clipped_update = [
            param * scale for param, scale in zip(selected_update, scaling_factors)
        ]

        return clipped_update

    return _func
