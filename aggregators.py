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
