from flex.pool import FlexPool
import multiprocessing as mp
import torch
from flex.model import FlexModel
from flex.pool.decorators import collect_clients_weights, aggregate_weights
from typing import Dict, List
from functools import reduce
import re
import tensorly as tl
from flexclash.pool import multikrum


def parallel_pool_map(pool: FlexPool, ammount: int, func, *args, **kwargs):
    ids = list(pool.keys())
    pool_size = len(ids)
    chunk_size = pool_size // ammount
    chunks = [ids[i : i + chunk_size] for i in range(0, pool_size, chunk_size)]
    # trunk-ignore(ruff/B023)
    pools = [pool.clients.select(lambda id, _: id in ids) for ids in chunks]
    executor = mp.Pool(ammount)

    executor.map(lambda i, p: p.map(func, *args, rank=i, **kwargs), enumerate(pools))


group_regexes = [r"features\.\d\.\d\.block\.*"]


def group_by_block_efficient(state_dict: Dict[str, torch.Tensor]):
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

    return group_by_block_efficient(state)


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
