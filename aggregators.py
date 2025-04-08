from typing import List

import tensorly as tl
import torch
from flex.model import FlexModel
from flex.pool import aggregate_weights
from flexclash.pool import multikrum


@aggregate_weights
def layerwise_krum(weights: List[List[tl.tensor]], f: int = 1):
    return [
        multikrum.__wrapped__([[weights[j][i]] for j in range(len(weights))], m=1, f=f)[
            0
        ]
        for i in range(len(weights[0]))
    ]


