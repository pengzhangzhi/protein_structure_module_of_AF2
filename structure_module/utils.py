from functools import partial
import torch
import torch.nn as nn
from typing import Tuple, List, Callable, Any, Dict, Sequence, Optional,callable


def dict_multimap(fn:callable, dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    first = dicts[0]
    new_dict = {}
    for k, v in first.items():
        all_v = [d[k] for d in dicts]
        if type(v) is dict:
            new_dict[k] = dict_multimap(fn, all_v)
        else:
            new_dict[k] = fn(all_v)

    return new_dict


def exists(x: Any) -> bool:
    return x is not None

def default(x: Any) -> Any:
    return x if exists(x) else None
