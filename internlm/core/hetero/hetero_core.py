# Copyright 2024 Shanghai Wuwen Xingqiong Intelligent Technology Co., Ltd., Hu Wenhao
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import wraps
from types import ModuleType
from typing import Dict, Tuple, Callable

import internlm
from internlm.core.hetero import is_hetero

HETERO_PATCH_DICT: Dict[ModuleType, Tuple[str, Callable]] = dict()


def register_hetero_patch(fn):
    global MOD_PATCH_DICT
    mod, name, call = fn()
    MOD_PATCH_DICT[mod] = (name, call)


@register_hetero_patch
def patch_hetero_communicate():
    from internlm.core.scheduler.comm.p2p import _communicate
    from internlm.core.hetero.p2p_gloo import _communicate_by_cpu_gloo
    return internlm.core.scheduler.comm.p2p, _communicate.__name__, _communicate_by_cpu_gloo


@register_hetero_patch
def patch_hetero_partition_uniform():
    from internlm.core.parallel.shard import partition_uniform
    from internlm.core.hetero.shard_hetero import partition_uniform_hetero
    return internlm.core.parallel.shard, partition_uniform.__name__, partition_uniform_hetero


def check_backend_in_args(*args, **kwargs):
    from internlm.accelerator import get_accelerator
    internlm_accelerator = get_accelerator()
    no_backend = True
    comm_backend_name = internlm_accelerator._communication_backend_name

    hit_idx = -1
    if args:
        args_new = list(args)
        for idx, arg in enumerate(args_new):
            if type(arg) == str and arg in (comm_backend_name, "gloo"):
                no_backend = False
            else:
                hit_idx = idx
                break
    if kwargs:
        backend = kwargs.get('backend', None)
        if isinstance(backend, str) and backend in (comm_backend_name, "gloo"):
            no_backend = False

    return no_backend, hit_idx


@register_hetero_patch
def patch_hetero_dist_new_group():
    from internlm.accelerator import get_accelerator
    internlm_accelerator = get_accelerator()

    def wrapper_new_group(fn):

        @wraps(fn)
        def decorated(*args, **kwargs):
            comm_backend_name = internlm_accelerator._communication_backend_name
            no_backend, _ = check_backend_in_args(*args, **kwargs)
            if no_backend:
                kwargs["backend"] = comm_backend_name
            return fn(*args, **kwargs)

        return decorated

    from internlm.core.context.process_group_initializer import dist
    return internlm.core.context.process_group_initializer.dist, dist.new_group.__name__, wrapper_new_group(dist.new_group)


@register_hetero_patch
def patch_hetero_dist_init_process_group():
    from internlm.accelerator import get_accelerator
    internlm_accelerator = get_accelerator()

    def wrapper_init_process_group(fn):

        @wraps(fn)
        def decorated(*args, **kwargs):
            comm_backend_name = internlm_accelerator._communication_backend_name
            global_backend = "gloo" if is_hetero else comm_backend_name
            no_backend, hit_idx = check_backend_in_args(*args, **kwargs)
            args_new = list(args)
            if no_backend:
                kwargs["backend"] = global_backend
            elif hit_idx != -1:
                args_new[hit_idx] = global_backend

            return fn(*args_new, **kwargs)

        return decorated

    from internlm.core.context.process_group_initializer import dist
    return (internlm.core.context.parallel_context.dist,
            dist.init_process_group.__name__,
            wrapper_init_process_group(dist.init_process_group))


if is_hetero:
    for k, v in HETERO_PATCH_DICT.items():
        setattr(k, v[0], v[1])
