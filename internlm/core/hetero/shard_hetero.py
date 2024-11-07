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

import os


def partition_uniform_hetero(num_items: int, pipeline_parallel_size: int, num_chunks: int):
    hetero_pp_stages = os.environ.get("HETERO_PP_STAGES", "")
    if hetero_pp_stages == "":
        per_stage = num_items // pipeline_parallel_size
        split_hetero_pp_stages = [per_stage for _ in range(pipeline_parallel_size)]
    else:
        split_hetero_pp_stages = [int(n) for n in hetero_pp_stages.split(",")]
    assert (len(split_hetero_pp_stages) == pipeline_parallel_size), "Num of hetero pp stages should equal to pp size."
    assert (
            num_items % num_chunks == 0
    ), "Layer length should be divided by the number of chunks, otherwise parameter method is recommended"

    parts = [[] for _ in range(pipeline_parallel_size)]
    partition_items = num_items // num_chunks
    assert (sum(split_hetero_pp_stages) == partition_items), \
        (f"Sum up hetero pp stages [{sum(split_hetero_pp_stages)}] should equal to partition_items [{partition_items}] "
         f"(partition_items = num_items // num_chunks).")
    for idx in range(num_chunks):
        base_idx = idx * partition_items
        for p in range(pipeline_parallel_size):
            st = base_idx
            base_idx += split_hetero_pp_stages[p]
            parts[p].append((st, base_idx))

    indexes = []
    for _parts in parts:
        for s, e in _parts:
            indexes.extend(list(range(s, e)))
    assert len(indexes) == len(set(indexes)), indexes  # should have no duplicates
    assert set(indexes) == set(list(range(num_items))), (indexes, num_items)  # should have the same indexes as expected

    return parts
