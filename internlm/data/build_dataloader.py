# Copyright (c) InternLM. All rights reserved.
import subprocess
from functools import partial

import torch.distributed as dist
from torch.utils.data import ConcatDataset, DataLoader

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.data.megatron.batch_sampler import MegatronBatchSampler
from internlm.data.megatron.collaters import megatron_collate_fn
from internlm.data.megatron.dataset import build_megatron_dataset
from internlm.data.mocked.batch_sampler import MockedSequentialBatchSampler
from internlm.data.mocked.dataset import MockedDataset
from internlm.data.streaming.batch_sampler import StreamingStaticBatchSampler
from internlm.data.streaming.collaters import streaming_packed_collate_fn
from internlm.data.streaming.dataset import (
    StreamingDataset,
    StreamingDatasetPackSampleIntoOneWithCut,
    StreamingDatasetPackSampleWithPad,
)
from internlm.data.tokenized.batch_sampler import (
    StaticBatchSampler,
    get_dpsampler_dataloader,
)
from internlm.data.tokenized.collaters import (
    generation_collate_fn,
    jsonl_ds_collate_fn,
    packed_collate_fn,
)
from internlm.data.tokenized.dataset import get_dataset_dict
from internlm.data.tokenized.dummy_dataset import RandomDataset
from internlm.data.tokenized.dummy_dataset_multimodal import RandomDatasetMultimodal
from internlm.data.tokenized.packed_dataset import (
    PackedDatasetWithCut,
    PackedDatasetWithoutCuSeqlen,
    PackedDatasetWithPadForMultimodal,
    get_packed_dataset_without_short_length,
)
from internlm.data.utils import get_dataset_type_ids_map
from internlm.utils.logger import get_logger
from internlm.utils.utils import DataType

# global llm logger
logger = get_logger(__file__)


def get_tokenized_train_loader_items(data_cfg):
    """Get the training data loader for tokenized dataset."""
    if data_cfg.get("train_folder", None) is None:
        if data_cfg.get("is_multimodal", False):
            image_token_size = int(data_cfg.image_size // data_cfg.patch_size) ** 2
            train_ds = RandomDatasetMultimodal(
                num_samples=gpc.get_world_size(ParallelMode.DATA) * 500,
                max_len=data_cfg.seq_len,
                image_size=data_cfg.image_size,
                image_token_size=image_token_size,
            )
            train_ds = PackedDatasetWithPadForMultimodal(
                train_ds, max_length_per_sample=data_cfg.seq_len, packed_length=data_cfg.packed_length
            )
        else:
            train_ds = RandomDataset(
                num_samples=gpc.get_world_size(ParallelMode.DATA) * 500,
                max_len=data_cfg.seq_len,
                fixed_seqlen=data_cfg.fixed_random_dataset_seqlen,
            )

            if data_cfg.pack_sample_into_one:
                train_ds = PackedDatasetWithoutCuSeqlen(
                    train_ds, max_length_per_sample=data_cfg.seq_len, packed_length=data_cfg.packed_length
                )
            else:
                train_ds = PackedDatasetWithCut(
                    train_ds, max_length_per_sample=data_cfg.seq_len, packed_length=data_cfg.packed_length
                )
    else:
        train_ds = get_packed_dataset_without_short_length(
            folder=data_cfg.train_folder,
            packed_length=data_cfg.packed_length,
            max_length_per_sample=data_cfg.seq_len,
            show_progress=dist.get_rank() == 0,
            min_length=data_cfg.get("min_length", 0),
            min_length_dict=data_cfg.get("min_length_dict", None),
            pack_sample_into_one=data_cfg.get("pack_sample_into_one", False),
        )

    train_sampler = StaticBatchSampler(
        train_ds.datasets if isinstance(train_ds, ConcatDataset) else [train_ds],
        batch_size=data_cfg.micro_num,
        rampup_batch_size=data_cfg.rampup_batch_size,
        micro_bsz=data_cfg.micro_bsz,
        seed=data_cfg.get("seed", 1024),
        drop_last=True,
        data_rank=gpc.get_local_rank(ParallelMode.DATA),
        data_world_size=gpc.get_world_size(ParallelMode.DATA),
    )
    train_collate_fn = partial(packed_collate_fn, packed_length=data_cfg.packed_length)

    return train_ds, train_sampler, train_collate_fn


def get_tokenized_valid_loader_items(data_cfg):
    """Get the validation data loader for tokenized dataset."""
    if not data_cfg.valid_folder:
        if data_cfg.get("is_multimodal", False):
            image_token_size = int(data_cfg.image_size // data_cfg.patch_size) ** 2
            valid_ds = RandomDatasetMultimodal(
                num_samples=gpc.get_world_size(ParallelMode.DATA) * 500,
                max_len=data_cfg.seq_len,
                image_size=data_cfg.image_size,
                image_token_size=image_token_size,
            )
        else:
            valid_ds = RandomDataset(
                num_samples=gpc.get_world_size(ParallelMode.DATA) * 500,
                max_len=data_cfg.seq_len,
                fixed_seqlen=data_cfg.fixed_random_dataset_seqlen,
            )
    else:
        valid_ds = get_dataset_dict(folder=data_cfg.valid_folder, split="")

    if not isinstance(valid_ds, dict):
        valid_ds = {"val": valid_ds}

    valid_collate_fn = partial(jsonl_ds_collate_fn, max_length_per_sample=data_cfg.seq_len)

    return valid_ds, valid_collate_fn


def get_streaming_train_loader_items(data_cfg):
    train_ds = StreamingDataset(
        train_folder=data_cfg.train_folder,
        tokenizer_path=data_cfg.tokenizer_path,
        model_max_length=data_cfg.seq_len,
        image_folder=data_cfg.get("image_folder", None),
        content_name=data_cfg.get("content_name", "text"),
        subset_name=data_cfg.get("subset_name", None),
    )
    if data_cfg.get("pack_sample_into_one", False):
        train_ds = StreamingDatasetPackSampleIntoOneWithCut(
            dataset=train_ds,
            seq_len=data_cfg.seq_len,
            micro_bsz=data_cfg.micro_bsz,
        )
    else:
        train_ds = StreamingDatasetPackSampleWithPad(
            dataset=train_ds,
            seq_len=data_cfg.seq_len,
            micro_bsz=data_cfg.micro_bsz,
        )
    train_sampler = StreamingStaticBatchSampler(
        batch_size=data_cfg.micro_num, rampup_batch_size=data_cfg.rampup_batch_size
    )
    return train_ds, train_sampler, streaming_packed_collate_fn


def get_megatron_train_loader_items(data_cfg):
    try:
        from internlm.data.megatron import helpers  # noqa # pylint: disable=W0611
    except ImportError:
        if gpc.is_rank_for_log():
            subprocess.run(  # noqa # pylint: disable=W1510
                [
                    "g++",
                    "-O3",
                    "-shared",
                    "-std=c++11",
                    "-fPIC",
                    "-fdiagnostics-color",
                    "-I$(python3-config --includes)",
                    "-I$(python3 -m pybind11 --includes)",
                    "internlm/data/megatron/helpers.cpp",
                    "-o",
                    "internlm/data/megatron/helpers.so",
                ]
            )
    train_ds = build_megatron_dataset(
        data_prefix=data_cfg.train_folder,
        data_impl=data_cfg.get("data_impl", "infer"),
        splits_string="1.0, 0.0, 0.0",
        train_valid_test_num_samples=[9600000, 0, 0],
        seq_len=data_cfg.seq_len,
        seed=data_cfg.get("seed", 1024),
        skip_warmup=True,
    )

    train_sampler = MegatronBatchSampler(
        total_samples=len(train_ds),
        consumed_samples=0,
        batch_size=data_cfg.micro_num * data_cfg.micro_bsz,
        drop_last=True,
    )

    train_collate_fn = partial(
        megatron_collate_fn, micro_num=data_cfg.micro_num, micro_bsz=data_cfg.micro_bsz, seq_len=data_cfg.seq_len
    )

    return train_ds, train_sampler, train_collate_fn


def get_mock_train_loader_items(data_cfg):
    assert data_cfg.get(
        "pack_sample_into_one", False
    ), "mocked dataloader curently only supports pack_sample_into_one=True"
    train_ds = MockedDataset(
        train_folder=data_cfg.train_folder,
        micro_bsz=data_cfg.micro_bsz,
        micro_num=data_cfg.micro_num,
        seq_len=data_cfg.seq_len,
    )
    train_sampler = MockedSequentialBatchSampler(train_ds, data_cfg.micro_num)
    train_collate_fn = partial(packed_collate_fn, packed_length=data_cfg.seq_len * data_cfg.micro_bsz)
    return train_ds, train_sampler, train_collate_fn


def build_train_loader_with_data_type():
    """
    Build and return the training data loader based on data type.

    Returns: A tuple of (train_dl, dataset_types).
    """
    data_cfg = gpc.config.data

    if data_cfg.type == DataType.tokenized.name:
        train_ds, train_sampler, train_collate_fn = get_tokenized_train_loader_items(data_cfg)
        train_folder = data_cfg.get("train_folder", None)
        dataset_types = list(get_dataset_type_ids_map(train_folder).keys()) if train_folder else ["en", "cn", "code"]
    elif data_cfg.type == DataType.streaming.name:
        train_ds, train_sampler, train_collate_fn = get_streaming_train_loader_items(data_cfg)
        # TODO: support more dataset_types
        dataset_types = ["en"]
    elif data_cfg.type == DataType.megatron.name:
        train_ds, train_sampler, train_collate_fn = get_megatron_train_loader_items(data_cfg)
        # TODO: support more dataset_types
        dataset_types = ["en"]
    elif data_cfg.type == DataType.mocked.name:
        train_ds, train_sampler, train_collate_fn = get_mock_train_loader_items(data_cfg)
        # TODO: support more dataset_types
        dataset_types = ["en"]
    else:
        raise ValueError(f"dataset type {data_cfg.type} is not supported")

    # Create the training data loader
    train_dl = DataLoader(
        dataset=train_ds,
        batch_sampler=train_sampler,
        num_workers=data_cfg.get("num_worker", 4),
        pin_memory=True,
        collate_fn=train_collate_fn,
        persistent_workers=data_cfg.get("num_worker", 4) > 0,
    )

    return train_dl, dataset_types


def build_valid_loader_with_data_type():
    """Generate and return the validation data loader based on data type."""

    data_cfg = gpc.config.data

    # TODO: For validation, currenlt we only support dummy dataset for streaming/megatron/mocked DataType.
    if data_cfg.type in [
        DataType.tokenized.name,
        DataType.streaming.name,
        DataType.megatron.name,
        DataType.mocked.name,
    ]:
        valid_ds, valid_collate_fn = get_tokenized_valid_loader_items(data_cfg)
    else:
        raise ValueError(f"dataset type {data_cfg.type} is not supported")

    if valid_ds is None:
        return None

    val_dls = {}
    for val_name, ds in valid_ds.items():
        # making the batch_size of validate larger can speed up the evaluation, but it should not be too large,
        # otherwise too much data may be dropped
        batch_size = min(
            data_cfg.valid_micro_num * data_cfg.micro_bsz, len(ds) // gpc.get_world_size(ParallelMode.DATA)
        )
        batch_size = batch_size // data_cfg.micro_bsz * data_cfg.micro_bsz

        if batch_size == 0 and gpc.is_rank_for_log():
            logger.info(f"skip validate {val_name}.")
            continue

        val_dls[val_name] = get_dpsampler_dataloader(
            ds,
            shuffle=False,
            num_workers=data_cfg.get("num_worker", 0),
            batch_size=batch_size,
            collate_fn=valid_collate_fn,
            drop_last=True,
        )  # drop_last=True, otherwise it may cause problems in the last batch

        if gpc.is_rank_for_log():
            logger.info(
                f"load validation dataset {val_name} with valid batch size {str(batch_size)} and "
                f"samples {str(len(val_dls[val_name]))}."
            )

    return val_dls


def build_generation_loader_with_data_type(data_cfg, generation_cfg):
    """Generate and return the validation data loader based on data type."""

    if data_cfg.type == DataType.tokenized.name:
        gene_ds, _ = get_tokenized_valid_loader_items(data_cfg)
    else:
        raise ValueError(f"dataset type {data_cfg.type} is not supported")

    if gene_ds is None:
        return None

    gene_dls = {}
    for gene_name, ds in gene_ds.items():
        # making the batch_size of validate larger can speed up the evaluation, but it should not be too large,
        # otherwise too much data may be dropped
        batch_size = min(
            data_cfg.valid_micro_num * data_cfg.micro_bsz, len(ds) // gpc.get_world_size(ParallelMode.DATA)
        )
        batch_size = batch_size // data_cfg.micro_bsz * data_cfg.micro_bsz
        if generation_cfg.batch_size:
            batch_size = generation_cfg.batch_size

        if batch_size == 0 and gpc.is_rank_for_log():
            logger.info(f"skip validate {gene_name}.")
            continue

        gene_dls[gene_name] = get_dpsampler_dataloader(
            ds,
            shuffle=False,
            num_workers=data_cfg.get("num_worker", 0),
            batch_size=batch_size,
            collate_fn=partial(generation_collate_fn, pad_id=generation_cfg.pad_id),
        )

        if gpc.is_rank_for_log():
            logger.info(
                f"load validation dataset {gene_name} with valid batch size {str(batch_size)} and "
                f"samples {str(len(gene_dls[gene_name]))}."
            )

    return gene_dls
