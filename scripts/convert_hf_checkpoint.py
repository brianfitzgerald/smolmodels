import gc
import json
import sys
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import fire
import os

import torch
from lightning.fabric.utilities.load import _NotYetLoadedTensor as NotYetLoadedTensor

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from model import Config, ModelFamily
from utils import incremental_save, lazy_load


def copy_weights_stablelm(
    config: Config,
    qkv_weights: Dict[int, List[Optional[NotYetLoadedTensor]]],
    state_dict: Dict[str, torch.Tensor],
    hf_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
    dtype: Optional[torch.dtype] = None,
) -> None:
    weight_map = {
        "model.embed_tokens.weight": "transformer.wte.weight",
        "model.layers.{}.input_layernorm.bias": "transformer.h.{}.norm_1.bias",
        "model.layers.{}.input_layernorm.weight": "transformer.h.{}.norm_1.weight",
        "model.layers.{}.attention.query_key_value.bias": "transformer.h.{}.attn.attn.bias",
        "model.layers.{}.attention.query_key_value.weight": "transformer.h.{}.attn.attn.weight",
        "model.layers.{}.attention.dense.bias": "transformer.h.{}.attn.proj.bias",
        "model.layers.{}.attention.dense.weight": "transformer.h.{}.attn.proj.weight",
        "model.layers.{}.attention.rotary_emb.inv_freq": None,
        "model.layers.{}.self_attn.q_proj.weight": None,
        "model.layers.{}.self_attn.k_proj.weight": None,
        "model.layers.{}.self_attn.v_proj.weight": None,
        "model.layers.{}.self_attn.o_proj.weight": "transformer.h.{}.attn.proj.weight",
        "model.layers.{}.attention.bias": None,
        "model.layers.{}.attention.masked_bias": None,
        "model.layers.{}.post_attention_layernorm.bias": "transformer.h.{}.norm_2.bias",
        "model.layers.{}.post_attention_layernorm.weight": "transformer.h.{}.norm_2.weight",
        "model.layers.{}.mlp.down_proj.weight": "transformer.h.{}.mlp.down_proj.weight",
        "model.layers.{}.mlp.up_proj.weight": "transformer.h.{}.mlp.up_proj.weight",
        "model.layers.{}.mlp.gate_proj.weight": "transformer.h.{}.mlp.gate_proj.weight",
        "model.norm.bias": "norm.bias",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "lm_head.weight",
    }

    for name, param in hf_weights.items():
        if "model.layers" in name:
            from_name, number = layer_template(name, 2)
            to_name = weight_map[from_name]
            qkv = qkv_weights.setdefault(number, [None, None, None])
            if "q_proj" in name:
                qkv[0] = param  # type: ignore
            elif "k_proj" in name:
                qkv[1] = param  # type: ignore
            elif "v_proj" in name:
                qkv[2] = param  # type: ignore
            if to_name is None:
                continue
            to_name = to_name.format(number)
        else:
            to_name = weight_map[name]
        param = load_param(param, name, dtype)
        if saver is not None:
            param = saver.store_early(param)
        state_dict[to_name] = param  # type: ignore
        for i, (q, k, v) in list(qkv_weights.items()):
            if q is None or k is None or v is None:
                # split across different .bin files
                continue
            q = load_param(q, f"layer {i} q", dtype)
            k = load_param(k, f"layer {i} k", dtype)
            v = load_param(v, f"layer {i} v", dtype)
            assert config.n_query_groups
            q_per_kv = config.n_head // config.n_query_groups
            qs = torch.split(q, config.head_size * q_per_kv)
            ks = torch.split(k, config.head_size)
            vs = torch.split(v, config.head_size)
            cycled = [t for group in zip(qs, ks, vs) for t in group]
            qkv = torch.cat(cycled)
            state_dict[f"transformer.h.{i}.attn.attn.weight"] = qkv
            del qkv_weights[i]


def copy_weights_hf_llama(
    config: Config,
    qkv_weights: Dict[int, List[Optional[NotYetLoadedTensor]]],
    state_dict: Dict[str, torch.Tensor],
    hf_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
    dtype: Optional[torch.dtype] = None,
) -> None:
    weight_map = {
        "model.embed_tokens.weight": "transformer.wte.weight",
        "model.layers.{}.input_layernorm.weight": "transformer.h.{}.norm_1.weight",
        "model.layers.{}.self_attn.q_proj.weight": None,
        "model.layers.{}.self_attn.k_proj.weight": None,
        "model.layers.{}.self_attn.v_proj.weight": None,
        "model.layers.{}.self_attn.o_proj.weight": "transformer.h.{}.attn.proj.weight",
        "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
        "model.layers.{}.post_attention_layernorm.weight": "transformer.h.{}.norm_2.weight",
        "model.layers.{}.mlp.gate_proj.weight": "transformer.h.{}.mlp.fc_1.weight",
        "model.layers.{}.mlp.up_proj.weight": "transformer.h.{}.mlp.fc_2.weight",
        "model.layers.{}.mlp.down_proj.weight": "transformer.h.{}.mlp.proj.weight",
        "model.norm.weight": "transformer.norm.weight",
        "lm_head.weight": "lm_head.weight",
    }

    for name, param in hf_weights.items():
        if "model.layers" in name:
            from_name, number = layer_template(name, 2)
            qkv = qkv_weights.setdefault(number, [None, None, None])
            if "q_proj" in name:
                qkv[0] = param  # type: ignore
            elif "k_proj" in name:
                qkv[1] = param  # type: ignore
            elif "v_proj" in name:
                qkv[2] = param  # type: ignore
            to_name = weight_map[from_name]
            if to_name is None:
                continue
            to_name = to_name.format(number)
        else:
            to_name = weight_map[name]
        param = load_param(param, name, dtype)
        if saver is not None:
            param = saver.store_early(param)
        state_dict[to_name] = param  # type: ignore

    for i, (q, k, v) in list(qkv_weights.items()):
        if q is None or k is None or v is None:
            # split across different .bin files
            continue
        q = load_param(q, f"layer {i} q", dtype)
        k = load_param(k, f"layer {i} k", dtype)
        v = load_param(v, f"layer {i} v", dtype)
        assert config.n_query_groups
        q_per_kv = config.n_head // config.n_query_groups
        qs = torch.split(q, config.head_size * q_per_kv)
        ks = torch.split(k, config.head_size)
        vs = torch.split(v, config.head_size)
        cycled = [t for group in zip(qs, ks, vs) for t in group]
        qkv = torch.cat(cycled)
        state_dict[f"transformer.h.{i}.attn.attn.weight"] = qkv
        del qkv_weights[i]


def copy_weights_phi(
    config: Config,
    state_dict: Dict[str, torch.Tensor],
    hf_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
    dtype: Optional[torch.dtype] = None,
) -> None:
    weight_map = {
        "layers.0.wte.weight": "transformer.wte.weight",
        "layers.{}.ln.bias": "transformer.h.{}.norm_1.bias",
        "layers.{}.ln.weight": "transformer.h.{}.norm_1.weight",
        "layers.{}.mixer.Wqkv.bias": "transformer.h.{}.attn.attn.bias",
        "layers.{}.mixer.Wqkv.weight": "transformer.h.{}.attn.attn.weight",
        "layers.{}.mixer.out_proj.bias": "transformer.h.{}.attn.proj.bias",
        "layers.{}.mixer.out_proj.weight": "transformer.h.{}.attn.proj.weight",
        "layers.{}.mixer.rotary_emb.inv_freq": None,
        "layers.{}.mlp.fc_1.bias": "transformer.h.{}.mlp.fc.bias",
        "layers.{}.mlp.fc_1.weight": "transformer.h.{}.mlp.fc.weight",
        "layers.{}.mlp.fc2.bias": "transformer.h.{}.mlp.proj.bias",
        "layers.{}.mlp.fc2.weight": "transformer.h.{}.mlp.proj.weight",
        f"layers.{config.n_layer + 1}.ln.bias": "transformer.ln_f.bias",
        f"layers.{config.n_layer + 1}.ln.weight": "transformer.ln_f.weight",
        f"layers.{config.n_layer + 1}.linear.weight": "lm_head.weight",
        f"layers.{config.n_layer + 1}.linear.bias": "lm_head.bias",
    }

    for name, param in hf_weights.items():
        if "layers" in name:
            from_name, number = layer_template(name, 1)
            if number in (0, config.n_layer + 1):
                # these are part of the layers in phi, but not in our implementation
                to_name = weight_map[name]
            else:
                to_name = weight_map[from_name]
                if to_name is None:
                    continue
                # the phi layer numbering is off by 1 compared to ours
                to_name = to_name.format(number - 1)
        else:
            to_name = weight_map[name]
        param = load_param(param, name, dtype)
        if "Wqkv" in name:
            assert config.n_query_groups
            q_per_kv = config.n_head // config.n_query_groups
            total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
            param = param.view(total_qkv, config.n_query_groups, -1).transpose(0, 1)
            param = param.reshape(config.hidden_size * 3, -1)
            if "bias" in name:
                param = param.squeeze()
        if saver is not None:
            param = saver.store_early(param)
        state_dict[to_name] = param  # type: ignore


def layer_template(layer_name: str, idx: int) -> Tuple[str, int]:
    split = layer_name.split(".")
    number = int(split[idx])
    split[idx] = "{}"
    from_name = ".".join(split)
    return from_name, number


def load_param(
    param: Union[torch.Tensor, NotYetLoadedTensor],
    name: str,
    dtype: Optional[torch.dtype],
) -> torch.Tensor:
    if hasattr(param, "_load_tensor"):
        # support tensors loaded via `lazy_load()`
        # print(f"Loading {name!r} into RAM")
        param = param._load_tensor()  # type: ignore
    if (
        dtype is not None
        and type(dtype) is not NotYetLoadedTensor
        and dtype != param.dtype
    ):
        print(f"Converting {name!r} from {param.dtype} to {dtype}")
        param = param.to(dtype)
    return param  # type: ignore


@torch.inference_mode()
def convert_hf_checkpoint(
    checkpoint_dir: str,
    model_name: Optional[str] = None,
    dtype: Optional[str] = None,
) -> None:
    if model_name is None:
        model_name = checkpoint_dir.split("/")[-1]
    if dtype is not None:
        dtype = getattr(torch, dtype)

    print(f"Converting model: {model_name}")
    config = Config.from_name(model_name)
    config_dict = asdict(config)
    print(f"Using model config: {config_dict}")

    if config.model_family == ModelFamily.LLAMA.value:
        # holder to reconstitute the split q, k, v
        qkv_weights = {}
        copy_fn = partial(copy_weights_hf_llama, config, qkv_weights)
    elif config.model_family == ModelFamily.PHI.value:
        copy_fn = partial(copy_weights_phi, config)
    elif config.model_family == ModelFamily.STABLE_LM.value:
        qkv_weights = {}
        copy_fn = partial(copy_weights_stablelm, config, qkv_weights)

    # initialize a new empty state dict to hold our new weights
    sd = {}

    # Load the json file containing weight mapping
    pytorch_bin_map_json_path = os.path.join(
        checkpoint_dir, "pytorch_model.bin.index.json"
    )
    checkpoint_path = Path(checkpoint_dir)
    if os.path.exists(pytorch_bin_map_json_path):  # not all checkpoints have this file
        with open(pytorch_bin_map_json_path) as json_map:
            bin_index = json.load(json_map)
        bin_files = {checkpoint_dir / bin for bin in bin_index["weight_map"].values()}
    else:
        bin_files = set(checkpoint_path.glob("*.bin"))
        # some checkpoints serialize the training arguments
        bin_files = {f for f in bin_files if f.name != "training_args.bin"}
    if not bin_files:
        raise ValueError(f"Expected {str(checkpoint_path)!r} to contain .bin files")

    with incremental_save(checkpoint_path / "lit_model.pth") as saver:
        # for checkpoints that split the QKV across several files, we need to keep all the bin files
        # open, so we use `ExitStack` to close them all together at the end
        for bin_file in sorted(bin_files):
            hf_weights = lazy_load(bin_file)
            copy_fn(sd, hf_weights, saver=saver, dtype=dtype)  # type: ignore
        gc.collect()
        print("Saving converted checkpoint")
        saver.save(sd)


if __name__ == "__main__":
    fire.Fire(convert_hf_checkpoint)
