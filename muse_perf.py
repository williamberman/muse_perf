from muse.modeling_taming_vqgan import VQGANModel
from muse.modeling_transformer import MaskGiTUViT

from transformers import CLIPTextModel, AutoTokenizer

import torch
from torch.utils.benchmark import Timer, Compare
from argparse import ArgumentParser

torch.manual_seed(0)
torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")

num_threads = torch.get_num_threads()
model = "openMUSE/muse-cc12m-uvit-clip-130k"


def make_encoder_hidden_states(*, device, dtype, batch_size, tokenizer, text_encoder):
    prompt = "A high tech solarpunk utopia in the Amazon rainforest"

    text_tokens = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
    ).input_ids
    text_tokens = text_tokens.to(device)

    encoder_hidden_states = text_encoder(text_tokens).last_hidden_state

    encoder_hidden_states = encoder_hidden_states.expand(batch_size, -1, -1)

    encoder_hidden_states = encoder_hidden_states.to(dtype)

    return encoder_hidden_states


def make_transformer(*, device, compiled, dtype):
    transformer = MaskGiTUViT.from_pretrained(model, subfolder="transformer")

    transformer = transformer.to(device=device, dtype=dtype)

    if compiled:
        transformer = torch.compile(transformer)

    return transformer


def make_vae(*, device, compiled, dtype):
    vae = VQGANModel.from_pretrained(model, subfolder="vae")

    vae = vae.to(device=device, dtype=dtype)

    if compiled:
        vae = torch.compile(vae)

    return vae


def benchmark_transformer(
    *, device, dtype, compiled, batch_size, autocast, transformer, encoder_hidden_states
):
    label = f"{model}, single pass transformer, batch_size: {batch_size}"
    description = f"dtype: {dtype}, compiled {compiled}, autocast {autocast}"

    print("*******")
    print(label)
    print(description)

    image_tokens = torch.full(
        (batch_size, 256), fill_value=5, dtype=torch.long, device=device
    )

    def benchmark_fn():
        with torch.cuda.amp.autocast(enabled=autocast):
            transformer(image_tokens, encoder_hidden_states=encoder_hidden_states)

    if compiled:
        benchmark_fn()

    out = Timer(
        stmt="benchmark_fn()",
        globals={"benchmark_fn": benchmark_fn},
        num_threads=num_threads,
        label=label,
        description=description,
    ).blocked_autorange(min_run_time=1)

    return out


def benchmark_vae(*, device, batch_size, dtype, compiled, autocast, vae):
    label = f"{model}, single pass vae, batch_size: {batch_size}"
    description = f"dtype: {dtype}, compiled {compiled}, autocast {autocast}"

    print("*******")
    print(label)
    print(description)

    image_tokens = torch.full(
        (batch_size, 256), fill_value=5, dtype=torch.long, device=device
    )

    def benchmark_fn():
        with torch.cuda.amp.autocast(enabled=autocast):
            vae.decode_code(image_tokens)

    if compiled:
        benchmark_fn()

    out = Timer(
        stmt="benchmark_fn()",
        globals={"benchmark_fn": benchmark_fn},
        num_threads=num_threads,
        label=label,
        description=description,
    ).blocked_autorange(min_run_time=1)

    return out


transformer_params = {
    "cuda": {
        "batch_size": [1, 2, 4, 8, 16, 32],
        "dtype": [torch.float32, torch.float16],
        "compiled": [False, True],
        "autocast": [False, True],
    },
    "cpu": {
        "batch_size": [1, 2, 4, 8],
        "dtype": [torch.float32],
        "compiled": [False, True],
        "autocast": [False, True],
    },
}


def main_transformer(device, file):
    results = []

    text_encoder = CLIPTextModel.from_pretrained(model, subfolder="text_encoder").to(
        device
    )
    tokenizer = AutoTokenizer.from_pretrained(model, subfolder="text_encoder")

    for batch_size in transformer_params[device]["batch_size"]:
        for dtype in transformer_params[device]["dtype"]:
            encoder_hidden_states = make_encoder_hidden_states(
                device=device,
                dtype=dtype,
                batch_size=batch_size,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
            )

            for compiled in transformer_params[device]["compiled"]:
                transformer = make_transformer(
                    device=device, compiled=compiled, dtype=dtype
                )

                for autocast in transformer_params[device]["autocast"]:
                    bm = benchmark_transformer(
                        device=device,
                        dtype=dtype,
                        compiled=compiled,
                        batch_size=batch_size,
                        autocast=autocast,
                        transformer=transformer,
                        encoder_hidden_states=encoder_hidden_states,
                    )

                    results.append(bm)

    out = str(Compare(results))

    with open(file, "w") as f:
        f.write(out)


vae_params = {
    "cuda": {
        "batch_size": [1, 2, 4, 8, 16, 32],
        "dtype": [torch.float32, torch.float16],
        "compiled": [False, True],
        "autocast": [False, True],
    },
    "cpu": {
        "batch_size": [1, 2, 4, 8],
        "dtype": [torch.float32, torch.float16],
        "compiled": [False, True],
        "autocast": [False, True],
    },
}


def main_vae(device, file):
    results = []

    for batch_size in vae_params[device]["batch_size"]:
        for dtype in vae_params[device]["dtype"]:
            for compiled in vae_params[device]["compiled"]:
                vae = make_vae(device=device, compiled=compiled, dtype=dtype)

                for autocast in vae_params[device]["autocast"]:
                    bm = benchmark_vae(
                        device=device,
                        dtype=dtype,
                        compiled=compiled,
                        batch_size=batch_size,
                        autocast=autocast,
                        vae=vae,
                    )

                    results.append(bm)

    out = str(Compare(results))

    with open(file, "w") as f:
        f.write(out)


parser = ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--device", required=True)
parser.add_argument("--file", required=True)

args = parser.parse_args()

if args.model == "transformer":
    main_transformer(args.device, args.file)
elif args.model == "vae":
    main_vae(args.device, args.file)
