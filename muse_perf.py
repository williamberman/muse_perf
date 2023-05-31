from muse.modeling_taming_vqgan import VQGANModel
from muse.modeling_transformer import MaskGiTUViT

from transformers import CLIPTextModel, AutoTokenizer

import torch
from torch.utils.benchmark import Timer, Compare

torch.manual_seed(0)
torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")

num_threads = torch.get_num_threads()
model = "openMUSE/muse-cc12m-uvit-clip-130k"


def make_encoder_hidden_states(*, device, dtype, batch_size):
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
        with torch.cuda.amp(enabled=autocast):
            transformer(image_tokens, encoder_hidden_states=encoder_hidden_states)

    if compiled:
        benchmark_fn()

    out = Timer(
        stmt="benchmark_fn()",
        setup="from __main__ import benchmark_fn",
        num_threads=num_threads,
        label=label,
        description=description,
    ).blocked_autorange(min_run_time=1)

    return out


def benchmark_vae(*, batch_size, dtype, compiled, autocast, vae):
    label = f"{model}, single pass vae, batch_size: {batch_size}"
    description = f"dtype: {dtype}, compiled {compiled}, autocast {autocast}"

    print("*******")
    print(label)
    print(description)

    image_tokens = torch.full(
        (batch_size, 256), fill_value=5, dtype=torch.long, device=device, dtype=dtype
    )

    def benchmark_fn():
        with torch.cuda.amp(enabled=autocast):
            vae.decode_code(image_tokens)

    if compiled:
        benchmark_fn()

    out = Timer(
        stmt="benchmark_fn()",
        setup="from __main__ import benchmark_fn",
        num_threads=num_threads,
        label=label,
        description=description,
    ).blocked_autorange(min_run_time=1)

    return out


params = {
    "cuda": {
        "batch_size": [1, 2, 4, 8, 16, 32],
        "dtypes": [torch.float32, torch.float16],
        "compiled": [False, True],
        "autocast": [False, True],
    },
    "cpu": {
        "batch_size": [1, 2, 4, 8],
        "dtypes": [torch.float32, torch.float16],
        "compiled": [False, True],
        "autocast": [False, True],
    },
}

results = []

for device in ["cuda", "cpu"]:
    text_encoder = CLIPTextModel.from_pretrained(model, subfolder="text_encoder").to(
        device
    )
    tokenizer = AutoTokenizer.from_pretrained(model, subfolder="text_encoder")

    for batch_size in params[device]["batch_size"]:
        for dtype in params[device]["dtype"]:

            encoder_hidden_states = make_encoder_hidden_states(
                device=device, dtype=dtype, batch_size=batch_size
            )

            for compiled in params[device][dtype]:

                transformer = make_transformer(
                    device=device, compiled=compiled, dtype=dtype
                )

                for autocast in params[device]["autocast"]:
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

for device in ["cuda", "cpu"]:
    for batch_size in params[device]["batch_size"]:
        for dtype in params[device]["dtype"]:
            for compiled in params[device][dtype]:

                vae = make_vae(device=device, compiled=compiled, dtype=dtype)

                for autocast in params[device]["autocast"]:
                    bm = benchmark_vae(
                        device=device,
                        dtype=dtype,
                        compiled=compiled,
                        batch_size=batch_size,
                        autocast=autocast,
                        vae=vae,
                    )

                    results.append(bm)


Compare(results).print()
