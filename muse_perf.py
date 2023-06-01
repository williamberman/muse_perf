from muse.modeling_taming_vqgan import VQGANModel
from muse.modeling_transformer import MaskGiTUViT
from muse import PipelineMuse

from transformers import CLIPTextModel, AutoTokenizer

from diffusers import UNet2DConditionModel, AutoencoderKL, StableDiffusionPipeline

import torch
from torch.utils.benchmark import Timer, Compare
from argparse import ArgumentParser

torch.manual_seed(0)
torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")

num_threads = torch.get_num_threads()
muse_model = "openMUSE/muse-laiona6-uvit-clip-220k"
sd_model = "runwayml/stable-diffusion-v1-5"
prompt = "A high tech solarpunk utopia in the Amazon rainforest"


def make_encoder_hidden_states(*, device, dtype, batch_size, tokenizer, text_encoder):
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


def make_muse_transformer(*, device, compiled, dtype):
    transformer = MaskGiTUViT.from_pretrained(muse_model, subfolder="transformer")

    transformer = transformer.to(device=device, dtype=dtype)

    if compiled:
        transformer = torch.compile(transformer)

    return transformer


def make_sd_unet(*, device, compiled, dtype):
    unet = UNet2DConditionModel.from_pretrained(sd_model, subfolder="unet")

    unet = unet.to(device=device, dtype=dtype)

    if compiled:
        unet = torch.compile(unet)

    return unet


def make_muse_vae(*, device, compiled, dtype):
    vae = VQGANModel.from_pretrained(muse_model, subfolder="vae")

    vae = vae.to(device=device, dtype=dtype)

    if compiled:
        vae = torch.compile(vae)

    return vae


def make_sd_vae(*, device, compiled, dtype):
    vae = AutoencoderKL.from_pretrained(sd_model, subfolder="vae")

    vae = vae.to(device=device, dtype=dtype)

    if compiled:
        vae = torch.compile(vae)

    return vae


def benchmark_transformer_backbone(
    *, device, dtype, compiled, batch_size, transformer, encoder_hidden_states
):
    label = f"single pass backbone, batch_size: {batch_size}, dtype: {dtype}"
    description = f"{muse_model}, compiled {compiled}"

    print("*******")
    print(label)
    print(description)

    image_tokens = torch.full(
        (batch_size, 256), fill_value=5, dtype=torch.long, device=device
    )

    def benchmark_fn():
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


def benchmark_unet_backbone(
    *, device, dtype, compiled, batch_size, unet, encoder_hidden_states
):
    label = f"single pass backbone, batch_size: {batch_size}, dtype: {dtype}"
    description = f"{sd_model}, compiled {compiled}"

    print("*******")
    print(label)
    print(description)

    latent_image = torch.randn((batch_size, 4, 64, 64), dtype=dtype, device=device)

    t = torch.randint(1, 999, (batch_size,), dtype=dtype, device=device)

    def benchmark_fn():
        unet(latent_image, timestep=t, encoder_hidden_states=encoder_hidden_states)

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


def benchmark_muse_vae(*, device, batch_size, dtype, compiled, vae):
    label = f"single pass vae, batch_size: {batch_size}, dtype: {dtype}"
    description = f"{muse_model}, compiled {compiled}"

    print("*******")
    print(label)
    print(description)

    image_tokens = torch.full(
        (batch_size, 256), fill_value=5, dtype=torch.long, device=device
    )

    def benchmark_fn():
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


def benchmark_sd_vae(*, device, batch_size, dtype, compiled, vae):
    label = f"single pass vae, batch_size: {batch_size}, dtype: {dtype}"
    description = f"{sd_model}, compiled {compiled}"

    print("*******")
    print(label)
    print(description)

    latent_image = torch.randn((batch_size, 4, 64, 64), dtype=dtype, device=device)

    def benchmark_fn():
        vae.decode(latent_image)

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


def benchmark_muse_full(*, device, batch_size, dtype, compiled, pipe):
    label = f"full pipeline, batch_size: {batch_size}, dtype: {dtype}"
    description = f"{muse_model}, compiled {compiled}"

    print("*******")
    print(label)
    print(description)

    def benchmark_fn():
        pipe(prompt, num_images_per_prompt=batch_size, timesteps=12)

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


def benchmark_sd_full(*, device, batch_size, dtype, compiled, pipe):
    label = f"full pipeline, batch_size: {batch_size}, dtype: {dtype}"
    description = f"{sd_model}, compiled {compiled}"

    print("*******")
    print(label)
    print(description)

    def benchmark_fn():
        pipe(prompt, num_images_per_prompt=batch_size, timesteps=20)

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


backbone_params = {
    "cuda": {
        "batch_size": [1, 2, 4, 8, 16, 32],
        "dtype": [torch.float16],
        "compiled": [False, True],
    },
    "cpu": {
        "batch_size": [1, 2, 4, 8],
        "dtype": [torch.float32],
        "compiled": [False, True],
    },
}


def main_backbone(device, file):
    results = []

    text_encoder = CLIPTextModel.from_pretrained(
        muse_model, subfolder="text_encoder"
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(muse_model, subfolder="text_encoder")

    for dtype in backbone_params[device]["dtype"]:
        for batch_size in backbone_params[device]["batch_size"]:
            encoder_hidden_states = make_encoder_hidden_states(
                device=device,
                dtype=dtype,
                batch_size=batch_size,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
            )

            for compiled in backbone_params[device]["compiled"]:
                muse_transformer = make_muse_transformer(
                    device=device, compiled=compiled, dtype=dtype
                )

                bm = benchmark_transformer_backbone(
                    device=device,
                    dtype=dtype,
                    compiled=compiled,
                    batch_size=batch_size,
                    transformer=muse_transformer,
                    encoder_hidden_states=encoder_hidden_states,
                )

                results.append(bm)

                sd_unet = make_sd_unet(device=device, compiled=compiled, dtype=dtype)

                bm = benchmark_unet_backbone(
                    device=device,
                    dtype=dtype,
                    compiled=compiled,
                    batch_size=batch_size,
                    unet=sd_unet,
                    encoder_hidden_states=encoder_hidden_states,
                )

                results.append(bm)

    out = str(Compare(results))

    with open(file, "w") as f:
        f.write(out)


vae_params = {
    "cuda": {
        "batch_size": [1, 2, 4, 8, 16, 32],
        "dtype": [torch.float16],
        "compiled": [False, True],
    },
    "cpu": {
        "batch_size": [1, 2, 4, 8],
        "dtype": [torch.float32],
        "compiled": [False, True],
    },
}


def main_vae(device, file):
    results = []

    for dtype in vae_params[device]["dtype"]:
        for batch_size in vae_params[device]["batch_size"]:
            for compiled in vae_params[device]["compiled"]:
                muse_vae = make_muse_vae(device=device, compiled=compiled, dtype=dtype)

                bm = benchmark_muse_vae(
                    device=device,
                    dtype=dtype,
                    compiled=compiled,
                    batch_size=batch_size,
                    vae=muse_vae,
                )

                results.append(bm)

                sd_vae = make_sd_vae(device=device, compiled=compiled, dtype=dtype)

                bm = benchmark_sd_vae(
                    device=device,
                    dtype=dtype,
                    compiled=compiled,
                    batch_size=batch_size,
                    vae=sd_vae,
                )

                results.append(bm)

    out = str(Compare(results))

    with open(file, "w") as f:
        f.write(out)


full_params = {
    "cuda": {
        "batch_size": [1, 2, 4, 8, 16, 32],
        "dtype": [torch.float16],
        "compiled": [False, True],
    },
    "cpu": {
        "batch_size": [1, 2, 4, 8],
        "dtype": [torch.float32],
        "compiled": [False, True],
    },
}


def main_full(device, file):
    results = []

    tokenizer = AutoTokenizer.from_pretrained(muse_model, subfolder="text_encoder")

    for dtype in vae_params[device]["dtype"]:
        text_encoder = CLIPTextModel.from_pretrained(
            muse_model, subfolder="text_encoder"
        ).to(device=device, dtype=dtype)

        for batch_size in vae_params[device]["batch_size"]:
            for compiled in vae_params[device]["compiled"]:
                muse_vae = make_muse_vae(device=device, compiled=compiled, dtype=dtype)
                muse_transformer = make_muse_transformer(
                    device=device, compiled=compiled, dtype=dtype
                )

                pipe = PipelineMuse(
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    vae=muse_vae,
                    transformer=muse_transformer,
                )
                pipe.device = device
                pipe.dtype = dtype

                bm = benchmark_muse_full(
                    device=device,
                    dtype=dtype,
                    compiled=compiled,
                    batch_size=batch_size,
                    pipe=pipe,
                )

                results.append(bm)

                sd_vae = make_sd_vae(device=device, compiled=compiled, dtype=dtype)
                sd_unet = make_sd_unet(device=device, compiled=compiled, dtype=dtype)

                pipe = StableDiffusionPipeline.from_pretrained(
                    sd_model,
                    vae=sd_vae,
                    unet=sd_unet,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    safety_checker=None,
                )

                bm = benchmark_sd_full(
                    device=device,
                    dtype=dtype,
                    compiled=compiled,
                    batch_size=batch_size,
                    pipe=pipe,
                )

                results.append(bm)

    out = str(Compare(results))

    with open(file, "w") as f:
        f.write(out)


parser = ArgumentParser()
parser.add_argument("--model", required=False)
parser.add_argument("--full", required=False, action="store_true")
parser.add_argument("--device", required=True)
parser.add_argument("--file", required=True)

args = parser.parse_args()

if args.full:
    main_full(args.device, args.file)
else:
    if args.model == "backbone":
        main_backbone(args.device, args.file)
    elif args.model == "vae":
        main_vae(args.device, args.file)
    else:
        assert False
