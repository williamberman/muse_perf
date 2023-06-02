from muse.modeling_taming_vqgan import VQGANModel
from muse.modeling_transformer import MaskGiTUViT
from muse import PipelineMuse

from transformers import CLIPTextModel, AutoTokenizer, CLIPTokenizer

from diffusers import UNet2DConditionModel, AutoencoderKL, StableDiffusionPipeline

import torch
from torch.utils.benchmark import Timer, Compare
from argparse import ArgumentParser

torch.manual_seed(0)
torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")

num_threads = torch.get_num_threads()
prompt = "A high tech solarpunk utopia in the Amazon rainforest"

all_models = {
    "muse_f16": "openMUSE/muse-laiona6-uvit-clip-220k",
    "sd": "runwayml/stable-diffusion-v1-5",
    "muse_f8": "williamberman/laiona6plus_uvit_clip_f8",
}


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


def make_muse_transformer(*, device, compiled, dtype, model):
    transformer = MaskGiTUViT.from_pretrained(model, subfolder="transformer")

    transformer = transformer.to(device=device, dtype=dtype)

    if compiled is not None:
        transformer = torch.compile(transformer, mode=compiled)

    return transformer


def make_sd_unet(*, device, compiled, dtype):
    unet = UNet2DConditionModel.from_pretrained(all_models["sd"], subfolder="unet")

    unet = unet.to(device=device, dtype=dtype)

    if compiled is not None:
        unet = torch.compile(unet, mode=compiled)

    return unet


def make_muse_vae(*, device, compiled, dtype, model):
    if model == all_models["muse_f16"]:
        vae = VQGANModel.from_pretrained(model, subfolder="vae")
    elif model == all_models["muse_f8"]:
        vae = PaellaVQModel.from_pretrained(model, subfolder="vae")
    else:
        assert False, model

    vae = vae.to(device=device, dtype=dtype)

    if compiled is not None:
        vae = torch.compile(vae, mode=compiled)

    return vae


def make_sd_vae(*, device, compiled, dtype):
    vae = AutoencoderKL.from_pretrained(all_models["sd"], subfolder="vae")

    vae = vae.to(device=device, dtype=dtype)

    if compiled is not None:
        vae = torch.compile(vae, mode=compiled)

    return vae


def benchmark_transformer_backbone(
    *, device, dtype, compiled, batch_size, transformer, encoder_hidden_states, model
):
    label = f"single pass backbone, batch_size: {batch_size}, dtype: {dtype}"
    description = f"{model}, compiled {compiled}"

    print("*******")
    print(label)
    print(description)

    image_tokens = torch.full(
        (batch_size, 256), fill_value=5, dtype=torch.long, device=device
    )

    def benchmark_fn():
        transformer(image_tokens, encoder_hidden_states=encoder_hidden_states)

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
    description = f"{all_models['sd']}, compiled {compiled}"

    print("*******")
    print(label)
    print(description)

    latent_image = torch.randn((batch_size, 4, 64, 64), dtype=dtype, device=device)

    t = torch.randint(1, 999, (batch_size,), dtype=dtype, device=device)

    def benchmark_fn():
        unet(latent_image, timestep=t, encoder_hidden_states=encoder_hidden_states)

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
    description = f"{all_models['muse_f16']}, compiled {compiled}"

    print("*******")
    print(label)
    print(description)

    image_tokens = torch.full(
        (batch_size, 256), fill_value=5, dtype=torch.long, device=device
    )

    def benchmark_fn():
        vae.decode_code(image_tokens)

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
    description = f"{all_models['sd']}, compiled {compiled}"

    print("*******")
    print(label)
    print(description)

    latent_image = torch.randn((batch_size, 4, 64, 64), dtype=dtype, device=device)

    def benchmark_fn():
        vae.decode(latent_image)

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
    description = f"{all_models['muse_f16']}, compiled {compiled}"

    print("*******")
    print(label)
    print(description)

    def benchmark_fn():
        pipe(prompt, num_images_per_prompt=batch_size, timesteps=12)

    pipe(prompt, num_images_per_prompt=batch_size, timesteps=5)

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
    description = f"{all_models['sd']}, compiled {compiled}"

    print("*******")
    print(label)
    print(description)

    def benchmark_fn():
        pipe(prompt, num_images_per_prompt=batch_size, num_inference_steps=20)

    pipe(prompt, num_images_per_prompt=batch_size, num_inference_steps=5)

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
        "compiled": [None, "default", "reduce-overhead"],
    },
    "cpu": {
        "batch_size": [1, 2, 4, 8],
        "dtype": [torch.float32],
        "compiled": [None, "default", "reduce-overhead"],
    },
}


def main_backbone(device, batch_size=None, dtype=None, compiled=None, model=None):
    results = []

    dtype_ = dtype or backbone_params[device]["dtype"]
    batch_size_ = batch_size or backbone_params[device]["batch_size"]
    compiled_ = compiled or backbone_params[device]["compiled"]
    model_ = model or all_models.keys()

    for dtype in dtype_:
        for batch_size in batch_size_:
            for compiled in compiled_:
                for model in model_:
                    if model in ["muse_f16", "muse_f8"]:
                        text_encoder = CLIPTextModel.from_pretrained(
                            all_models[model], subfolder="text_encoder"
                        ).to(device)

                        tokenizer = AutoTokenizer.from_pretrained(
                            all_models[model], subfolder="text_encoder"
                        )

                        encoder_hidden_states = make_encoder_hidden_states(
                            device=device,
                            dtype=dtype,
                            batch_size=batch_size,
                            tokenizer=tokenizer,
                            text_encoder=text_encoder,
                        )

                        muse_transformer = make_muse_transformer(
                            device=device,
                            compiled=compiled,
                            dtype=dtype,
                            model=all_models[model],
                        )

                        bm = benchmark_transformer_backbone(
                            device=device,
                            dtype=dtype,
                            compiled=compiled,
                            batch_size=batch_size,
                            transformer=muse_transformer,
                            encoder_hidden_states=encoder_hidden_states,
                            model=all_models[model],
                        )

                        results.append(bm)
                    elif model == "sd":
                        text_encoder = CLIPTextModel.from_pretrained(
                            all_models[model], subfolder="text_encoder"
                        ).to(device)

                        tokenizer = AutoTokenizer.from_pretrained(
                            all_models[model], subfolder="text_encoder"
                        )

                        encoder_hidden_states = make_encoder_hidden_states(
                            device=device,
                            dtype=dtype,
                            batch_size=batch_size,
                            tokenizer=tokenizer,
                            text_encoder=text_encoder,
                        )

                        sd_unet = make_sd_unet(
                            device=device, compiled=compiled, dtype=dtype
                        )

                        bm = benchmark_unet_backbone(
                            device=device,
                            dtype=dtype,
                            compiled=compiled,
                            batch_size=batch_size,
                            unet=sd_unet,
                            encoder_hidden_states=encoder_hidden_states,
                        )

                        results.append(bm)
                    else:
                        assert False, model

    return results


vae_params = {
    "cuda": {
        "batch_size": [1, 2, 4, 8, 16, 32],
        "dtype": [torch.float16],
        "compiled": [None, "default", "reduce-overhead"],
    },
    "cpu": {
        "batch_size": [1, 2, 4, 8],
        "dtype": [torch.float32],
        "compiled": [None, "default", "reduce-overhead"],
    },
}


def main_vae(device, dtype=None, batch_size=None, compiled=None, model=None):
    results = []

    dtype_ = dtype or vae_params[device]["dtype"]
    batch_size_ = batch_size or vae_params[device]["batch_size"]
    compiled_ = compiled or vae_params[device]["compiled"]
    model_ = model or all_models.keys()

    for dtype in dtype_:
        for batch_size in batch_size_:
            for compiled in compiled_:
                for model in model_:
                    if model in ["muse_f16", "muse_f8"]:
                        muse_vae = make_muse_vae(
                            device=device,
                            compiled=compiled,
                            dtype=dtype,
                            model=all_models[model],
                        )

                        bm = benchmark_muse_vae(
                            device=device,
                            dtype=dtype,
                            compiled=compiled,
                            batch_size=batch_size,
                            vae=muse_vae,
                        )

                        results.append(bm)
                    elif model == "sd":
                        sd_vae = make_sd_vae(
                            device=device, compiled=compiled, dtype=dtype
                        )

                        bm = benchmark_sd_vae(
                            device=device,
                            dtype=dtype,
                            compiled=compiled,
                            batch_size=batch_size,
                            vae=sd_vae,
                        )

                        results.append(bm)
                    else:
                        assert False, model

    return results


full_params = {
    "cuda": {
        "batch_size": [1, 2, 4, 8, 16, 32],
        "dtype": [torch.float16],
        "compiled": [None, "default", "reduce-overhead"],
    },
    "cpu": {
        "batch_size": [1, 2, 4, 8],
        "dtype": [torch.float32],
        "compiled": [None, "default", "reduce-overhead"],
    },
}


def main_full(device, batch_size=None, dtype=None, compiled=None, model=None):
    results = []

    dtype_ = dtype or full_params[device]["dtype"]
    batch_size_ = batch_size or full_params[device]["batch_size"]
    compiled_ = compiled or full_params[device]["compiled"]
    model_ = model or all_models.keys()

    for dtype in dtype_:
        for batch_size in batch_size_:
            for compiled in compiled_:
                for model in model_:
                    if model in ["muse_f16", "muse_f8"]:
                        tokenizer = CLIPTokenizer.from_pretrained(
                            all_models[model], subfolder="text_encoder"
                        )

                        text_encoder = CLIPTextModel.from_pretrained(
                            all_models[model], subfolder="text_encoder"
                        ).to(device=device, dtype=dtype)

                        muse_vae = make_muse_vae(
                            device=device,
                            compiled=compiled,
                            dtype=dtype,
                            model=all_models[model],
                        )
                        muse_transformer = make_muse_transformer(
                            device=device,
                            compiled=compiled,
                            dtype=dtype,
                            model=all_models[model],
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
                    elif model == "sd":
                        # skip for stable diffusion
                        if batch_size > 1 and device == "cpu":
                            continue

                        tokenizer = CLIPTokenizer.from_pretrained(
                            all_models[model], subfolder="text_encoder"
                        )

                        text_encoder = CLIPTextModel.from_pretrained(
                            all_models[model], subfolder="text_encoder"
                        ).to(device=device, dtype=dtype)

                        sd_vae = make_sd_vae(
                            device=device, compiled=compiled, dtype=dtype
                        )
                        sd_unet = make_sd_unet(
                            device=device, compiled=compiled, dtype=dtype
                        )

                        pipe = StableDiffusionPipeline.from_pretrained(
                            all_models["sd"],
                            vae=sd_vae,
                            unet=sd_unet,
                            text_encoder=text_encoder,
                            tokenizer=tokenizer,
                            safety_checker=None,
                        )

                        # diffusers strips the compilation from the module
                        if compiled is not None:
                            pipe.vae = torch.compile(pipe.vae, mode=compiled)
                            pipe.unet = torch.compile(pipe.unet, mode=compiled)

                        bm = benchmark_sd_full(
                            device=device,
                            dtype=dtype,
                            compiled=compiled,
                            batch_size=batch_size,
                            pipe=pipe,
                        )

                        results.append(bm)
                    else:
                        assert False, model

    return results


def compiled_parser(string):
    if string in ["default", "reduce-overhead"]:
        return string
    elif string == "None":
        return None
    else:
        assert False, f"{string} can't parse as compiled option"


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--component", required=False)
    parser.add_argument("--full", required=False, action="store_true")
    parser.add_argument("--device", required=True)
    parser.add_argument("--file", required=False)
    parser.add_argument("--batch_size", required=False, nargs="+", type=int)
    parser.add_argument("--dtype", required=False, nargs="+")
    parser.add_argument("--compiled", required=False, nargs="+", type=compiled_parser)
    parser.add_argument("--model", required=False, nargs="+")

    args = parser.parse_args()

    if args.dtype is not None:
        dtypes = []
        for dtype in args.dtype:
            if dtype == "float32":
                dtypes.append(torch.float32)
            elif dtype == "float16":
                dtypes.append(torch.float16)
            else:
                assert False
    else:
        dtypes = None

    if args.full:
        results = main_full(
            args.device,
            batch_size=args.batch_size,
            dtype=dtypes,
            compiled=args.compiled,
            model=args.model,
        )
    else:
        if args.component == "backbone":
            results = main_backbone(
                args.device,
                batch_size=args.batch_size,
                dtype=dtypes,
                compiled=args.compiled,
                model=args.model,
            )
        elif args.component == "vae":
            results = main_vae(
                args.device,
                batch_size=args.batch_size,
                dtype=dtypes,
                compiled=args.compiled,
                model=args.model,
            )
        else:
            assert False, args.component

    results = str(Compare(results))

    if args.file is not None:
        with open(args.file, "a") as f:
            f.write(results)
    else:
        print(results)
