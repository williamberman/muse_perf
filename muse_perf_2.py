import torch
from torch.utils.benchmark import Timer, Compare
from muse.modeling_taming_vqgan import VQGANModel
from muse.modeling_transformer import MaskGiTUViT
from muse import PipelineMuse, PaellaVQModel
import multiprocessing
import traceback
from argparse import ArgumentParser
import csv
from diffusers import UNet2DConditionModel, AutoencoderKL

from transformers import CLIPTextModel, AutoTokenizer, CLIPTokenizer

torch.manual_seed(0)
torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")

num_threads = torch.get_num_threads()
prompt = "A high tech solarpunk utopia in the Amazon rainforest"


all_models = [
    "openMUSE/muse-laiona6-uvit-clip-220k",
    "runwayml/stable-diffusion-v1-5",
    "williamberman/laiona6plus_uvit_clip_f8",
]

all_batch_sizes = [1, 2, 4, 8, 16, 32]

all_compiled = [None, "default", "reduce-overhead"]

all_components = ["backbone", "vae", "full"]

all_devices = ["4090", "t4", "a100", "cpu"]

skip = {
    ("4090", "runwayml/stable-diffusion-v1-5", "backbone", 32, "reduce-overhead"),
}


def main():
    args = ArgumentParser()
    args.add_argument("--device", required=True)

    args = args.parse_args()

    assert args.device in all_devices

    if args.device in ["4090", "a100", "t4"]:
        dtype = torch.float16
        torch_device = "cuda"
    elif args.device in ["cpu"]:
        dtype = torch.float32
        torch_device = "cpu"
    else:
        assert False

    csv_data = []

    for model in all_models:
        for component in all_components:
            for batch_size in all_batch_sizes:
                for compiled in all_compiled:
                    label = f"single pass {component}, batch_size: {batch_size}, dtype: {dtype}"
                    description = f"{model}, compiled {compiled}"

                    print(label)
                    print(description)

                    if (args.device, model, component, batch_size, compiled) in skip:
                        print("========skipping========")
                        print("*******")
                        continue

                    inputs = [
                        torch_device,
                        dtype,
                        compiled,
                        batch_size,
                        model,
                        label,
                        description,
                    ]

                    fn = model_config[model][component]["fn"]

                    out = run_in_subprocess(fn, inputs=inputs)

                    median = out.median * 1000

                    mean = out.mean * 1000

                    csv_data.append(
                        [
                            batch_size,
                            model,
                            str(compiled),
                            median,
                            mean,
                            args.device,
                            "backbone",
                        ]
                    )

                    Compare([out]).print()
                    print("*******")

    with open("all.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_data)


def muse_benchmark_transformer_backbone(in_queue, out_queue, timeout):
    wrap_subprocess_fn(
        in_queue, out_queue, timeout, _muse_benchmark_transformer_backbone
    )


def _muse_benchmark_transformer_backbone(
    device, dtype, compiled, batch_size, model, label, description
):
    text_encoder = CLIPTextModel.from_pretrained(model, subfolder="text_encoder").to(
        device
    )

    tokenizer = AutoTokenizer.from_pretrained(model, subfolder="text_encoder")

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

    transformer = MaskGiTUViT.from_pretrained(model, subfolder="transformer")

    transformer = transformer.to(device=device, dtype=dtype)

    if compiled is not None:
        transformer = torch.compile(transformer, mode=compiled)

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


def sd_benchmark_unet_backbone(in_queue, out_queue, timeout):
    wrap_subprocess_fn(in_queue, out_queue, timeout, _sd_benchmark_unet_backbone)


def _sd_benchmark_unet_backbone(
    device, dtype, compiled, batch_size, model, label, description
):
    unet = UNet2DConditionModel.from_pretrained(model, subfolder="unet")

    unet = unet.to(device=device, dtype=dtype)

    if compiled is not None:
        unet = torch.compile(unet, mode=compiled)

    text_encoder = CLIPTextModel.from_pretrained(model, subfolder="text_encoder").to(
        device
    )

    tokenizer = CLIPTokenizer.from_pretrained(model, subfolder="tokenizer")

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


def muse_benchmark_vae(in_queue, out_queue, timeout):
    wrap_subprocess_fn(in_queue, out_queue, timeout, _muse_benchmark_vae)


def _muse_benchmark_vae(device, batch_size, dtype, compiled, model, label, description):
    vae_cls = model_config[model]["vae"]["cls"]
    vae = vae_cls.from_pretrained(model, subfolder="vae")

    vae = vae.to(device=device, dtype=dtype)

    if compiled is not None:
        vae = torch.compile(vae, mode=compiled)

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


def sd_benchmark_vae(in_queue, out_queue, timeout):
    wrap_subprocess_fn(in_queue, out_queue, timeout, _sd_benchmark_vae)


def _sd_benchmark_vae(device, batch_size, dtype, compiled, model, label, description):
    vae = AutoencoderKL.from_pretrained(model, subfolder="vae")

    vae = vae.to(device=device, dtype=dtype)

    if compiled is not None:
        vae = torch.compile(vae, mode=compiled)

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


def wrap_subprocess_fn(in_queue, out_queue, timeout, fn):
    error = None
    out = None

    try:
        args = in_queue.get(timeout=timeout)
        out = fn(*args)

    except Exception:
        error = f"{traceback.format_exc()}"

    results = {"error": error, "out": out}
    out_queue.put(results, timeout=timeout)
    out_queue.join()


def run_in_subprocess(target_func, inputs=None):
    timeout = None

    ctx = multiprocessing.get_context("spawn")

    input_queue = ctx.Queue(1)
    output_queue = ctx.JoinableQueue(1)

    # We can't send `unittest.TestCase` to the child, otherwise we get issues regarding pickle.
    input_queue.put(inputs, timeout=timeout)

    process = ctx.Process(target=target_func, args=(input_queue, output_queue, timeout))
    process.start()
    # Kill the child process if we can't get outputs from it in time: otherwise, the hanging subprocess prevents
    # the test to exit properly.
    try:
        results = output_queue.get(timeout=timeout)
        output_queue.task_done()
    except Exception as e:
        process.terminate()
        raise e
    process.join(timeout=timeout)

    if results["error"] is not None:
        raise Exception(results["error"])

    return results["out"]


model_config = {
    "openMUSE/muse-laiona6-uvit-clip-220k": {
        "backbone": {
            "fn": muse_benchmark_transformer_backbone,
        },
        "vae": {
            "fn": muse_benchmark_vae,
            "cls": VQGANModel,
        },
    },
    "runwayml/stable-diffusion-v1-5": {
        "backbone": {
            "fn": sd_benchmark_unet_backbone,
        },
        "vae": {
            "fn": sd_benchmark_vae,
        },
    },
    "williamberman/laiona6plus_uvit_clip_f8": {
        "backbone": {
            "fn": muse_benchmark_transformer_backbone,
        },
        "vae": {
            "fn": muse_benchmark_vae,
            "cls": PaellaVQModel,
        },
    },
}


if __name__ == "__main__":
    main()
