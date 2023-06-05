import torch
from torch.utils.benchmark import Timer, Compare
from muse.modeling_taming_vqgan import VQGANModel
from muse.modeling_transformer import MaskGiTUViT
from muse import PipelineMuse, PaellaVQModel
import multiprocessing
import traceback
from argparse import ArgumentParser
import csv

from transformers import CLIPTextModel, AutoTokenizer, CLIPTokenizer

torch.manual_seed(0)
torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")

num_threads = torch.get_num_threads()
prompt = "A high tech solarpunk utopia in the Amazon rainforest"

all_compiled = [None, "default", "reduce-overhead"]

params = {
    "4090": {
        "openMUSE/muse-laiona6-uvit-clip-220k": {
            "backbone": {
                "batch_size": {
                    1: all_compiled,
                    2: all_compiled,
                    4: all_compiled,
                    8: all_compiled,
                    16: all_compiled,
                    32: all_compiled,
                }
            }
        },
        "runwayml/stable-diffusion-v1-5": {
            "backbone": {
                "batch_size": {
                    1: all_compiled,
                    2: all_compiled,
                    4: all_compiled,
                    8: all_compiled,
                    16: all_compiled,
                    32: {"compiled": [None, "default"]},
                }
            }
        },
        "williamberman/laiona6plus_uvit_clip_f8": {
            "backbone": {
                "batch_size": {
                    1: all_compiled,
                    2: all_compiled,
                    4: all_compiled,
                    8: all_compiled,
                    16: all_compiled,
                    32: all_compiled,
                }
            }
        },
    },
}


def main():
    args = ArgumentParser()
    args.add_argument("--device", required=True)

    args = args.parse_args()

    assert args.device in params

    if args.device in ["4090", "a100", "t4"]:
        dtype = torch.float16
        torch_device = "cuda"
    elif args.device in ["cpu"]:
        dtype = torch.float32
        torch_device = "cpu"
    else:
        assert False

    csv_data = []

    for model in params[args.device].keys():
        for batch_size in params[args.device][model]["backbone"]["batch_size"].keys():
            for compiled in params[args.device][model]["backbone"]["batch_size"][
                batch_size
            ]:
                out = muse_benchmark_transformer_backbone(
                    torch_device, dtype, compiled, batch_size, model
                )

                median = out.median * 1000

                mean = out.mean * 1000

                csv_data.append(
                    [batch_size, model, compiled, median, mean, args.device, "backbone"]
                )

    with open("all.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_data)


def muse_benchmark_transformer_backbone(device, dtype, compiled, batch_size, model):
    label = f"single pass backbone, batch_size: {batch_size}, dtype: {dtype}"
    description = f"{model}, compiled {compiled}"

    print(label)
    print(description)

    inputs = [device, dtype, compiled, batch_size, model, label, description]

    out = run_in_subprocess(_muse_benchmark_transformer_backbone, inputs=inputs)

    Compare([out]).print()

    print("*******")
    print()
    print()
    print()

    return out


def _muse_benchmark_transformer_backbone(in_queue, out_queue, timeout):
    error = None
    out = None

    try:
        device, dtype, compiled, batch_size, model, label, description = in_queue.get(
            timeout=timeout
        )

        text_encoder = CLIPTextModel.from_pretrained(
            model, subfolder="text_encoder"
        ).to(device)

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

    except Exception:
        error = f"{traceback.format_exc()}"

    results = {"error": error, "out": out}
    out_queue.put(results, timeout=timeout)
    out_queue.join()


def run_in_subprocess(target_func, inputs=None):
    timeout = 600

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


if __name__ == "__main__":
    main()
