import matplotlib.pyplot as plt
import pandas as pd
from argparse import ArgumentParser

df = pd.read_csv("all.csv")
bar_width = 0.25

model_names = [
    "openMUSE/muse-laiona6-uvit-clip-220k",
    "williamberman/laiona6plus_uvit_clip_f8",
    "runwayml/stable-diffusion-v1-5",
]


def chart(device, component, compiled, plot_on, legend):
    fdf = df[
        (df["Device"] == device)
        & (df["Component"] == component)
        & (df["Compilation Type"] == compiled)
    ]

    placement = range(6)

    def inc_placement():
        nonlocal placement
        placement = [x + bar_width for x in placement]

    for model_name in model_names:
        filter_ = fdf["Model Name"] == model_name

        ffdf = fdf[filter_]

        times = ffdf["Time"].tolist()

        for _ in range(6 - len(times)):
            times.append(0)

        bars = plot_on.bar(placement, times, width=bar_width, label=f"{model_name}")

        for bar in bars:
            yval = bar.get_height()
            plot_on.text(
                bar.get_x() + bar.get_width() / 2,
                yval + 0.05,
                yval,
                ha="center",
                va="bottom",
                rotation=80,
            )

        inc_placement()

    plot_on.set_xlabel("Batch Size")
    plot_on.set_ylabel("Time (ms)")
    plot_on.set_xticks([r + bar_width for r in range(6)], [1, 2, 4, 8, 16, 32])
    plot_on.set_title(f"{device}, {component}, compiled: {compiled}")

    if legend:
        plot_on.legend(fontsize="x-small")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--component", required=True)

    args = parser.parse_args()

    assert args.component in ["full", "backbone", "vae"]

    fig, axs = plt.subplots(4, 3, sharey="row")

    for row_idx, device in enumerate(["a100", "4090", "t4", "cpu"]):
        for col_idx, compiled in enumerate(["None", "default", "reduce-overhead"]):
            legend = row_idx == 0 and col_idx == 2
            chart(device, args.component, compiled, axs[row_idx, col_idx], legend)

    plt.subplots_adjust(hspace=0.75, wspace=0.50)

    plt.show()
