import re
import csv

# header_regex = re.compile(r'full pipeline, batch_size: (\d+), dtype: (.+)')
header_regex = re.compile(r"batch_size: (\d+), dtype: (.+)")
model_regex = re.compile(r"(\w+/[\w|-]+), compiled ([\w|-]+)")
benchmark_regex = re.compile(r"benchmark_fn\(\)\s+\|\s+(.+)")
time_regex = re.compile(r"Times are in (.+)\.")

header = ["Batch Size", "Model Name", "Compilation Type", "Time", "Device", "Component"]

csv_data = [header]


def parse_file(data, device, component):
    lines = data.strip().split("\n")

    i = 0

    while i < len(lines):
        header_match = header_regex.search(lines[i])

        if header_match is None:
            import ipdb

            ipdb.set_trace()

        batch_size = header_match.group(1)
        batch_size = int(batch_size)

        dtype = header_match.group(2)  # Not used in this example

        model_matches = model_regex.findall(lines[i + 1])
        model_names = [model[0] for model in model_matches]
        model_compilations = [model[1] for model in model_matches]

        benchmark_match = benchmark_regex.search(lines[i + 3])

        if benchmark_match is None:
            import ipdb

            ipdb.set_trace()

        times = benchmark_match.group(1).split()
        times = [float(x) for x in times if x != "|"]

        time_unit_match = time_regex.search(lines[i + 5])

        if time_unit_match is None:
            import ipdb

            ipdb.set_trace()

        time_unit = time_unit_match.group(1)

        try:
            num_datapoints = len(model_names)
            assert num_datapoints == len(model_compilations)
            assert num_datapoints == len(times)

            assert time_unit in ["milliseconds (ms)", "seconds (s)"]
        except:
            import ipdb

            ipdb.set_trace()

        for model_name, model_compilation, time in zip(
            model_names, model_compilations, times
        ):
            if time_unit == "seconds (s)":
                time = time * 1000
            csv_data.append(
                [batch_size, model_name, model_compilation, time, device, component]
            )

        if i + 6 < len(lines) and lines[i + 6] == "":
            i += 7
        else:
            i += 6


filenames = [
    "backbone_4090.txt",
    "backbone_a100.txt",
    "backbone_cpu.txt",
    "backbone_f8_4090.txt",
    "backbone_f8_a100.txt",
    "backbone_f8_t4.txt",
    "backbone_t4.txt",
    "full_4090.txt",
    "full_a100.txt",
    "full_cpu.txt",
    "full_f8_4090.txt",
    "full_f8_a100.txt",
    "full_f8_t4.txt",
    "full_t4.txt",
    "vae_4090.txt",
    "vae_a100.txt",
    "vae_cpu.txt",
    "vae_f8_4090.txt",
    "vae_f8_a100.txt",
    "vae_f8_t4.txt",
    "vae_t4.txt",
]

for filename in filenames:
    splitted = filename.split(".")[0].split("_")
    component = splitted[0]
    device = splitted[-1]

    with open(filename) as f:
        data = f.read()

    print(component, device)

    parse_file(data, device, component)


with open("all.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(csv_data)
