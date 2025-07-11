import json
import matplotlib.pyplot as plt
import os

# List of JSON files to load
filenames = [
    "results/benchmark_redis_query.json",
    "results/benchmark_redis_session.json",
    "results/benchmark_wo_cpu_query.json",
    "results/benchmark_wo_cpu_session.json",
]

for filename in filenames:
    with open(filename, "r") as f:
        data = json.load(f)

    if "query" not in filename:
        continue

    ids = [item["_id"] for item in data]
    if "query" in filename:
        cold_ttft = [item["cold"]["tfft"] for item in data]
        warm_ttft = [item["warm"]["tfft"] for item in data]
    else:
        tfft = [item["tfft"] for item in data]

    plt.figure(figsize=(12, 6))
    if "query" in filename:
        plt.plot(ids, cold_ttft, marker='o', label='Cold TTFT')
        plt.plot(ids, warm_ttft, marker='o', label='Warm TTFT')
    else:
        plt.plot(ids, tfft, marker='o', label='TTFT')
    plt.xlabel('Query ID')
    plt.ylabel('TTFT (seconds)')
    if "query" in filename:
        plt.title(f'TTFT for Cold vs Warm Requests\n({os.path.basename(filename)})')
    else:
        plt.title(f'TTFT for Requests\n({os.path.basename(filename)})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save plot to a file with the same base name but PNG extension
    plot_filename = os.path.splitext(filename)[0] + "_plot.png"
    plt.savefig(plot_filename)
    plt.show()
    plt.close()
