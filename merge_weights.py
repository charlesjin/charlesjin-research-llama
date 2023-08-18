import os
import re
import torch
from tqdm.cli import tqdm

# Which files are merged into one
model_path = "llama-2-13b-chat"
merge_groups = [[0, 1]]

# model_path = "llama-2-70b-chat"
# merge_groups = [[0, 1, 2, 3], [4, 5, 6, 7]]

num_to_merge = sum(len(g) for g in merge_groups)
print(f"Merging {num_to_merge} shards into {len(merge_groups)} shard(s).")

weights = {
    int(fn.split(".")[1]): torch.load(
        f"{model_path}/{fn}", map_location=torch.device("cpu")
    )
    for fn in tqdm(sorted(os.listdir(model_path)))
    if fn.endswith(".pth")
}

# These tensors are duplicated rather than distributed among the files

not_distributed = {
    k
    for k in weights[0].keys()
    if all((weights[0][k] == weights[i][k]).min() for i in range(1, num_to_merge))
}

# What tensor dimensions should be merged, based on whether they are implemented
# as Embedding, Row or Column Parallel.

merge_dimensions = {
    r"^layers.\d+.attention.wq.weight$": 0,
    r"^layers.\d+.attention.wk.weight$": 0,
    r"^layers.\d+.attention.wv.weight$": 0,
    r"^layers.\d+.attention.wo.weight$": 1,
    r"^tok_embeddings.weight$": 1,
    r"^layers.\d+.feed_forward.w1.weight$": 0,
    r"^layers.\d+.feed_forward.w2.weight$": 1,
    r"^layers.\d+.feed_forward.w3.weight$": 0,
    r"^output.weight$": 0,
}

# Merging (or copying if not distributed)
output_weights = {}
for output, group in enumerate(merge_groups):
    output_weights[output] = dict()
    for name in tqdm(weights[group[0]], leave=False):
        if name in not_distributed:
            output_weights[output][name] = weights[0][name]
        else:
            axis = next(
                axis for exp, axis in merge_dimensions.items() if re.match(exp, name)
            )
            output_weights[output][name] = torch.cat(
                [weights[member][name] for member in group], axis=axis
            )

num_nodes = {1: "one", 2: "two", 4: "four", 8: "eight"}[len(output_weights)]
s = "s" if len(output_weights) > 1 else ""
merged_name = f"{model_path}-{num_nodes}-node{s}/"

os.makedirs(merged_name, exist_ok=True)
with open(f"{model_path}/params.json") as fin:
    with open(f"{merged_name}/params.json", "w") as fout:
        fout.write(fin.read())

for idx in range(len(merge_groups)):
    output_weight = output_weights[idx]
    idx = str(idx).zfill(2)
    torch.save(output_weight, f"{merged_name}/consolidated.{idx}.pth")

print(f"Finished writing merged shares to {merged_name}.")
