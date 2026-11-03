"""Export a trained checkpoint for release.

Writes OUT_DIR/model.pt (the EMA weights when available, otherwise the model
weights), OUT_DIR/args.json, and the Mean.npy/Std.npy the model was trained with,
so generate.py can use the folder on its own.
"""

import argparse
import json
import os
import shutil

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True, help="save/RUN/modelNNNNNNNNN.pt")
    parser.add_argument("--mean", required=True, help="Mean.npy used to train the model.")
    parser.add_argument("--std", required=True, help="Std.npy used to train the model.")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.model_path, map_location="cpu")
    state_dict = checkpoint.get("model_avg") or checkpoint.get("model") or checkpoint
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(state_dict, os.path.join(args.out_dir, "model.pt"))

    with open(os.path.join(os.path.dirname(args.model_path), "args.json")) as f:
        model_args = json.load(f)
    model_args["save_dir"] = args.out_dir
    with open(os.path.join(args.out_dir, "args.json"), "w") as f:
        json.dump(model_args, f, indent=4, sort_keys=True)

    for src, name in ((args.mean, "Mean.npy"), (args.std, "Std.npy")):
        assert np.load(src).shape == (407,), f"{src} is not a 407-dim feature statistic"
        shutil.copy2(src, os.path.join(args.out_dir, name))
    print(f"Exported {args.model_path} to {args.out_dir}")


if __name__ == "__main__":
    main()
