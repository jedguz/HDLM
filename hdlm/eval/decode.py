import sys
import json
import codecs
import hydra
import torch
from transformers import AutoTokenizer
import argparse

def main():
    args = argparse.ArgumentParser()
    args.add_argument("--path", type=str, required=True)
    args = args.parse_args()
    # path = sys.argv[1]
    path = hydra.utils.to_absolute_path(args.path)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    zs = torch.load(path, weights_only=True)

    xs = tokenizer.batch_decode(zs, skip_special_tokens=True)

    # xs = [codecs.decode(x, 'unicode_escape') for x in xs]

    print(json.dumps(xs, indent=2))

    output_path = path.replace('.pt', '.json')
    with open(output_path, 'w') as f:
        json.dump(xs, f, indent=2)


if __name__ == "__main__":
    main()
