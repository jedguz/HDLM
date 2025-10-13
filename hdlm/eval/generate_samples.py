import hydra
import tqdm
import torch
import os
import sys

# Add the project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


from hdlm.utils import parse_dtype
from hdlm.checkpoints import load_checkpoint
from hdlm.sampling import get_sampler


@hydra.main(config_path="../configs", config_name="hsdd-base-cluster_128-generate", version_base="1.1")
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    torch.set_grad_enabled(False)

    print(f"Generating {args.num_samples} samples from {args.path}")

    ckpt_path = hydra.utils.to_absolute_path(args.path)
    config = args
    torch.manual_seed(args.training.get("seed", 0))

    model, noise_schedule, tokenizer, old_config = load_checkpoint(ckpt_path, device=device)
    model.eval()
    old_config.training.eval_batch_size = args.batch_size
    dtype = parse_dtype(old_config.training.dtype)

    sampler = get_sampler(old_config, model, tokenizer, noise_schedule, min_p=args.min_p, new_config=config)
    model.eval()

    samples = []
    with tqdm.tqdm(total=args.num_samples, desc="Sampling", dynamic_ncols=True) as pbar:
        with torch.no_grad(), torch.autocast(device.type, dtype=dtype):
            for i in range(0, args.num_samples, args.batch_size):
                bs = min(args.batch_size, args.num_samples - i)
                z_t = sampler.generate(bs, args.num_denoising_steps, decode=False, show_progress=False)
                samples.append(z_t)
                pbar.update(bs)
    samples = torch.cat(samples, dim=0).cpu()

    torch.save(samples, hydra.utils.to_absolute_path(args.samples_path))


if __name__ == "__main__":
    main()
