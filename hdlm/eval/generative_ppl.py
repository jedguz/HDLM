import json
from pathlib import Path

import hydra
import numpy as np
import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def check_for_nan_and_debug(tensor, name, step_info=""):
    """check if the tensor contains NaN and provide debugging information"""
    if torch.isnan(tensor).any():
        print(f"⚠️  NaN detected in {name} at {step_info}")
        print(f"   Shape: {tensor.shape}")
        print(f"   NaN count: {torch.isnan(tensor).sum().item()}")
        print(f"   Min: {tensor.min().item()}, Max: {tensor.max().item()}")
        return True
    return False


def safe_model_forward(model, input_ids, attention_mask, max_retries=3):
    """safe model forward pass, handling NaN issues"""
    
    for attempt in range(max_retries):
        try:
            # check the input
            if check_for_nan_and_debug(input_ids.float(), "input_ids", f"attempt {attempt+1}"):
                print("❌ Input contains NaN, skipping batch")
                return None
                
            if check_for_nan_and_debug(attention_mask.float(), "attention_mask", f"attempt {attempt+1}"):
                print("❌ Attention mask contains NaN, skipping batch")
                return None
            
            # check if the token IDs are in the valid range
            vocab_size = model.config.vocab_size
            if (input_ids >= vocab_size).any():
                print(f"⚠️  Found token IDs >= vocab_size ({vocab_size})")
                # 将超出范围的 token IDs 替换为 unk_token_id
                unk_id = getattr(model.config, 'unk_token_id', 0) or 0
                input_ids = torch.where(input_ids >= vocab_size, unk_id, input_ids)
            
            # use more stable precision
            # with torch.autocast(device_type=input_ids.device.type, enabled=False):
            with torch.cuda.amp.autocast(dtype=torch.float32):
                outputs = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    use_cache=False,
                    return_dict=True
                )
                logits = outputs.logits
            
            # check the output
            if check_for_nan_and_debug(logits, "logits", f"attempt {attempt+1}"):
                if attempt < max_retries - 1:
                    print(f"🔄 Retrying with different settings (attempt {attempt+2})")
                    # try using more stable settings
                    torch.cuda.empty_cache()
                    continue
                else:
                    print("❌ Max retries reached, returning None")
                    return None
            
            # check for extremely large values
            if torch.isinf(logits).any():
                print("⚠️  Infinite values detected in logits")
                logits = torch.where(torch.isinf(logits), 
                                   torch.sign(logits) * 1e6, 
                                   logits)
            
            # limit the range of logits
            logits = torch.clamp(logits, -1e6, 1e6)
            
            return logits
            
        except Exception as e:
            print(f"❌ Error in model forward pass (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                torch.cuda.empty_cache()
                continue
            else:
                return None
    
    return None


@hydra.main(config_path="../configs", config_name="gen_ppl", version_base="1.1")
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    torch.set_grad_enabled(False)
    
    # set more stable numerical settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_tokenizer = AutoTokenizer.from_pretrained(args.model_tokenizer)

    print(f"Loading model {args.pretrained_model}")

    # use more stable loading settings
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model, 
        device_map="auto",
        torch_dtype=torch.float32,  # use float32 to avoid precision issues
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"Model vocab size: {model.config.vocab_size}")
    
    # check if the vocab sizes match
    if len(tokenizer) != model.config.vocab_size:
        print(f"⚠️  Tokenizer and model vocab sizes don't match!")
    
    if args.torch_compile:
        print("⚠️  Skipping torch.compile due to potential NaN issues")
        # model = torch.compile(model)  # skip torch.compile due to potential NaN issues

    samples_path = hydra.utils.to_absolute_path(args.samples_path)
    z_ts = torch.load(samples_path, weights_only=True)
    
    # fix for bug in self-correct script:
    if z_ts.shape[1] == 1:
        z_ts = z_ts.squeeze(1)
    
    print(f"Loaded samples shape: {z_ts.shape}")
    print(f"Sample token range: [{z_ts.min().item()}, {z_ts.max().item()}]")
    
    texts = model_tokenizer.batch_decode(z_ts, skip_special_tokens=True)

    total_acc = 0
    total_nll = 0
    total_tokens = 0
    all_nlls = []
    skipped_batches = 0
    
    with torch.no_grad():
        for i in tqdm.trange(0, len(texts), args.batch_size, desc="Inference", dynamic_ncols=True):
            xs = texts[i:i + args.batch_size]

            try:
                batch = tokenizer(
                    xs, 
                    padding=True, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=512
                ).to(device)
                
                attn_mask = batch["attention_mask"]
                input_ids = batch["input_ids"]
                
                logits = safe_model_forward(model, input_ids, attn_mask)
                
                if logits is None:
                    print(f"⚠️  Skipping batch {i//args.batch_size + 1} due to NaN issues")
                    skipped_batches += 1
                    continue
                

                logits = logits[:, :-1]  # remove the last position
                labels = input_ids[:, 1:]
                loss_mask = attn_mask[:, :-1]

                try:
                    nll = F.cross_entropy(
                        logits.transpose(-1, -2), 
                        labels, 
                        reduction='none'
                    ).view_as(labels)
                    nll = nll.to(torch.float32)
                    
                    if check_for_nan_and_debug(nll, "nll", f"batch {i//args.batch_size + 1}"):
                        print(f"⚠️  Skipping batch {i//args.batch_size + 1} due to NaN in loss")
                        skipped_batches += 1
                        continue
                    
                    # filter valid loss values
                    valid_mask = (loss_mask == 1)
                    valid_nlls = nll[valid_mask]
                    
                    if len(valid_nlls) > 0:
                        all_nlls.extend(valid_nlls.cpu().numpy().tolist())
                        total_nll += (nll * loss_mask).sum().item()

                        acc = (logits.argmax(-1) == labels).float()
                        total_acc += (acc * loss_mask).sum().item()
                        total_tokens += loss_mask.sum().item()
                    
                except Exception as e:
                    print(f"❌ Error in loss calculation for batch {i//args.batch_size + 1}: {e}")
                    skipped_batches += 1
                    continue
                    
            except Exception as e:
                print(f"❌ Error processing batch {i//args.batch_size + 1}: {e}")
                skipped_batches += 1
                continue


    if total_tokens == 0:
        print("❌ No valid tokens processed!")
        return

    nll = total_nll / total_tokens
    ppl = np.exp(total_nll / total_tokens)
    acc = total_acc / total_tokens

    metrics = {
        "file": Path(args.samples_path).stem,
        "pretrained_model": args.pretrained_model,
        "median_nll": np.median(all_nlls) if all_nlls else float('nan'),
        "avg_nll": nll,
        "ppl": ppl,
        "acc": acc,
        "tokens": total_tokens,
        "total_batches": len(texts) // args.batch_size + (1 if len(texts) % args.batch_size else 0),
        "skipped_batches": skipped_batches,
        "success_rate": (1 - skipped_batches / (len(texts) // args.batch_size + 1)) * 100
    }

    print(json.dumps(metrics, indent=4))
    print("=== RESULTS ===")
    print(",".join(map(str, metrics.values())))
    print("===============")
    
    if skipped_batches > 0:
        print(f"⚠️  Warning: {skipped_batches} batches were skipped due to NaN issues")

    with open(hydra.utils.to_absolute_path(args.metrics_path), "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    main()
