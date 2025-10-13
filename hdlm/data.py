import functools
import hashlib
import json
import os
import re
import shutil
from typing import Callable

import numpy as np
import torch
import hydra
from transformers import BatchEncoding, PreTrainedTokenizer
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def get_dataset(config, num_proc=32):
    n_proc = min(os.cpu_count(), num_proc)
    
    if "openwebtext" in config.data.dataset_name:
        test_size = int(config.data.test_size)
        train_ds = load_dataset(
            config.data.dataset_name,
            config.data.dataset_subset,
            split=f"train[:-{test_size}]",
            trust_remote_code=config.data.trust_remote_code,
            num_proc=n_proc,
            cache_dir=hydra.utils.to_absolute_path(config.data.cache_dir),
        )
        test_ds = load_dataset(
            config.data.dataset_name,
            config.data.dataset_subset,
            split=f"train[-{test_size}:]",
            trust_remote_code=config.data.trust_remote_code,
            num_proc=n_proc,
            cache_dir=hydra.utils.to_absolute_path(config.data.cache_dir),
        )
    elif config.data.dataset_name == "lm1b":
        # Handle lm1b dataset similar to Score-Entropy-Discrete-Diffusion
        dataset = load_dataset(
            "lm1b",
            cache_dir=hydra.utils.to_absolute_path(config.data.cache_dir),
            streaming=False,
            trust_remote_code=config.data.trust_remote_code,
            num_proc=n_proc,
        )
        
        # For lm1b, we use the standard train/test split
        train_ds = dataset["train"]
        test_ds = dataset["test"]
        
        # Optional: subsample for faster experimentation if specified in config
        if hasattr(config.data, 'subsample_train') and config.data.subsample_train > 0:
            train_ds = train_ds.select(range(min(config.data.subsample_train, len(train_ds))))
        if hasattr(config.data, 'subsample_test') and config.data.subsample_test > 0:
            test_ds = test_ds.select(range(min(config.data.subsample_test, len(test_ds))))
            
    else:
        dataset = load_dataset(
            config.data.dataset_name,
            config.data.dataset_subset,
            cache_dir=hydra.utils.to_absolute_path(config.data.cache_dir),
            streaming=False,
            trust_remote_code=config.data.trust_remote_code,
            num_proc=n_proc,
        )
        train_ds = dataset["train"]
        test_ds = dataset["test"]

    return train_ds, test_ds


def cached_dataset(cache_dir: str, file_name: str, generate_fn: Callable[[], Dataset]) -> Dataset:
    if cache_dir is None:
        return generate_fn()

    cache_path = os.path.join(cache_dir, file_name)
    if os.path.exists(cache_path):
        ds = Dataset.load_from_disk(cache_path)
        return ds
    else:
        ds = generate_fn()
        os.makedirs(cache_dir, exist_ok=True)
        try:
            ds.save_to_disk(cache_path)
        except Exception as e:
            shutil.rmtree(cache_path)
            raise e
        return ds


def lm1b_detokenizer(x):
  x = x.replace('http : / / ', 'http://')
  x = x.replace('https : / / ', 'https://')
  x = re.sub(r' \'(\w+)', r"'\1", x)
  x = re.sub(r' (\w+) \. ', r' \1. ', x)
  x = re.sub(r' (\w+) \.$', r' \1.', x)
  x = x.replace(' ? ', '? ')
  x = re.sub(r' \?$', '?', x)
  x = x.replace(' ! ', '! ')
  x = re.sub(r' \!$', '!', x)
  x = x.replace(' , ', ', ')
  x = x.replace(' : ', ': ')
  x = x.replace(' ; ', '; ')
  x = x.replace(' / ', '/')
  x = re.sub(r'\" ([^\"]+) \"', r'"\1"', x)
  x = re.sub(r'\' ([^\']+) \'', r"'\1'", x)
  x = re.sub(r'\( ([^\(\)]+) \)', r"(\1)", x)
  x = re.sub(r'\[ ([^\[\]]+) \]', r"[\1]", x)
  x = x.replace('$ ', '$')
  x = x.replace('£ ', '£')
  return x

def tokenize_dataset(
    ds: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_seq_len: int = 512,
    sequence_packing: bool = False,
    batch_size: int = 1024,
    num_proc: int = 32,
    text_key: str = "text",
    detokenizer: Callable = None
):
    n_proc = min(os.cpu_count(), num_proc)
    bos_token_id = tokenizer.bos_token_id or tokenizer.cls_token_id
    eos_token_id = tokenizer.eos_token_id or tokenizer.sep_token_id

    tokenizer_max_len = tokenizer.model_max_length
    tokenizer.model_max_length = 10_000_000

    def tokenize_fn(examples):
        # Handle different text field names (lm1b uses "sentence", others use "text")
        text_field = text_key if text_key in examples else "sentence" if "sentence" in examples else "text"
        text = examples[text_field]
        if detokenizer is not None:
            for i, t in enumerate(text):
                text[i] = detokenizer(t)
        tokens = tokenizer(
            text,
            truncation=False,
            padding=False,
        )["input_ids"]
        tokens = [[bos_token_id] + x + ([] if sequence_packing else [eos_token_id]) for x in tokens]
        if sequence_packing:
            tokens = np.concatenate(tokens, axis=0)
            tokens = tokens[: len(tokens) - len(tokens) % max_seq_len]
            tokens = tokens.reshape(-1, max_seq_len)
        else:
            tokens = [
                np.pad(x, (0, max_seq_len - len(x) % max_seq_len), mode="constant", constant_values=tokenizer.pad_token_id)
                for x in tokens
            ]
            tokens = [x.reshape(-1, max_seq_len) for x in tokens]
            tokens = np.concatenate(tokens, axis=0)
        return {"input_ids": tokens}

    # Determine which columns to remove based on the dataset
    columns_to_remove = []
    if "text" in ds.column_names:
        columns_to_remove.append("text")
    if "sentence" in ds.column_names:
        columns_to_remove.append("sentence")

    ds = ds.map(
        tokenize_fn,
        batched=True,
        batch_size=batch_size,
        remove_columns=columns_to_remove,
        num_proc=n_proc,
    )

    tokenizer.model_max_length = tokenizer_max_len
    return ds


def default_collator(config, tokenizer, examples, text_key="text"):
    examples = [x[text_key] for x in examples]
    return tokenizer(examples, padding="max_length", truncation=True, max_length=config.model.max_seq_len, return_tensors="pt")


def pretokenized_collator(examples, pad_token_id=0, tokens_key="input_ids"):
    input_ids = np.stack([np.array(x[tokens_key]) for x in examples], axis=0)
    attn_masks = (input_ids != pad_token_id).astype(np.int32)
    input_ids = torch.from_numpy(input_ids).to(torch.long)
    attn_masks = torch.from_numpy(attn_masks).to(torch.long)
    return BatchEncoding({"input_ids": input_ids, "attention_mask": attn_masks}, tensor_type="pt", n_sequences=len(input_ids))


def subsample_collator(config, tokenizer, examples, text_key="text"):
    bos_token_id = tokenizer.bos_token_id or tokenizer.cls_token_id
    eos_token_id = tokenizer.eos_token_id or tokenizer.sep_token_id

    examples = [x[text_key] for x in examples]
    tokens = tokenizer(examples, truncation=False, return_tensors="np")
    max_length = config.model.max_seq_len
    input_ids = []
    attn_masks = []
    for i in range(len(examples)):
        toks = tokens["input_ids"][i]
        attn_mask = tokens["attention_mask"][i]
        if toks[0] != bos_token_id:
            toks = np.concatenate([[bos_token_id], toks])
            attn_mask = np.concatenate([[1], attn_mask])
        if toks[-1] != eos_token_id:
            toks = np.concatenate([toks, [eos_token_id]])
            attn_mask = np.concatenate([attn_mask, [1]])

        if len(toks) > max_length:
            overflow = len(toks) - max_length
            start_idx = np.random.randint(0, overflow + config.data.max_add_padding)
            toks = toks[start_idx : start_idx + max_length]
            attn_mask = attn_mask[start_idx : start_idx + max_length]
        if len(toks) < max_length:
            underflow = max_length - len(toks)
            toks = np.pad(toks, (0, underflow), mode="constant", constant_values=tokenizer.pad_token_id)
            attn_mask = np.pad(attn_mask, (0, underflow), mode="constant", constant_values=0)
        assert len(toks) == max_length
        assert len(attn_mask) == max_length
        input_ids.append(toks)
        attn_masks.append(attn_mask)
    input_ids = torch.from_numpy(np.array(input_ids)).to(torch.long)
    attn_masks = torch.from_numpy(np.array(attn_masks)).to(torch.long)
    return BatchEncoding({"input_ids": input_ids, "attention_mask": attn_masks}, tensor_type="pt", n_sequences=len(input_ids))


def split_sequence_collator(config, tokenizer, examples, text_key="text"):
    bos_token_id = tokenizer.bos_token_id or tokenizer.cls_token_id
    eos_token_id = tokenizer.eos_token_id or tokenizer.sep_token_id
    max_length = config.model.max_seq_len

    # Tokenize all examples
    examples = [x[text_key] for x in examples]
    tokens = tokenizer(examples, truncation=False, return_tensors="np")

    # Process each example
    all_chunks = []
    for i in range(len(examples)):
        toks = tokens["input_ids"][i]
        attn_mask = tokens["attention_mask"][i]

        # Add BOS and EOS tokens
        if toks[0] != bos_token_id:
            toks = np.concatenate([[bos_token_id], toks])
            attn_mask = np.concatenate([[1], attn_mask])
        if toks[-1] != eos_token_id:
            toks = np.concatenate([toks, [eos_token_id]])
            attn_mask = np.concatenate([attn_mask, [1]])

        # Split into chunks of max_length
        n_chunks = len(toks) // max_length
        if n_chunks > 0:
            # Reshape into chunks, dropping the last incomplete chunk
            chunks = toks[:n_chunks * max_length].reshape(-1, max_length)
            mask_chunks = attn_mask[:n_chunks * max_length].reshape(-1, max_length)
            all_chunks.extend([(chunk, mask) for chunk, mask in zip(chunks, mask_chunks)])

    # If we have more chunks than batch_size, randomly select batch_size chunks
    if len(all_chunks) > config.training.train_batch_size:
        indices = np.random.choice(len(all_chunks), config.training.train_batch_size, replace=False)
        all_chunks = [all_chunks[i] for i in indices]

    # Stack the chunks
    input_ids = torch.from_numpy(np.stack([chunk for chunk, _ in all_chunks])).to(torch.long)
    attn_masks = torch.from_numpy(np.stack([mask for _, mask in all_chunks])).to(torch.long)
    print(input_ids.shape, attn_masks.shape)

    return BatchEncoding(
        {"input_ids": input_ids, "attention_mask": attn_masks},
        tensor_type="pt",
        n_sequences=len(input_ids)
    )


def _get_dataloader(config, ds, shuffle, drop_last, batch_size, collate_fn):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        sampler = DistributedSampler(ds, seed=config.training.seed, shuffle=shuffle)
        _shuffle = False
    else:
        sampler = None
        _shuffle = shuffle

    return DataLoader(
        ds,
        collate_fn=collate_fn,
        batch_size=batch_size,
        drop_last=drop_last,
        sampler=sampler,
        num_workers=config.data.num_workers,
        shuffle=_shuffle,
        pin_memory=True,
        persistent_workers=True,
    )


def get_dataloaders(config, tokenizer, train_batch_size=None, eval_batch_size=None):
    if train_batch_size is None:
        train_batch_size = config.training.train_batch_size
    if eval_batch_size is None:
        eval_batch_size = config.training.eval_batch_size

    train_ds, test_ds = get_dataset(config)

    if config.data.pre_tokenize:
        max_seq_len = config.model.max_seq_len
        sequence_packing = config.data.sequence_packing
        
        # Determine text key based on dataset
        text_key = "sentence" if config.data.dataset_name == "lm1b" else "text"
        
        cache_key = hashlib.sha256(
            json.dumps(
                {
                    "dataset_name": config.data.dataset_name,
                    "subset": config.data.dataset_subset,
                    "tokenizer_name": config.data.tokenizer_name,
                    "max_seq_len": max_seq_len,
                    "sequence_packing": sequence_packing,
                    "text_key": text_key,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        detokenizer = lm1b_detokenizer if config.data.dataset_name == "lm1b" else None
        train_ds = cached_dataset(
            cache_dir=hydra.utils.to_absolute_path(config.data.cache_dir),
            file_name=f"cache-{config.data.dataset_name.replace('/', '--')}-train-{cache_key}",
            generate_fn=functools.partial(tokenize_dataset, ds=train_ds, tokenizer=tokenizer, max_seq_len=max_seq_len, sequence_packing=sequence_packing, text_key=text_key, detokenizer=detokenizer),
        )
        test_ds = cached_dataset(
            cache_dir=hydra.utils.to_absolute_path(config.data.cache_dir),
            file_name=f"cache-{config.data.dataset_name.replace('/', '--')}-test-{cache_key}",
            generate_fn=functools.partial(tokenize_dataset, ds=test_ds, tokenizer=tokenizer, max_seq_len=max_seq_len, sequence_packing=sequence_packing, text_key=text_key, detokenizer=detokenizer),
        )

        collate_fn = functools.partial(pretokenized_collator, pad_token_id=tokenizer.pad_token_id, tokens_key="input_ids")
    else:
        if config.data.sequence_packing:
            raise ValueError("Sequence packing requires pre-tokenization.")

        # Determine text key based on dataset
        text_key = "sentence" if config.data.dataset_name == "lm1b" else "text"

        if config.data.get("split_sequences", False):
            collate_fn = functools.partial(split_sequence_collator, config, tokenizer, text_key=text_key)
        else:
            collate_fn = functools.partial(subsample_collator, config, tokenizer, text_key=text_key)

    train_dl = _get_dataloader(config, train_ds, shuffle=True, drop_last=True, batch_size=train_batch_size, collate_fn=collate_fn)
    test_dl = _get_dataloader(config, test_ds, shuffle=False, drop_last=False, batch_size=eval_batch_size, collate_fn=collate_fn)

    return train_dl, test_dl
