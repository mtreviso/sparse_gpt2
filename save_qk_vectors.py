import argparse
import random
from collections import defaultdict

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from gpt2_sparse import SparseGPT2LMHeadModel
from gpt_neox_sparse import SparseGPTNeoXForCausalLM
from sparsity_stats import get_sparsity_ratio_for_alpha


def main():
    parser = argparse.ArgumentParser(description="Run a causal language model from Hugging Face Transformers.")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Base model name")
    parser.add_argument("--model_revision", type=str, default="main", help="Model revision (for Pythia models)")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--max_samples", type=int, default=None, help="Max number of validation samples")
    parser.add_argument("--heads_layers_ids", type=str, default=None, help="Layer and head ids to compute sparsity")
    args = parser.parse_args()

    # Load and prepare the dataset
    print("Loading tokenizer and model...")
    if "pythia" in args.model_name:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, revision=args.model_revision)
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        model = SparseGPTNeoXForCausalLM.from_pretrained(
            args.model_name,
            revision=args.model_revision,
            attn_implementation="eager",
            alpha=1.0,
            save_qk=True,
        )

    elif "gpt2" in args.model_name:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        model = SparseGPT2LMHeadModel.from_pretrained(
            args.model_name,
            alpha=1.0,
            save_qk=True,
        )

    else:
        raise ValueError("Invalid model name")

    # Set model to eval mode and move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    # Load the dataset
    print("Loading dataset...")
    # dataset = load_dataset('Skylion007/openwebtext')
    # dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
    # dataset = load_dataset("wikipedia", language="en", date="20220301")
    dataset = load_dataset("kmfoda/booksum", split="validation")

    # filter dataset to only include examples with at least 128 tokens
    print("Dataset size:", len(dataset))
    if args.max_length > 0:
        print(f"Filtering dataset with at least {args.max_length} tokens...")
        dataset = dataset.filter(
            lambda example: len(tokenizer(example['chapter'])['input_ids']) >= args.max_length
        )
        print("Dataset filtered size:", len(dataset))
        if len(dataset) == 0:
            raise ValueError(f"No examples with at least {args.max_length} tokens found in the dataset")

    # Limit the number of samples
    dataset = dataset.select(range(min(args.max_samples, len(dataset)))) if args.max_samples else dataset

    # Tokenize the dataset
    def tokenize_function(example):
        return tokenizer(example['chapter'],
                         padding='max_length', truncation=True, max_length=args.max_length)

    print("Tokenizing dataset...")
    tok_dataset = dataset.map(tokenize_function, batched=True)
    tok_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    dataloader = DataLoader(tok_dataset, batch_size=args.batch_size)

    # Stats
    total_loss = 0
    total_examples = 0
    num_layers = model.config.n_layer if "gpt2" in model.config.model_type else model.config.num_hidden_layers
    num_heads = model.config.n_head if "gpt2" in model.config.model_type else model.config.num_attention_heads
    layer_heads_ids = [(l, h) for l in range(num_layers) for h in range(num_heads)]
    qk_vectors = defaultdict(list)

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Batches", total=len(dataloader)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = input_ids.masked_fill(attention_mask == 0, -100)

            # Perform inference
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            total_loss += loss.item() * input_ids.shape[0]
            total_examples += input_ids.shape[0]

            for l, h in layer_heads_ids:
                if model.config.model_type == "gpt2":
                    layer = model.transformer.h[l].attn
                else:
                    layer = model.gpt_neox.layers[l].attention
                q = layer.q_vectors[:, h].squeeze(0)
                k = layer.k_vectors[:, h].squeeze(0)
                if q.shape[0] > 0 and len(q.shape) == 3:
                    for j in range(q.shape[0]):
                        qk_vectors[f"{l}_{h}"].append((q[j], k[j]))
                else:
                    qk_vectors[f"{l}_{h}"].append((q, k))

    print("Done!")
    print("Total samples:", total_examples)
    print("Total loss:", total_loss / total_examples)

    selected_ids = None
    if args.heads_layers_ids is not None and len(args.heads_layers_ids) > 0:
        # select only a few layers and heads
        selected_ids = set(args.heads_layers_ids.replace(" ", "").split(","))
        qk_vectors = {k: v for k, v in qk_vectors.items() if k in selected_ids}

    for layer_head, qk_list in tqdm(qk_vectors.items(), desc="Computing sparsity with alpha 1.5"):
        if len(qk_list) > 0:
            layer, head = layer_head.split("_")
            sparsity = get_sparsity_ratio_for_alpha(qk_list, alpha=1.5, device=device)
            print(f"Layer {layer} Head {head} Alpha 1.5 sparsity: {sparsity}")

    for layer_head, qk_list in tqdm(qk_vectors.items(), desc="Computing sparsity with alpha 2.0"):
        if len(qk_list) > 0:
            layer, head = layer_head.split("_")
            sparsity = get_sparsity_ratio_for_alpha(qk_list, alpha=2.0, device=device)
            print(f"Layer {layer} Head {head} Alpha 2.0 sparsity: {sparsity}")

    if selected_ids is not None:
        # Save the q, k vectors to file
        filename = f'qk_vectors/qk_2heads_revision{args.model_revision}_maxlen{args.max_length}_maxsamples{args.max_samples}.pt'
        print(f"Saving to {filename}")
        torch.save(qk_vectors, filename)
        print("Done!")


if __name__ == "__main__":
    main()
