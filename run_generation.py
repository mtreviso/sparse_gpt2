from collections import defaultdict

import torch
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, GPT2Config, AutoTokenizer
from datasets import load_dataset
import pytorch_lightning as pl
import argparse
import random

from gpt2_sparse import SparseGPT2LMHeadModel
from gpt_neox_sparse import SparseGPTNeoXForCausalLM
from sparsity_stats import get_sparsity_ratio_for_alpha


class CausalLanguageModel(pl.LightningModule):
    def __init__(
        self,
        model,
        learning_rate=2e-5,
        save_epochs=(1, 5, 10),
        num_heads_layers=10,
        entmax_alpha=1.5,
        revision_step="main",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = model
        self.learning_rate = learning_rate
        self.save_epochs = save_epochs
        self.num_heads_layers = num_heads_layers
        self.qk_vectors = defaultdict(list)
        self.val_losses = []
        self.entmax_alpha = entmax_alpha
        self.revision_step = revision_step

        if "gpt2" in self.model.config.model_type:
            num_layers = self.model.config.n_layer
            num_heads = self.model.config.n_head
        else:
            num_layers = self.model.config.num_hidden_layers
            num_heads = self.model.config.num_attention_heads

        self.layer_heads_ids = [(l, h) for l in range(num_layers) for h in range(num_heads)]
        # random.shuffle(self.layer_heads_ids)
        # self.layer_heads_ids = self.layer_heads_ids[:self.num_heads_layers]

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120_000, eta_min=6e-5),
                "monitor": "val_loss",
                "interval": "step",
            },
        }

    def forward(self, input_ids, labels=None):
        output = self.model(input_ids, labels=labels)
        return output.loss, output.hidden_states

    def training_step(self, batch, batch_idx):
        input_ids, labels = batch['input_ids'], batch['labels']
        loss, _ = self(input_ids, labels=labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        input_ids, labels = batch['input_ids'], batch['labels']
        loss, _ = self(input_ids, labels=labels)

        if self.current_epoch + 1 in self.save_epochs:
            first_key = "{}_{}".format(self.layer_heads_ids[0][0], self.layer_heads_ids[0][1])
            for l, h in self.layer_heads_ids:
                if self.model.config.model_type == "gpt2":
                    layer = self.model.transformer.h[l].attn
                else:
                    layer = self.model.gpt_neox.layers[l].attention
                q = layer.q_vectors[:, h].squeeze(0)
                k = layer.k_vectors[:, h].squeeze(0)
                self.qk_vectors[f"{l}_{h}"].append((q, k))

        # Log validation loss
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.val_losses.append(loss)

        return {'val_loss': loss}

    def on_train_epoch_end(self):
        self.qk_vectors.clear()

    def on_validation_epoch_end(self):
        if len(self.qk_vectors) > 0:
            # print stats
            print("Total samples:", len(self.qk_vectors["0_0"]))
            for layer_head, qk_list in self.qk_vectors.items():
                if len(qk_list) > 0:
                    layer, head = layer_head.split("_")
                    sparsity = get_sparsity_ratio_for_alpha(qk_list, alpha=1.5)
                    print(f"Layer {layer} Head {head} Alpha 1.5 sparsity: {sparsity}")

            for layer_head, qk_list in self.qk_vectors.items():
                if len(qk_list) > 0:
                    layer, head = layer_head.split("_")
                    sparsity = get_sparsity_ratio_for_alpha(qk_list, alpha=2.0)
                    print(f"Layer {layer} Head {head} Alpha 2.0 sparsity: {sparsity}")

            # Save the q, k vectors to file
            # print(f"\nSaving q, k vectors for epoch {self.current_epoch + 1}\n")
            # filename = f'qk_vectors/qk_epoch{self.current_epoch + 1}_alpha{self.entmax_alpha}_{self.revision_step}.pt'
            # torch.save(self.qk_vectors, filename)
            # print(f"Saved q, k vectors to {filename}\n")

        self.qk_vectors.clear()

        # Log average validation loss
        avg_loss = torch.stack(self.val_losses).mean()
        self.log('avg_val_loss', avg_loss)
        self.val_losses = []


def main():
    parser = argparse.ArgumentParser(
        description="Train a causal language model with PyTorch Lightning and Hugging Face Transformers.")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Base model name")
    parser.add_argument("--model_revision", type=str, default="main", help="Model revision (for Pythia models)")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--logger", action="store_true", help="Use Weights & Biases logger")
    parser.add_argument("--max_training_samples", type=int, default=None, help="Max number of training samples")
    parser.add_argument("--max_validation_samples", type=int, default=None, help="Max number of validation samples")
    parser.add_argument("--entmax_alpha", type=float, default=1.0, help="Entmax alpha parameter (1.0, 1.5, 2.0)")
    parser.add_argument("--num_heads_layers", type=int, default=10, help="Num of layers and heads to save q, k vectors")
    parser.add_argument("--save_epochs", type=int, nargs="+", default=[], help="Epochs to save q, k vectors")
    parser.add_argument("--reinit_weights", action="store_true", help="Reinitialize model weights")
    parser.add_argument("--from_pretrained", action="store_true", help="Initialize from pretrained weights")

    args = parser.parse_args()

    # Initialize Weights & Biases logger
    wandb_logger = None
    if args.logger:
        wandb_logger = WandbLogger(project="causal-language-model", log_model="all")

    # Load and prepare the dataset
    if "pythia" in args.model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            revision=args.model_revision,
        )
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        model = SparseGPTNeoXForCausalLM.from_pretrained(
            args.model_name,
            revision=args.model_revision,
            attn_implementation="eager",
            alpha=args.entmax_alpha,
            save_qk=True,
            reinit_weights=args.reinit_weights
        )

    elif "gpt2" in args.model_name:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

        if args.from_pretrained:
            # For using a pre-trained model
            model = SparseGPT2LMHeadModel.from_pretrained(
                args.model_name,
                alpha=args.entmax_alpha,
                save_qk=True,
                reinit_weights=args.reinit_weights
            )
        else:
            # For training a model from scratch
            config = GPT2Config(n_positions=args.max_length)
            model = SparseGPT2LMHeadModel(
                config,
                alpha=args.entmax_alpha,
                save_qk=True,
                reinit_weights=args.reinit_weights
            )
    else:
        raise ValueError("Invalid model name")

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=args.max_length)

    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.map(lambda examples: {'labels': examples['input_ids']}, batched=True)
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'labels'])

    train_dataset = tokenized_datasets['train']
    val_dataset = tokenized_datasets['validation']

    if args.max_training_samples:
        train_dataset = train_dataset.select(range(args.max_training_samples))
    if args.max_validation_samples:
        val_dataset = val_dataset.select(range(args.max_validation_samples))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    model = CausalLanguageModel(
        model=model,
        learning_rate=args.learning_rate,
        save_epochs=args.save_epochs,
        num_heads_layers=args.num_heads_layers,
        entmax_alpha=args.entmax_alpha,
        revision_step=args.model_revision
    )
    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=args.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'auto',
        gradient_clip_val=1.0,  # Clip gradients to avoid exploding gradients
        check_val_every_n_epoch=1,  # Set to 1 to validate at every epoch
        val_check_interval=1.0,  # Validate 1 time per epoch,
        # limit_train_batches=0.0,  # Set to 0.1 for debugging
        enable_checkpointing=False,  # Disable checkpointing
    )
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
