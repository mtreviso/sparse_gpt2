import torch
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from datasets import load_dataset
import pytorch_lightning as pl
import argparse

from gpt2_sparse import SparseGPT2LMHeadModel


class CausalLanguageModel(pl.LightningModule):
    def __init__(
        self,
        model_name='gpt2',
        learning_rate=2e-5,
        save_epochs=(1, 5, 10),
        layer_num=6,
        head_num=0,
        max_batches=1000,
        entmax_alpha=1.5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = SparseGPT2LMHeadModel.from_pretrained(model_name, alpha=entmax_alpha, save_qk=True)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.learning_rate = learning_rate
        self.save_epochs = save_epochs
        self.layer_num = layer_num
        self.head_num = head_num
        self.max_batches = max_batches
        self.qk_vectors = []
        self.val_losses = []

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def forward(self, input_ids, labels=None):
        output = self.model(input_ids, labels=labels)
        return output.loss, output.hidden_states

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        labels = batch['labels']
        loss, _ = self(input_ids, labels=labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        input_ids, labels = batch['input_ids'], batch['labels']
        loss, hidden_states = self(input_ids, labels=labels)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if self.current_epoch + 1 in self.save_epochs:
            if len(self.qk_vectors) < self.max_batches:  # Accumulate q, k vectors up to 1000 samples
                layer = self.model.transformer.h[self.layer_num].attn
                q = layer.q_vectors[:, self.head_num]
                k = layer.k_vectors[:, self.head_num]
                self.qk_vectors.append((q, k))

        self.val_losses.append(loss)

        return {'val_loss': loss}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_losses).mean()
        self.log('avg_val_loss', avg_loss)

        if len(self.qk_vectors) > 0:
            # Ensure not to exceed 1000 samples
            self.qk_vectors = self.qk_vectors[:1000]
            # Save the q, k vectors to file
            torch.save(
                self.qk_vectors,
                f'qk_vectors/qk_epoch{self.current_epoch + 1}_layer{self.layer_num}_head{self.head_num}.pt'
            )

        self.val_losses = []  # Clear the buffer for the next validation epoch
        self.qk_vectors = []  # Clear the buffer for the next validation epoch


def main():
    parser = argparse.ArgumentParser(
        description="Train a causal language model with PyTorch Lightning and Hugging Face Transformers.")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Base model name")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--logger", action="store_true", help="Use Weights & Biases logger")
    parser.add_argument("--max_training_samples", type=int, default=None, help="Max number of training samples")
    parser.add_argument("--max_validation_samples", type=int, default=None, help="Max number of validation samples")
    parser.add_argument("--entmax_alpha", type=float, default=1.5, help="Entmax alpha parameter")
    parser.add_argument("--save_epochs", type=int, nargs="+", default=[], help="Epochs to save q, k vectors")
    parser.add_argument("--save_layer_num", type=int, default=6, help="Layer number to save q, k vectors")
    parser.add_argument("--save_head_num", type=int, default=0, help="Head number to save q, k vectors")
    parser.add_argument("--save_max_batches", type=int, default=1000, help="Max number of batches to save q, k vectors")
    args = parser.parse_args()

    # Initialize Weights & Biases logger
    wandb_logger = None
    if args.logger:
        wandb_logger = WandbLogger(project="causal-language-model", log_model="all")

    # Load and prepare the dataset
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='longest', truncation=True, max_length=args.max_length)

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
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        save_epochs=args.save_epochs,
        layer_num=args.save_layer_num,
        head_num=args.save_head_num,
        max_batches=args.save_max_batches,
        entmax_alpha=args.entmax_alpha,
    )
    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=args.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'auto'
    )
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
