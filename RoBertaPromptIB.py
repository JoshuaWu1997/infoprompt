import random
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from utils.import_glue import params, batch_data, score_func, encode
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from transformers import get_linear_schedule_with_warmup

assert params.model_path.startswith('RoBertaPrompt')
from model.RoBertaPrompt import BertDIBPromptRE as Model

# ---------------------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------------------

# Seed rng for reproducibility
random.seed(params.seed)
torch.manual_seed(params.seed)
if params.cuda:
    torch.cuda.manual_seed_all(params.seed)
device = 'cuda' if params.cuda else 'cpu'

if __name__ == '__main__':
    # Setup model
    model = Model(params).to(device)

    # Set Frozen
    print("Turning off gradients in both the encoder and the encoder")
    unfrozen_layers = ["lm", "prompt"]
    for name, param in model.named_parameters():
        if not any([layer in name for layer in unfrozen_layers]):
            print("[FROZE]: %s" % name)
            param.requires_grad = False
        else:
            print("[FREE]: %s" % name)
            param.requires_grad = True

    # Setup dataloader
    train_dataset = load_dataset('glue', params.task, split='train', cache_dir='./cache')
    train_dataset = train_dataset.map(encode, batched=True, num_proc=8)
    train_dataset = train_dataset.map(lambda examples: {'labels': examples['label']}, batched=True, num_proc=8)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)

    valid_dataset = load_dataset('glue', params.task, split='validation', cache_dir='./cache')
    valid_dataset = valid_dataset.map(encode, batched=True, num_proc=8)
    valid_dataset = valid_dataset.map(lambda examples: {'labels': examples['label']}, batched=True, num_proc=8)
    valid_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=params.batch_test, shuffle=False)

    # Set Optimizer
    num_train_optimization_steps = (len(train_dataset) // params.batch_size) * params.epochs
    warmup_steps = int(num_train_optimization_steps * 0.06) if not params.continued else 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=num_train_optimization_steps)

    # Setup global variable
    best_score = 0

    # Logging
    fprefix = f'{params.output_dir}/{params.model_path}[{params.n_latent}]-{params.task}-{params.n_prompt}-{params.decay}/'
    os.makedirs(fprefix, exist_ok=True)
    if params.continued:
        print(f'Load parameters:', fprefix + f'best_tune.pt')
        model.tune.load_state_dict(torch.load(fprefix + f'best_tune.pt'))

    # ---------------------------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------------------------
    for epoch in range(params.epochs):
        preds, labels, losses, add_losses = [], [], [], []
        for i, batch in enumerate(tqdm(train_loader)):
            batch_null = {k: v.clone() for k, v in batch.items()}
            batch_null = {k: v.to(device) for k, v in batch_data(model.dict, batch_null, 0).items()}
            batch = {k: v.to(device) for k, v in batch_data(model.dict, batch, params.n_prompt).items()}
            labels.extend(batch['labels'].tolist())
            main_loss, add_loss, logits = model(batch, batch_null, 'train')
            loss = main_loss + add_loss * (1 - params.decay * (epoch + 1) / params.epochs)
            _, predicted = torch.max(logits.detach().data, 1)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            preds.extend(predicted.tolist())
            losses.append(main_loss.item())
            add_losses.append(add_loss.item())