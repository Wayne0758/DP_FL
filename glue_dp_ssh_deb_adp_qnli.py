import zipfile
import urllib.request
import os
from datasets import load_from_disk
from packaging import version
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
import warnings
import pandas as pd
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
import torch
import numpy as np
import argparse
from tqdm import tqdm
import torch.nn as nn
import transformers
from torch.utils.data import TensorDataset
from transformers.data.processors.utils import InputExample
from transformers.data.processors.glue import glue_convert_examples_to_features
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)
import adapters
from adapters import BnConfig
from typing import Callable, Dict, Optional, List, Union

warnings.simplefilter("ignore")

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',                          default='0',                type=str)
    parser.add_argument('--dir',                          default='runs',             type=str)
    parser.add_argument('--model_name_or_path',           default='deberta-base',     type=str)
    parser.add_argument('--task_name',                    default='qnli',             type=str)
    parser.add_argument('--tuning_type',                  default='dp-adapter',       type=str)
    parser.add_argument('--learning_rate',                default=1e-3,               type=float)
    parser.add_argument('--num_clients',                  default=10,                  type=int)
    parser.add_argument('--batch_size',                   default=32,                 type=int)
    parser.add_argument('--num_communication_rounds',     default=20,                type=int)
    parser.add_argument('--num_local_updates',            default=5,                 type=int)
    parser.add_argument('--private',                      default=False,              type=bool)
    parser.add_argument('--tensor_rank',                  default=5,                  type=int)
    parser.add_argument('--adapter_dim',                  default=64,                 type=int)
    parser.add_argument('--adapter_alpha',                default=8,                  type=int)
    parser.add_argument('--seed',                         default=42,                 type=int)
    parser.add_argument('--EPSILON',                      default=float("inf"),                  type=float)
    parser.add_argument('--DELTA',                        default=1e-5,               type=float)
    parser.add_argument('--MAX_GRAD_NORM',                default=2,                  type=float)
    parser.add_argument('--LOGGING_INTERVAL',             default=1000,               type=int)
    
    return parser.parse_args()

def main():
    
    our_args = parse()

    set_seed(our_args.seed)
    task_name_map = {
        'sst2': 'sst-2',
        'stsb': 'sts-b',
        'mnli': 'mnli',
        'cola': 'cola',
        'qqp': 'qqp',
        'qnli': 'qnli',
        'rte': 'rte',
        'mrpc': 'mrpc',
    }
    our_args.task_name = task_name_map[our_args.task_name]

    num_labels = glue_tasks_num_labels[our_args.task_name]
    output_mode = glue_output_modes[our_args.task_name]

    config = AutoConfig.from_pretrained(
        f"./deberta-base-config",
        num_labels=num_labels,
        finetuning_task=our_args.task_name,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        f"./deberta-base-tokenizer",
        do_lower_case=False,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        f"./deberta-base-model",
        config=config,
        ignore_mismatched_sizes=True
    )

    if our_args.tuning_type == 'dp-adapter':
        adapters.init(model)
        adapter_config = BnConfig(
            mh_adapter = False,
            output_adapter = True,
            reduction_factor=16,
            non_linearity="gelu"
        )
        model.add_adapter("bottleneck_adapter", adapter_config)
        model.set_active_adapters("bottleneck_adapter")
        for name, param in model.named_parameters():
            if 'bottleneck_adapter' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    

    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst-2": ("sentence", None),
        "sts-b": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }

    def prepare_glue_datasets(task_name):
        """Load, preprocess, and return GLUE datasets."""

        dataset = load_from_disk("./glue_data/"+task_name)
        sentence1_key, sentence2_key = task_to_keys[task_name]

        def tokenize_function(examples):
            texts = (
                (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
            )
            result = tokenizer(*texts, padding='max_length', max_length=128, truncation=True)
            return result
        
        dataset = dataset.map(tokenize_function, batched=True)
        train_features = dataset["train"]
        eval_features = dataset["validation_matched" if task_name == "mnli" else "validation"]

        train_features = train_features.map(lambda examples: {'labels': examples['label']}, batched=True)
        eval_features = eval_features.map(lambda examples: {'labels': examples['label']}, batched=True)

        train_features.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        eval_features.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        return train_features, eval_features
    
    def split_dataset(dataset, num_clients):
        client_datasets = []
        indices = np.arange(len(dataset))
        
        np.random.shuffle(indices)
        
        split_indices = np.array_split(indices, num_clients)
        for client_idx in split_indices:
            client_dataset = dataset.select(client_idx)
            client_datasets.append(client_dataset)
            
        return client_datasets
    

    train_dataset, dev_dataset = prepare_glue_datasets(our_args.task_name)

    client_train_datasets = split_dataset(train_dataset, our_args.num_clients)

    train_dataloader = []
    for client in range(our_args.num_clients):
        train_dataloader.append(torch.utils.data.DataLoader(client_train_datasets[client], batch_size=our_args.batch_size, shuffle=True))
    eval_dataloader = DataLoader(dev_dataset, sampler=SequentialSampler(dev_dataset), batch_size=our_args.batch_size)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model = model.train()
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Trainable: {name}")

    def accuracy(preds, labels):
        return (preds == labels).mean()


    def evaluate(model, test_dataloader):
        model.eval()

        loss_arr = []
        accuracy_arr = []

        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():

                outputs = model(**batch)
                loss, logits = outputs[:2]

                preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
                labels = batch['labels'].detach().cpu().numpy()

                loss_arr.append(loss.item())
                accuracy_arr.append(accuracy(preds, labels))

        model.train()
        return np.mean(loss_arr), np.mean(accuracy_arr)

    os.makedirs("dp", exist_ok=True)
    result_dir = 'dp/' + str(our_args.task_name) + '-' + str(our_args.model_name_or_path.replace('/', '-')) + '-' \
                        + str(our_args.learning_rate)  + '-' + str(our_args.tuning_type) + '_batch_' + str(our_args.batch_size) \
                              + '_num_clients_' + str(our_args.num_clients) + '_num_local_updates_' + str(our_args.num_local_updates) \
                                + '_EPS_' + str(our_args.EPSILON) + '_DEL_' + str(our_args.DELTA) + '_MAX_' + str(our_args.MAX_GRAD_NORM) \
                                    + '_private_' + str(our_args.private)  + '.txt'
    
    # our_args.DELTA = 1 / len(train_dataloader)
    for epoch in range(1, our_args.num_communication_rounds+1):
        losses = []
        local_weights = []
        num_samples_per_client = []
        for client in range(our_args.num_clients):

            local_model = AutoModelForSequenceClassification.from_pretrained(
                f"./deberta-base-model",
                config=config,
                ignore_mismatched_sizes=True
            )
            adapters.init(local_model)
            local_model.add_adapter("bottleneck_adapter", adapter_config)
            local_model.set_active_adapters("bottleneck_adapter")
            for name, param in local_model.named_parameters():
                if 'bottleneck_adapter' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            local_model.load_state_dict(model.state_dict())
            optimizer = torch.optim.AdamW(local_model.parameters(), lr=our_args.learning_rate)
            local_model.train()

            if our_args.private == True:

                privacy_engine = PrivacyEngine()
                model_private, optimizer_private, train_dataloader[client] = privacy_engine.make_private_with_epsilon(
                    module=local_model,
                    optimizer=optimizer,
                    data_loader=train_dataloader[client],
                    target_delta=our_args.DELTA,
                    target_epsilon=our_args.EPSILON,
                    epochs=our_args.num_local_updates,
                    max_grad_norm=our_args.MAX_GRAD_NORM,
                )
            else:
                model_private = local_model
                optimizer_private = optimizer
            model_private = model_private.to(device)
            model_private = model_private.train()

            
            best_loss = float('inf')  
            best_model_state = None 
            for LU in range(1, our_args.num_local_updates+1):
                if our_args.EPSILON == float("inf"):
                    memory_safe_data_loader = train_dataloader[client]
                    for step, batch in enumerate(tqdm(memory_safe_data_loader)):
                        optimizer_private.zero_grad()
                        batch = {k: v.to(device) for k, v in batch.items()}
                        outputs = model_private(**batch) 
                        loss = outputs[0]
                        loss.backward()
                        losses.append(loss.item())
                        optimizer_private.step()

                        if step > 0 and step % our_args.LOGGING_INTERVAL == 0:
                            train_loss = np.mean(losses)
                            eps = privacy_engine.get_epsilon(our_args.DELTA)

                            eval_loss, eval_accuracy = evaluate(model_private, eval_dataloader)

                            print(
                            f"Client: {client} | "
                            f"Local Updates: {LU} | "
                            f"Epoch: {epoch} | "
                            f"Step: {step} | "
                            f"Train loss: {train_loss:.3f} | "
                            f"Eval loss: {eval_loss:.3f} | "
                            f"Eval accuracy: {eval_accuracy:.3f} | "
                            f"ɛ: {eps:.2f}"
                            )
                eval_loss, eval_accuracy = evaluate(model_private, eval_dataloader)
                if eval_loss < best_loss:
                    best_loss = eval_loss
                    best_model_state = model_private.state_dict() 
            if best_model_state is not None:
                model_private.load_state_dict(best_model_state)

            peft_weights = {
                key.replace('_module.', '', 1): value
                for key, value in model_private.named_parameters()
                if value.requires_grad == True
            }
            local_weights.append(peft_weights)
            num_samples_per_client.append(len(client_train_datasets[client]))
        
        total_samples = sum(num_samples_per_client)
        global_peft_weights = {}
        for key in local_weights[0].keys(): 
            global_peft_weights[key] = sum(
                (num_samples_per_client[i] / total_samples) * local_weights[i][key]
                for i in range(our_args.num_clients)
            )
        global_state_dict = model.state_dict()
        global_state_dict.update(global_peft_weights)
        model.load_state_dict(global_state_dict)


        score = evaluate(model, eval_dataloader)

        with open(result_dir, "a") as myfile:
            myfile.write(f"{score}\n")


if __name__ == "__main__":
    main()
