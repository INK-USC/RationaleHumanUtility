from decimal import InvalidContext
import os
from pathlib import Path
from copy import deepcopy

import pickle5 as pickle
from tqdm import tqdm

from hydra.utils import get_original_cwd

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
from transformers import AutoTokenizer, GPT2TokenizerFast

from src.utils.data import data_keys


class DataModule(pl.LightningDataModule):

    def __init__(self,
                 dataset: str, data_path: str, arch: str, mode: str,
                 train_batch_size: int = 1, eval_batch_size: int = 1, eff_train_batch_size: int = 1, num_workers: int = 0,
                 num_train: int = None, num_dev: int = None, num_test: int = None,num_raw_test:int=None,
                 num_train_seed: int = None, num_dev_seed: int = None, num_test_seed: int = None,num_raw_test_seed: int=None,
                 save_dir: str = None, load_exp_id: str = None, gen_mode: str = None, incontext: str = None, prompt_type: str = None,
                 presample: int = None,
                 ):
        super().__init__()

        self.dataset = dataset
        self.data_path = data_path # ${data_dir}/${.dataset}/${model.arch}/
        
        if "davinci" in arch:
            self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(arch)

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.eff_train_batch_size = eff_train_batch_size
        self.num_workers = num_workers

        self.num_samples = {'train': num_train, 'dev': num_dev, 'test': num_test,'raw_test':num_raw_test}
        self.num_samples_seed = {'train': num_train_seed, 'dev': num_dev_seed, 'test': num_test_seed, 'raw_test':num_raw_test_seed}

        self.save_dir = save_dir
        self.load_exp_id = load_exp_id
        self.gen_mode = gen_mode
        self.incontext = incontext
        self.prompt_type = prompt_type
        self.presample = presample

    def load_dataset(self, split):
        dataset = {}
        if not self.gen_mode:
            data_path = os.path.join(self.data_path, split)
        else:
            data_path = os.path.join(self.data_path, self.gen_mode, self.incontext, self.prompt_type, split)
            print(data_path)
        
        assert Path(data_path).exists()

        for key in tqdm(data_keys, desc=f'Loading {split} set'):
            if self.num_samples[split] is not None:
                filename = f'{key}_{self.num_samples[split]}_{self.num_samples_seed[split]}.pkl'
            elif self.presample and split == 'test': # only for the test set
                filename = key + "_" + str(self.presample) + ".pkl"
            else:
                filename = f'{key}.pkl'
            
            data_path_ = data_path
            with open(os.path.join(data_path_, filename), 'rb') as f:
                dataset[key] = pickle.load(f)

        return dataset

    def setup(self, stage,  splits=['all']):
        self.dataset_dict = {}
        splits = ['train', 'dev', 'test'] if splits == ['all'] else splits
        for split in splits:
            self.dataset_dict[split] = TextClassificationDataset(self.load_dataset(split), split, self.tokenizer)

    def train_dataloader(self):
        return DataLoader(
            self.dataset_dict['train'],
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.dataset_dict['train'].collater,
            pin_memory=True
        )

    def val_dataloader(self, test=False):
        if test:
            return DataLoader(
                self.dataset_dict['dev'],
                batch_size=self.eval_batch_size,
                num_workers=self.num_workers,
                collate_fn=self.dataset_dict['dev'].collater,
                pin_memory=True
            )

        return [
            DataLoader(
            self.dataset_dict[eval_split],
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.dataset_dict[eval_split].collater,
            pin_memory=True)
            
            for eval_split in ['dev', 'test']
        ]

    def test_dataloader(self):     
        return DataLoader(
            self.dataset_dict['test'],
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.dataset_dict['test'].collater,
            pin_memory=True
        )


class TextClassificationDataset(Dataset):
    def __init__(self, data, split, tokenizer):
        self.data = data
        self.split = split
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data['item_idx'])

    def __getitem__(self, idx):
        item_idx = torch.LongTensor([self.data['item_idx'][idx]])
        input_ids = torch.LongTensor(self.data['input_ids'][idx])
        attention_mask = torch.LongTensor(self.data['attention_mask'][idx])
        labels = torch.LongTensor(self.data['labels'][idx])
        gold_label = self.data['gold_label'][idx]
        prompt_text = self.data['prompt_text'][idx]
        output_label_text = self.data['output_label_text'][idx]

        item = {
            'item_idx': item_idx,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'gold_label': gold_label,
            'prompt_text': prompt_text,
            'output_label_text':output_label_text
        }

        return item

    def collater(self, items):
        item_idx = torch.cat([x['item_idx'] for x in items])
        input_ids = pad_sequence([x['input_ids'] for x in items], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence([x['attention_mask'] for x in items], batch_first=True, padding_value=0)
        labels = pad_sequence([x['labels'] for x in items], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels[labels == self.tokenizer.pad_token_id] = -100
        gold_label = [x['gold_label'] for x in items]
        prompt_text = [x['prompt_text'] for x in items]
        output_label_text = [x['output_label_text'] for x in items]

        batch = {
            'item_idx': item_idx,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'gold_label': gold_label,
            'split': self.split, # when evaluate_ckpt=true, split always test
            'prompt_text' : prompt_text,
            'output_label_text' : output_label_text
        }
        
        return batch