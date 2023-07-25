import argparse, os, sys, random, logging
from collections import defaultdict as ddict, Counter

import numpy as np
import pandas as pd
from pickle5 import pickle
from tqdm import tqdm

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.append(os.path.join(sys.path[0], '..'))
from src.utils.data import dataset_info, data_keys
from process_prompt_template import process_strategyqa_prompt, process_openbookqa_prompt, process_qed_prompt, feb_gpt3_prompt_creator, cot_gpt3_prompt_creator, process_openbookqa_choices

logging.basicConfig(level=logging.DEBUG, format='%(relativeCreated)6d %(threadName)s %(message)s')
logger = logging.getLogger(__name__)

NUM_SPECIAL_TOKENS = 3
NUM_EXTRA_SPECIAL_TOKENS = 100

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def sample_test_split(data_path, num_train, num_test, seed):
    filename = f'test_split_{num_test}_{seed}.pkl'
    if os.path.exists(os.path.join(data_path, filename)):
        with open(os.path.join(data_path, filename), 'rb') as f:
            test_split = pickle.load(f)
    else:
        test_split = sorted(list(np.random.choice(np.arange(num_train+num_test), size=num_test, replace=False)))
        with open(os.path.join(data_path, filename), 'wb') as f:
            pickle.dump(test_split, f)
    return test_split

def stratified_sampling(x, n_samples, stratify):
    """Perform stratified sampling of a tensor.
    
    parameters
    ----------
    x: np.ndarray or torch.Tensor
        Array to sample from. Samples from first dimension.
        
    n_samples: int
        Number of samples to sample
        
    stratify: tuple of int
        Size of each subgroup. Note that the sum of all the sizes 
        need to be equal to `x.shape[']`.
    """
    n_total = x.shape[0]
    assert sum(stratify) == n_total
    
    n_strat_samples = [int(i*n_samples/n_total) for i in stratify]
    cum_n_samples = np.cumsum([0]+list(stratify))
    sampled_idcs = []
    for i, n_strat_sample in enumerate(n_strat_samples):
        sampled_idcs.append(np.random.choice(range(cum_n_samples[i], cum_n_samples[i+1]), 
                                            replace=False, 
                                            size=n_strat_sample))
        
    # might not be correct number of samples due to rounding
    n_current_samples = sum(n_strat_samples)
    if  n_current_samples < n_samples:
        delta_n_samples = n_samples - n_current_samples
        # might actually resample same as before, but it's only for a few
        sampled_idcs.append(np.random.choice(range(n_total), replace=False, size=delta_n_samples))
        
    sampled_idcs = np.concatenate(sampled_idcs)
    samples = x[sampled_idcs, ...]
    
    return samples, sampled_idcs

def sample_dataset(data_path, dataset_dict, classes, split, num_samples, seed):
    sampled_split_filename = f'{split}_split_{num_samples}_{seed}.pkl'
    if os.path.exists(os.path.join(data_path, sampled_split_filename)):
        with open(os.path.join(data_path, sampled_split_filename), 'rb') as f:
            sampled_split = pickle.load(f)
    else:
        if 'strategyqa' in data_path:
            labels = [classes.index(x) for x in dataset_dict['gold_label']]
            label_counts = list(Counter(labels).values())
            _, sampled_split = stratified_sampling(torch.tensor(labels), num_samples, label_counts)
        else:
            sampled_split = torch.randperm(len(dataset_dict['item_idx']))[:num_samples].numpy()
        sampled_split = list(sampled_split)
        with open(os.path.join(data_path, sampled_split_filename), 'wb') as f:
            pickle.dump(sampled_split, f)
    
    for key in data_keys:
        dataset_dict[key] = sampled_split if key == 'item_idx' else [dataset_dict[key][i] for i in sampled_split]

    return dataset_dict

def load_datasets(data_path, split, num_samples, seed):
    dataset_dict = ddict(list)
    for key in tqdm(data_keys, desc=f'Loading {split} dataset'):
        filename = f'{key}.pkl' if num_samples is None else f'{key}_{num_samples}_{seed}.pkl'
        with open(os.path.join(data_path, filename), 'rb') as f:
            dataset_dict[key] = pickle.load(f)
    return dataset_dict

def save_dataset(data_path, dataset_dict, split, num_samples, seed):
    save_keys = data_keys
    for key in tqdm(save_keys, desc=f'Saving {split} dataset'):
        filename = f'{key}_{num_samples}_{seed}.pkl' if num_samples is not None else f'{key}.pkl'
        if args.presample != 0:
            filename = key + "_" + str(args.presample) + ".pkl"
        with open(os.path.join(data_path, filename), 'wb') as f:
            pickle.dump(dataset_dict[key], f)

def main(args):
    set_random_seed(args.seed)

    assert args.split is not None and args.arch is not None
    assert args.num_samples is None or args.num_samples >= 1

    split, num_examples = dataset_info[args.dataset][args.split]
    if args.presample != 0:
        num_examples = args.presample
    if args.num_samples is not None:
        assert args.num_samples < num_examples

    if "davinci" in args.arch:
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.arch)

    if args.gen_mode:
        data_path = os.path.join(args.data_dir, args.dataset, args.arch, args.gen_mode, args.incontext, args.prompt_type, args.split)
    else:
        data_path = os.path.join(args.data_dir, args.dataset, args.arch, args.split)

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if args.num_samples:
        missing_data_keys = [x for x in data_keys if not os.path.exists(os.path.join(data_path, f'{x}_{args.num_samples}_{args.seed}.pkl'))]
    elif args.presample != 0:
        missing_data_keys = [x for x in data_keys if not os.path.exists(os.path.join(data_path, x + "_" + str(args.presample) + ".pkl"))]
    else:
        missing_data_keys = [x for x in data_keys if not os.path.exists(os.path.join(data_path, f'{x}.pkl'))]

    if args.num_samples is None and missing_data_keys:
        dataset_dict = ddict(list)
        if args.dataset == 'strategyqa':
            if args.presample != 0:
                dataset_path = os.path.join(args.data_dir, args.dataset, 'raw', 'strategyqa_processed_' + split + "_" + str(args.presample) + '.json')
            else:
                dataset_path = os.path.join(args.data_dir, args.dataset, 'raw', 'strategyqa_processed_' + split + '.json')
            if split=='raw_test':
                dataset_path = os.path.join(args.data_dir,args.dataset,'raw','strategyqa_test.json')
            train_prompts = None
            max_prompt_length = None
            if "feb" in args.incontext:  
                train_dataset = pd.read_json(os.path.join(args.data_dir, args.dataset, 'raw', 'strategyqa_processed_train.json'), orient='records')
                max_prompt_length = args.max_prompt_length
                train_prompts = []
                for index, row in train_dataset.iterrows():
                    if args.prompt_type == "feb":
                        if args.gen_mode == "I-O":
                            train_prompts.append(feb_gpt3_prompt_creator(row['question'], '', row['answer'], ['Yes','No']))
                        else:
                            train_prompts.append(feb_gpt3_prompt_creator(row['question'], row['rationale'], row['answer'], ['Yes','No']))
                    elif args.prompt_type == "cot":
                        if args.gen_mode == 'I-O':
                            train_prompts.append(cot_gpt3_prompt_creator(row['question'], '', row['answer'], ['Yes','No']))
                        else:
                            train_prompts.append(cot_gpt3_prompt_creator(row['question'], row['rationale'], row['answer'], ['Yes','No']))
            dataset = pd.read_json(dataset_path, orient='records')
            classes = dataset_info[args.dataset]['classes']
            indices = range(num_examples)
            for idx, item_idx in tqdm(enumerate(indices), total=num_examples, desc=f'Building {args.split} dataset'):
                instance = dataset.loc[item_idx]
                question = instance['question']
                if split=='raw_test':
                    label=False
                    rationale=''
                else:
                    label = classes[instance['answer']]
                    rationale = instance['rationale']
  
                prompt, output_label = process_strategyqa_prompt(question, rationale, label, args.gen_mode, args.incontext, args.prompt_type, train_prompts, max_prompt_length, tokenizer)

                prompt_ids = tokenizer(text=prompt, add_special_tokens=False).input_ids
                output_label_ids = tokenizer(text=output_label, add_special_tokens=False).input_ids

                prompt_ids.append(tokenizer.eos_token_id)
                output_label_ids.append(tokenizer.eos_token_id)
                attention_mask = [1] * len(prompt_ids)

                dataset_dict['item_idx'].append(item_idx)
                dataset_dict['input_ids'].append(prompt_ids) # numbers
                dataset_dict['attention_mask'].append(attention_mask)
                dataset_dict['labels'].append(output_label_ids)
                dataset_dict['gold_label'].append(label)
                dataset_dict['prompt_text'].append(prompt) # explain strategyqa {question}
                dataset_dict['output_label_text'].append(output_label) # {answer} because {explanation}

        elif args.dataset == 'openbookqa':
            train_prompts = None
            max_prompt_length = None
            if "feb" in args.incontext:  
                train_dataset = load_dataset('openbookqa','additional')['train']
                max_prompt_length = args.max_prompt_length
                train_prompts = []
                for index in range(len(train_dataset)):
                    row = train_dataset[index]
                    choices = process_openbookqa_choices(row['choices'])
                    if args.prompt_type == "feb":
                        if args.gen_mode == "I-O":
                            train_prompts.append(feb_gpt3_prompt_creator(row['question_stem'], '', row['answerKey'], choices, 'openbookqa'))
                        else:
                            train_prompts.append(feb_gpt3_prompt_creator(row['question_stem'], row['fact1'], row['answerKey'], choices, 'openbookqa'))
                    elif args.prompt_type == "cot":
                        if args.gen_mode == 'I-O':
                            train_prompts.append(cot_gpt3_prompt_creator(row['question_stem'], '', row['answerKey'], choices, 'openbookqa'))
                        else:
                            train_prompts.append(cot_gpt3_prompt_creator(row['question_stem'], row['fact1'], row['answerKey'], choices, 'openbookqa'))


            dataset = load_dataset('openbookqa','additional')[split]
            # dataset = pd.read_json(dataset_path, orient='records')
            classes = dataset_info['openbookqa']['classes']
            indices = range(num_examples)
            for idx, item_idx in tqdm(enumerate(indices), total=num_examples, desc=f'Building {args.split} dataset'):
                question = dataset[item_idx]['question_stem']
                choices = process_openbookqa_choices(dataset[item_idx]['choices'])
                label = dataset[item_idx]['answerKey']
                rationale = dataset[item_idx]['fact1']
  
                prompt, output_label = process_openbookqa_prompt(question, choices, rationale, label, args.gen_mode, args.incontext, args.prompt_type, train_prompts, max_prompt_length, tokenizer)

                prompt_ids = tokenizer(text=prompt, add_special_tokens=False).input_ids
                output_label_ids = tokenizer(text=output_label, add_special_tokens=False).input_ids

                prompt_ids.append(tokenizer.eos_token_id)
                output_label_ids.append(tokenizer.eos_token_id)
                attention_mask = [1] * len(prompt_ids)

                dataset_dict['item_idx'].append(item_idx)
                dataset_dict['input_ids'].append(prompt_ids) # numbers
                dataset_dict['attention_mask'].append(attention_mask)
                dataset_dict['labels'].append(output_label_ids)
                dataset_dict['gold_label'].append(label)
                dataset_dict['prompt_text'].append(prompt) # explain strategyqa {question}
                dataset_dict['output_label_text'].append(output_label) # {answer} because {explanation}

        elif args.dataset == 'qed':
            train_prompts = None
            max_prompt_length = None
            if "feb" in args.incontext:  
                train_dataset = pd.read_csv(os.path.join(args.data_dir, args.dataset, "raw", "processed_train.csv"))
                max_prompt_length = args.max_prompt_length
                train_prompts = []
                for index in range(len(train_dataset)):
                    row = train_dataset.loc[index]
                    if args.prompt_type == "feb":
                        if args.gen_mode == "I-O":
                            train_prompts.append(feb_gpt3_prompt_creator(row['question'], '', row['answer'], None, 'qed'))
                        else:
                            train_prompts.append(feb_gpt3_prompt_creator(row['question'], row['rationale'], row['answer'], None, 'qed'))
                    elif args.prompt_type == "cot":
                        if args.gen_mode == 'I-O':
                            train_prompts.append(cot_gpt3_prompt_creator(row['question'], '', row['answer'], None, 'qed'))
                        else:
                            train_prompts.append(cot_gpt3_prompt_creator(row['question'], row['rationale'], row['answer'], None, 'qed'))


            dataset = pd.read_csv(os.path.join(args.data_dir, args.dataset, "raw", f"processed_{split}.csv"))
            # dataset = pd.read_json(dataset_path, orient='records')
            indices = range(num_examples)
            for idx, item_idx in tqdm(enumerate(indices), total=num_examples, desc=f'Building {args.split} dataset'):
                instance = dataset.loc[item_idx]
                question = instance['question']
                label = instance['answer']
                rationale = instance['rationale']
  
                prompt, output_label = process_qed_prompt(question, rationale, label, args.gen_mode, args.incontext, args.prompt_type, train_prompts, max_prompt_length, tokenizer)

                prompt_ids = tokenizer(text=prompt, add_special_tokens=False).input_ids
                output_label_ids = tokenizer(text=output_label, add_special_tokens=False).input_ids

                prompt_ids.append(tokenizer.eos_token_id)
                output_label_ids.append(tokenizer.eos_token_id)
                attention_mask = [1] * len(prompt_ids)

                dataset_dict['item_idx'].append(item_idx)
                dataset_dict['input_ids'].append(prompt_ids) # numbers
                dataset_dict['attention_mask'].append(attention_mask)
                dataset_dict['labels'].append(output_label_ids)
                dataset_dict['gold_label'].append(label)
                dataset_dict['prompt_text'].append(prompt) # explain strategyqa {question}
                dataset_dict['output_label_text'].append(output_label) # {answer} because {explanation}

        else:
            raise NotImplementedError

    elif args.num_samples is not None:
        assert all([os.path.exists(os.path.join(data_path, f'{x}.pkl')) for x in data_keys])
        if not all([os.path.exists(os.path.join(data_path, f'{x}_{args.num_samples}_{args.seed}.pkl')) for x in data_keys]):
            dataset_dict = load_datasets(data_path, args.split, None, None)
            if args.dataset!='qed':
                classes = dataset_info[args.dataset]['classes']
            else:
                classes = None
            dataset_dict = sample_dataset(data_path, dataset_dict, classes, args.split, args.num_samples, args.seed)
        else:
            dataset_dict = load_datasets(data_path, args.split, args.num_samples, args.seed)
    
    save_dataset(data_path, dataset_dict, args.split, args.num_samples, args.seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset preprocessing.')
    parser.add_argument('--data_dir', type=str, default='../data/', help='Root directory for datasets.')
    parser.add_argument('--dataset', type=str, choices=['strategyqa','openbookqa', 'qed'])
    parser.add_argument('--gen_mode', type=str, choices=['I-OR','I-RO', 'I-O','IR-O'])
    parser.add_argument('--arch', type=str, default='t5-base', choices=['t5-small', 't5-base', 't5-large', 't5-3b', 't5-11b', 'davinci-instruct-beta','allenai/unifiedqa-t5-small','allenai/unifiedqa-t5-base','allenai/unifiedqa-t5-large','allenai/unifiedqa-t5-3b'])
    parser.add_argument('--incontext', type=str, default='None', choices=['None','cot', 'feb', 'feb_6', 'feb_random'])
    parser.add_argument('--prompt_type', type=str, default='cot', choices=['cot', 'squadt5', 'infilling', 'feb', 'qasimple', 't5like']) # squadt5, infilling no incontext, 
    parser.add_argument('--split', type=str, help='Dataset split.', choices=['train', 'dev', 'test','raw_test'])
    parser.add_argument('--num_samples', type=int, help='Number of examples to sample. None means all available examples are used.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--presample', type=int, default=0, help='presampled')
    parser.add_argument('--max_prompt_length', type=int, default=None, help='Max length of input demonstrations.')
    args = parser.parse_args()
    main(args)

"""
python scripts/build_dataset.py \
    --dataset qed \
    --gen_mode I-OR \
    --arch davinci-instruct-beta \
    --incontext feb_6 \
    --prompt_type feb \
    --max_prompt_length 100 \
    --split test 

"""

"""
python scripts/build_dataset.py \
    --dataset feb \
    --gen_mode I-RO \
    --arch davinci-instruct-beta \
    --incontext cot \
    --prompt_type cot \
    --split train \
    --max_prompt_length 100 \
    --num_samples 20 \
    --seed 0

python scripts/build_dataset.py \
    --dataset strategyqa \
    --gen_mode I-OR \
    --arch davinci-instruct-beta \
    --incontext feb \
    --prompt_type feb \
    --split test \
    --max_prompt_length 100 \
    --num_samples 200 \
    --seed 0
    """


"""
Ziyi's command

python scripts/build_dataset.py \
    --dataset openbookqa \
    --gen_mode I-O \
    --arch t5-3b \
    --prompt_type squadt5 \
    --split train \
    --presample 200

python scripts/build_dataset.py \
    --dataset strategyqa \
    --gen_mode I-OR \
    --arch t5-3b \
    --prompt_type qasimple \
    --split train \
    --num_samples 48 \
    --seed 0

python scripts/build_dataset.py \
    --dataset strategyqa \
    --gen_mode I-OR \
    --arch t5-3b \
    --prompt_type qasimple \
    --split dev

python scripts/build_dataset.py \
    --dataset qed \
    --gen_mode I-OR \
    --arch allenai/unifiedqa-t5-large \
    --prompt_type qasimple \
    --split train
"""

"""
python scripts/build_dataset.py \
    --dataset openbookqa \
    --gen_mode I-OR \
    --prompt_type infilling \
    --split train

python scripts/build_dataset.py \
    --dataset openbookqa \
    --arch davinci-instruct-beta \
    --gen_mode I-RO \
    --incontext feb_random \
    --prompt_type cot \
    --split test \
    --max_prompt_length 100
"""


"""
python scripts/build_dataset.py \
    --dataset openbookqa \
    --gen_mode I-O \
    --arch allenai/unifiedqa-t5-3b \
    --prompt_type t5like \
    --split train
    
python scripts/build_dataset.py \
    --dataset openbookqa \
    --gen_mode I-O \
    --arch t5-3b \
    --prompt_type infilling \
    --split train \
    --num_samples 128 \
    --seed 0

CUDA_VISIBLE_DEVICES=1 \
python main.py -m \
data=openbookqa \
data.gen_mode=I-O \
data.incontext=None \
data.prompt_type=squadt5 \
model=lm \
model.arch=t5-large \
setup.train_batch_size=4 \
setup.eval_batch_size=4 \
setup.num_workers=3 \
seed=0 \
trainer.max_steps=20000000 \
trainer.accelerator=gpu \
trainer.devices=1 \
setup.eff_train_batch_size=4 \
setup.accumulate_grad_batches=1 \
trainer.check_val_every_n_epoch=25 \
trainer.max_epochs=25 \
training.patience=25 \
trainer.log_every_n_steps=10



CUDA_VISIBLE_DEVICES=3 \
python main.py -m \
data=openbookqa \
data.gen_mode=I-O \
data.incontext=None \
data.prompt_type=t5like \
data.num_train=128 \
data.num_train_seed=0 \
model=lm \
model.arch=allenai/unifiedqa-t5-3b \
setup.train_batch_size=1 \
setup.eval_batch_size=1 \
setup.num_workers=3 \
seed=0 \
trainer.max_steps=20000000 \
trainer.accelerator=gpu \
trainer.devices=1 \
setup.eff_train_batch_size=4 \
setup.accumulate_grad_batches=4 \
trainer.check_val_every_n_epoch=25 \
trainer.max_epochs=25 \
training.patience=25 \
trainer.log_every_n_steps=10

CUDA_VISIBLE_DEVICES=3 \
python main.py -m \
data=openbookqa \
data.gen_mode=I-O \
data.incontext=None \
data.prompt_type=t5like \
data.num_train=128 \
data.num_train_seed=1 \
model=lm \
model.arch=allenai/unifiedqa-t5-3b \
setup.train_batch_size=1 \
setup.eval_batch_size=1 \
setup.num_workers=3 \
seed=0 \
trainer.max_steps=20000000 \
trainer.accelerator=gpu \
trainer.devices=1 \
setup.eff_train_batch_size=4 \
setup.accumulate_grad_batches=4 \
trainer.check_val_every_n_epoch=25 \
trainer.max_epochs=25 \
training.patience=25 \
trainer.log_every_n_steps=10

CUDA_VISIBLE_DEVICES=3 \
python main.py -m \
data=openbookqa \
data.gen_mode=I-O \
data.incontext=None \
data.prompt_type=t5like \
data.num_train=128 \
data.num_train_seed=2 \
model=lm \
model.arch=allenai/unifiedqa-t5-3b \
setup.train_batch_size=1 \
setup.eval_batch_size=1 \
setup.num_workers=3 \
seed=0 \
trainer.max_steps=20000000 \
trainer.accelerator=gpu \
trainer.devices=1 \
setup.eff_train_batch_size=4 \
setup.accumulate_grad_batches=4 \
trainer.check_val_every_n_epoch=25 \
trainer.max_epochs=25 \
training.patience=25 \
trainer.log_every_n_steps=10

    srun --gres=gpu:8000:1 -t 1-10 \
    python main.py \
    data=strategyqa \
    data.gen_mode=I-O \
    data.incontext=None \
    data.prompt_type=t5like \
    data.num_train=128 \
    data.num_train_seed=0 \
    model=lm \
    model.arch=allenai/unifiedqa-t5-3b \
    setup.train_batch_size=1 \
    setup.eval_batch_size=1 \
    setup.num_workers=3 \
    seed=0 \
    trainer.max_steps=20000 \
    trainer.devices=1 \
    trainer.accelerator='gpu' \
    setup.eff_train_batch_size=4 \
    setup.accumulate_grad_batches=4 \
    trainer.check_val_every_n_epoch=25 \
    trainer.max_epochs=25 \
    training.patience=25    
"""
