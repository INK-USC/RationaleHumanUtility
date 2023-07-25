import os, pickle, warnings, math
from typing import Optional, List
from itertools import chain

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from hydra.utils import instantiate, get_original_cwd
from omegaconf import DictConfig

from transformers import T5ForConditionalGeneration, AutoTokenizer
from lightning_transformers.utilities.deepspeed import enable_transformers_pretrained_deepspeed_sharding

from src.model.base_model import BaseModel
from src.model.mlp import MLP_factory
from src.utils.metrics import init_best_metrics, init_perf_metrics, process_outputs
from src.utils.optim import setup_optimizer_params, setup_scheduler
from src.utils.logging import log_step_losses, log_epoch_losses, log_epoch_metrics
from deepspeed.ops.adam import FusedAdam
from parallelformers import parallelize


class LanguageModel(BaseModel):
    def __init__(self,
                 model_type: str, arch: str, dataset: str, optimizer: DictConfig,
                 scheduler: DictConfig, neg_weight=1, save_outputs: bool = False, save_exp_id: str = None, 
                 load_exp_id: str = None, strategy: str = None, num_classes: int = None,
                 **kwargs):

        super().__init__()

        # if strategy is not None:
        #     enable_transformers_pretrained_deepspeed_sharding(self)
        #     parallelize(self.model, num_gpus=2, fp16=True, verbose='detail')

        self.save_hyperparameters()

        assert model_type in ['lm']
        assert arch in ['t5-small', 't5-base', 't5-large', 't5-3b', 't5-11b', 'allenai/unifiedqa-t5-small','allenai/unifiedqa-t5-base','allenai/unifiedqa-t5-large','allenai/unifiedqa-t5-3b']

        self.model_type = model_type
        self.arch = arch
        self.dataset = dataset
        self.optimizer = optimizer
        # self.num_classes = num_classes
       
        self.scheduler = scheduler
        self.neg_weight = neg_weight

        if save_outputs:
            assert save_exp_id is not None
        self.save_outputs = save_outputs
        self.save_exp_id = save_exp_id
        self.load_exp_id = load_exp_id

        self.model = T5ForConditionalGeneration.from_pretrained(arch)
        self.tokenizer = AutoTokenizer.from_pretrained(arch)

        self.best_metrics = init_best_metrics()
        self.perf_metrics = init_perf_metrics()

        # self.strategy = strategy

    def forward(self, input_ids, attention_mask, labels, mode):
        assert mode in ['train', 'eval']
        if mode == 'train':
            return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        else:
            return self.model.generate(input_ids=input_ids, return_dict_in_generate=True, output_scores=self.save_outputs,\
                max_length=100, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
        
    def run_step(self, batch, split, batch_idx):
        input_ids = batch['input_ids']
        attn_mask = batch['attention_mask']
        labels = batch['labels']

        eval_split: str = batch['split']
        if split == 'train':
            assert split == eval_split
        ret_dict, loss_dict = {}, {}
        ret_dict['eval_split'] = eval_split

        if split == 'train':
            outputs = self.forward(input_ids, attn_mask, labels, mode='train')
            loss = outputs.loss    

            loss_dict['loss'] = loss
            log_step_losses(self, loss_dict, eval_split)

            ret_dict['loss'] = loss

        else:
            print("++++++++++++++++++++++++++++++++ENTERING PREDICTION MODE++++++++++++++++++++++++++++++++")
            
            outputs = self.forward(input_ids, attn_mask, labels, mode='eval')
            ret_dict['pred_ids'] = outputs.sequences
            is_infilling = False
            if "<extra_id_0>" in batch['output_label_text'][0]:
                is_infilling = True
            # print(outputs.sequences)
            # print(batch['prompt_text'])
            # print(batch['output_label_text'])
            # print(batch['gold_label'])
            l, r = process_outputs(outputs.sequences, self.tokenizer, is_infilling)
            ret_dict['pred_label'], ret_dict['pred_rationale'] = l, r
            ret_dict['gold_label'] = batch['gold_label']
            if self.save_outputs:
                ret_dict['logits'] = outputs.scores
                ret_dict['item_idx'] = batch['item_idx']

        return ret_dict

    def aggregate_epoch(self, outputs, split):
        if split == 'train':
            splits = ['train']
        elif split == 'dev':
            splits = ['dev', 'test']
        elif split == 'test':
            splits = [outputs[0]['eval_split']]
        outputs_list = outputs if split == 'dev' else [outputs]
        
        for dataset_idx, eval_split in enumerate(splits):
            outputs = outputs_list[dataset_idx]
            if eval_split == 'train':
                log_epoch_losses(self, outputs, eval_split) # Log epoch losses
            else:
                log_epoch_metrics(self, outputs, eval_split, self.dataset) # Log epoch metrics

        # Save outputs to file
        if self.save_outputs and eval_split != 'train':
            out_dir = f'{get_original_cwd()}/../save/{self.save_exp_id}/model_outputs/{self.dataset}/{eval_split}'

            out_dir = os.path.join(out_dir)

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            keys = ['pred_label', 'gold_label', 'pred_rationale']
            for key in keys:
                if key == 'item_idx':
                    out_data = torch.cat([x[key] for x in outputs]).detach().cpu().numpy()
                elif key == 'logits':
                    out_data = [torch.stack(x[key], dim=1).detach().cpu().numpy() for x in outputs]
                elif key == 'pred_ids':
                    out_data = [x[key].detach().cpu().numpy() for x in outputs]
                else:
                    out_data = list(chain.from_iterable([x[key] for x in outputs]))
                out_file = os.path.join(out_dir, f'{eval_split}_{key}.pkl')
                pickle.dump(out_data, open(out_file, 'wb'))

    def configure_optimizers(self):
        # if self.strategy is not None:
        #     return FusedAdam(self.parameters())
        optimizer_params = setup_optimizer_params(self.model, self.optimizer)
        self.optimizer['lr'] = self.optimizer['lr'] * self.trainer.world_size
        optimizer = instantiate(
            self.optimizer, params=optimizer_params,
            _convert_="partial"
        )
        if self.scheduler.lr_scheduler == 'linear_with_warmup':
            scheduler = setup_scheduler(self.scheduler, self.total_steps, optimizer)
            return [optimizer], [scheduler]
        elif self.scheduler.lr_scheduler == 'fixed':
            return [optimizer]
        else:
            raise NotImplementedError
            