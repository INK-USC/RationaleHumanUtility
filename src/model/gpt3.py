import os
import random
from subprocess import list2cmdline
import time
from tqdm import tqdm
import openai
import pickle
from hydra.utils import get_original_cwd
from src.utils.gpt3_utils import get_API_key, parse_response, output_label_mapper
from src.utils.metrics import init_perf_metrics
from src.utils.logging import log_gpt3_to_neptune
from src.model.base_model import BaseModel
from rapidfuzz.fuzz import token_set_ratio


class GPT3Runner(BaseModel):
    def __init__(self, dataset, arch, gen_mode, incontext, prompt_type, save_dir, dataname):
        super().__init__()
        self.dataset = dataset
        self.openai = openai
        self.openai.api_key = get_API_key()

        self.arch = arch
        self.gen_mode = gen_mode
        self.incontext = incontext
        self.prompt_type = prompt_type
        self.save_dir = save_dir
        self.dataname = dataname

    def run_step(self, batch, split, batch_idx):

        ret_dict = {}

        # if (i+1) % 3 == 0:
        #     break
        prompt_text = batch['prompt_text']
        if type(prompt_text) == list:
            print(prompt_text)
            prompt_text = prompt_text[0]
        prompt_length = len(batch['input_ids'][0])
        print(prompt_length)
        if self.gen_mode == "I-O":
            if self.incontext == "cot":
                if self.prompt_type == "cot":
                    max_tokens = 6
            if self.incontext=='feb_random':
                max_tokens=6
        else:
            max_tokens = 2048 - prompt_length
            print(max_tokens)
        # print(type(prompt_text))
        # print(prompt_text)
        response = openai.Completion.create(engine=self.arch, prompt=prompt_text, max_tokens=max_tokens, temperature=0.0)
        print(response)
        response_text = response['choices'][0]['text']
        label, rationale = parse_response(response_text, self.gen_mode, self.incontext, self.prompt_type, self.dataname)
        ret_dict['pred_label'] = label
        ret_dict['pred_rationale'] = rationale

        gold_label, gold_rationale = parse_response(batch['output_label_text'][0], self.gen_mode, self.incontext, self.prompt_type, self.dataname)
        ret_dict['gold_label'] = gold_label
        ret_dict['gold_rationale'] = gold_rationale

        print(label)
        print(gold_label)
        print(batch['output_label_text'])
        print(token_set_ratio(gold_label, label))
        print("_-_-_-_-_-_-_-_-")
        print(rationale)
        print(gold_rationale)
        print("_-_-_-_-_-_-_-_-")
        print("_-_-_-_-_-_-_-_-")

        # print(labels)

        # predicted_labels.append(mapped_label)
        # ground_truth_labels.append(batch['output_label_text'])
        # rationales.append(rationale)

        # if (i+1) % 59 == 0:
        #     time.sleep(60)

        # log_gpt3_to_neptune(self, predicted_labels, ground_truth_labels, perf_metrics)
        time.sleep(2)

        return ret_dict

    def aggregate_epoch(self, outputs, split):
        if split != 'test':
            raise NotImplementedError
        else:
            gold_labels = []
            pred_labels = []
            pred_rationales = []
            gold_rationales = []
            for elem in outputs:
                gold_labels.append(str(elem['gold_label']))
                pred_labels.append(str(elem['pred_label']))
                pred_rationales.append(elem['pred_rationale'])
                gold_rationales.append(elem['gold_rationale'])
            log_gpt3_to_neptune(self, pred_labels, gold_labels, self.dataname)

            out_dir = self.save_dir

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            keys = ['pred_label', 'gold_label', 'pred_rationale', 'gold_rationale']
            for key in keys:
                if key == 'pred_label':
                    out_data = pred_labels
                elif key == 'gold_label':
                    out_data = gold_labels
                elif key == 'pred_rationale':
                    out_data = pred_rationales
                elif key == 'gold_rationale':
                    out_data = gold_rationales
                out_file = os.path.join(out_dir, f'{split}_{key}.pkl')
                pickle.dump(out_data, open(out_file, 'wb'))