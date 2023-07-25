import getpass, logging, socket
from typing import Any, List
from itertools import chain

import torch
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning.loggers import NeptuneLogger
# from neptunecontrib.monitoring.pytorch_lightning import NeptuneLogger

from src.utils.metrics import calc_preds, get_step_metrics, get_epoch_metrics
from sklearn.metrics import accuracy_score
from rapidfuzz.fuzz import token_set_ratio


API_LIST = {
    "neptune": {
        'brihi': 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5YWY2NzY2Zi1hYmM2LTQ1ZmEtYjQxNS1kNjAzMWYyZjY5ZTcifQ==',
        'zhewei': 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyZmUxNThhYy1lMTRmLTQ0NGEtYTE0ZC02YTg1N2IxZWE1MWIifQ==',
        'ziyi':'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4ZGM4NTNhOC1mOWIxLTQ0MGYtYWE4OC1jZThlZTAzZjNiZTIifQ=='
    },
}


def get_username():
    return getpass.getuser()

def flatten_cfg(cfg: Any) -> dict:
    if isinstance(cfg, dict):
        ret = {}
        for k, v in cfg.items():
            flatten: dict = flatten_cfg(v)
            ret.update({
                f"{k}/{f}" if f else k: fv
                for f, fv in flatten.items()
            })
        return ret
    return {"": cfg}

def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger

def get_neptune_logger(
    cfg: DictConfig, project_name: str,
    name: str, tag_attrs: List[str], log_db: str,
    offline: bool, logger: str,
):
    neptune_api_key = API_LIST["neptune"][get_username()]
    print(neptune_api_key)

    # flatten cfg
    args_dict = {
        **flatten_cfg(OmegaConf.to_object(cfg)),
        "hostname": socket.gethostname()
    }
    tags = tag_attrs
    tags.append(log_db)
    print("==============================================================")
    print(tags)
    print(project_name)


    neptune_logger = NeptuneLogger(
        api_key=neptune_api_key,
        project=project_name,
        name=name,
        prefix="",
        tags=list(tags),
        log_model_checkpoints=offline,
    )

    neptune_logger.log_hyperparams(args_dict)

    try:
        # for unknown reason, must access this field otherwise becomes None
        print(neptune_logger.experiment)
    except BaseException:
        pass

    return neptune_logger

# def get_neptune_logger(
#     cfg: DictConfig, project_name: str,
#     name: str, tag_attrs: List[str], log_db: str,
#     offline: bool, logger: str,
# ):
#     neptune_api_key = API_LIST["neptune"][get_username()]

#     # flatten cfg
#     args_dict = {
#         **flatten_cfg(OmegaConf.to_object(cfg)),
#         "hostname": socket.gethostname()
#     }
#     tags = tag_attrs
#     tags.append(log_db)

#     neptune_logger = NeptuneLogger(
#         api_key=neptune_api_key,
#         project=project_name,
#         # experiment_name=name,
#         # params=args_dict,
#         tags=tags,
#         name=name,
#         log_model_checkpoints=offline,
#     )

#     try:
#         # for unknown reason, must access this field otherwise becomes None
#         print("IGHAIGIHGBHUKVVBK")
#         print(neptune_logger.experiment)
#     except BaseException:
#         pass

#     return neptune_logger

def log_data_to_neptune(model_class, data, data_name, data_type, suffix, split, ret_dict=None, detach_data=True):
    data_key = 'loss' if f'{data_name}_{data_type}' == 'total_loss' else f'{data_name}_{data_type}'
    if detach_data == True:
        model_class.log(f'{split}_{data_key}_{suffix}', data.detach(), prog_bar=True, sync_dist=True)
    else:
        model_class.log(f'{split}_{data_key}_{suffix}', data, prog_bar=True, sync_dist=True)
    if ret_dict is not None:
        ret_dict[data_key] = data.detach() if detach_data else data
    return ret_dict

def log_step_losses(model_class, loss_dict, split):
    log_data_to_neptune(model_class, loss_dict['loss'], 'total', 'loss', 'step', split, detach_data=False)

def log_epoch_losses(model_class, outputs, split):
    loss = torch.stack([x['loss'] for x in outputs]).mean()
    log_data_to_neptune(model_class, loss, 'total', 'loss', 'epoch', split)

def log_gpt3_to_neptune(model, predicted_labels, ground_truth_labels, dataname):
    # preds = torch.LongTensor([x == ground_truth_labels[i] for i, x in enumerate(predicted_labels)])
    # gold_labels = torch.ones(len(ground_truth_labels)).long()
    if dataname == "qed":
        # Add fuzzy matching here
        matched = 0
        total = len(predicted_labels)
        for i in range(len(predicted_labels)):
            score = token_set_ratio(predicted_labels[i], ground_truth_labels[i])
            if score > 63.3:
                matched+=1
        perf_metrics = (matched/total) * 100
        pass
    else:
        perf_metrics = accuracy_score(predicted_labels, ground_truth_labels) * 100
    model.log('test_acc_metric_epoch', perf_metrics, prog_bar=True)

def log_epoch_metrics(model_class, outputs, split, dataname):
    preds_ = list(chain.from_iterable([x['pred_label'] for x in outputs]))
    gold_labels_ = list(chain.from_iterable([x['gold_label'] for x in outputs]))

    preds_ = [x.lower() for x in preds_]
    gold_labels_ = [x.lower() for x in gold_labels_]

    if dataname == "qed":
        matched = 0
        total = len(preds_)
        for i in range(len(preds_)):
            score = token_set_ratio(preds_[i], gold_labels_[i])
            if score > 63.3:
                matched+=1
        perf_metrics = (matched/total) * 100
        log_data_to_neptune(model_class, perf_metrics, 'acc', 'metric', 'epoch', split, detach_data=False)
    else:
        preds = torch.LongTensor([x == gold_labels_[i] for i, x in enumerate(preds_)])
        gold_labels = torch.ones(len(gold_labels_)).long()

        perf_metrics = get_step_metrics(preds, gold_labels, model_class.perf_metrics)
        perf_metrics = get_epoch_metrics(model_class.perf_metrics)
        log_data_to_neptune(model_class, perf_metrics['acc'], 'acc', 'metric', 'epoch', split)

    if outputs[0].get('correct_pred_label'):
        correct_pred_label_ = list(chain.from_iterable([x['correct_pred_label'] for x in outputs]))

        if True in correct_pred_label_:
            correct_preds_ = [preds_[i] for i in range(len(preds_)) if correct_pred_label_[i]]
            correct_gold_labels_ = [gold_labels_[i] for i in range(len(gold_labels_)) if correct_pred_label_[i]]

            correct_preds = torch.LongTensor([x == correct_gold_labels_[i] for i, x in enumerate(correct_preds_)])
            correct_gold_labels = torch.ones(len(correct_gold_labels_)).long()

            correct_perf_metrics = get_step_metrics(correct_preds, correct_gold_labels, model_class.perf_metrics)
            correct_perf_metrics = get_epoch_metrics(model_class.perf_metrics)
            log_data_to_neptune(model_class, correct_perf_metrics['acc'], 'correct_pred_acc', 'metric', 'epoch', split)
            log_data_to_neptune(model_class, torch.tensor(len(correct_preds_)), 'num_correct_pred', 'metric', 'epoch', split)
        else:
            log_data_to_neptune(model_class, torch.tensor(0), 'num_correct_pred', 'metric', 'epoch', split)

        if False in correct_pred_label_:
            incorrect_preds_ = [preds_[i] for i in range(len(preds_)) if not correct_pred_label_[i]]
            incorrect_gold_labels_ = [gold_labels_[i] for i in range(len(gold_labels_)) if not correct_pred_label_[i]]

            incorrect_preds = torch.LongTensor([x == incorrect_gold_labels_[i] for i, x in enumerate(incorrect_preds_)])
            incorrect_gold_labels = torch.ones(len(incorrect_gold_labels_)).long()

            incorrect_perf_metrics = get_step_metrics(incorrect_preds, incorrect_gold_labels, model_class.perf_metrics)
            incorrect_perf_metrics = get_epoch_metrics(model_class.perf_metrics)
            log_data_to_neptune(model_class, incorrect_perf_metrics['acc'], 'incorrect_pred_acc', 'metric', 'epoch', split)
            log_data_to_neptune(model_class, torch.tensor(len(incorrect_preds_)), 'num_incorrect_pred', 'metric', 'epoch', split)
        else:
            log_data_to_neptune(model_class, torch.tensor(0), 'num_incorrect_pred', 'metric', 'epoch', split)

        if outputs[0].get('leak_pred_label'):
            leak_pred_label_ = list(chain.from_iterable([x['leak_pred_label'] for x in outputs]))

            if True in leak_pred_label_:
                leak_preds_ = [preds_[i] for i in range(len(preds_)) if leak_pred_label_[i]]
                leak_gold_labels_ = [gold_labels_[i] for i in range(len(gold_labels_)) if leak_pred_label_[i]]

                leak_preds = torch.LongTensor([x == leak_gold_labels_[i] for i, x in enumerate(leak_preds_)])
                leak_gold_labels = torch.ones(len(leak_gold_labels_)).long()

                leak_perf_metrics = get_step_metrics(leak_preds, leak_gold_labels, model_class.perf_metrics)
                leak_perf_metrics = get_epoch_metrics(model_class.perf_metrics)
                log_data_to_neptune(model_class, leak_perf_metrics['acc'], 'leak_pred_acc', 'metric', 'epoch', split)
                log_data_to_neptune(model_class, torch.tensor(len(leak_preds_)), 'num_leak_pred', 'metric', 'epoch', split)
            else:
                log_data_to_neptune(model_class, torch.tensor(0), 'num_leak_pred', 'metric', 'epoch', split)

            if False in leak_pred_label_:
                nonleak_preds_ = [preds_[i] for i in range(len(preds_)) if not leak_pred_label_[i]]
                nonleak_gold_labels_ = [gold_labels_[i] for i in range(len(gold_labels_)) if not leak_pred_label_[i]]

                nonleak_preds = torch.LongTensor([x == nonleak_gold_labels_[i] for i, x in enumerate(nonleak_preds_)])
                nonleak_gold_labels = torch.ones(len(nonleak_gold_labels_)).long()

                nonleak_perf_metrics = get_step_metrics(nonleak_preds, nonleak_gold_labels, model_class.perf_metrics)
                nonleak_perf_metrics = get_epoch_metrics(model_class.perf_metrics)
                log_data_to_neptune(model_class, nonleak_perf_metrics['acc'], 'nonleak_pred_acc', 'metric', 'epoch', split)
                log_data_to_neptune(model_class, torch.tensor(len(nonleak_preds_)), 'num_nonleak_pred', 'metric', 'epoch', split)
            else:
                log_data_to_neptune(model_class, torch.tensor(0), 'num_nonleak_pred', 'metric', 'epoch', split)

    if outputs[0].get('correct_pred_label') and outputs[0].get('leak_pred_label'):

        # Log results for correct+leak preds
        correct_leak_preds_ = [preds_[i] for i in range(len(preds_)) if correct_pred_label_[i] and leak_pred_label_[i]]
        if correct_leak_preds_:
            correct_leak_gold_labels_ = [gold_labels_[i] for i in range(len(gold_labels_)) if correct_pred_label_[i] and leak_pred_label_[i]]

            correct_leak_preds = torch.LongTensor([x == correct_leak_gold_labels_[i] for i, x in enumerate(correct_leak_preds_)])
            correct_leak_gold_labels = torch.ones(len(correct_leak_gold_labels_)).long()

            correct_leak_perf_metrics = get_step_metrics(correct_leak_preds, correct_leak_gold_labels, model_class.perf_metrics)
            correct_leak_perf_metrics = get_epoch_metrics(model_class.perf_metrics)
            log_data_to_neptune(model_class, correct_leak_perf_metrics['acc'], 'correct_leak_pred_acc', 'metric', 'epoch', split)
            log_data_to_neptune(model_class, torch.tensor(len(correct_leak_preds_)), 'num_correct_leak_pred', 'metric', 'epoch', split)
        else:
            log_data_to_neptune(model_class, torch.tensor(0), 'num_correct_leak_pred', 'metric', 'epoch', split)

        # Log results for correct+nonleak preds
        correct_nonleak_preds_ = [preds_[i] for i in range(len(preds_)) if correct_pred_label_[i] and not leak_pred_label_[i]]
        if correct_nonleak_preds_:
            correct_nonleak_gold_labels_ = [gold_labels_[i] for i in range(len(gold_labels_)) if correct_pred_label_[i] and not leak_pred_label_[i]]

            correct_nonleak_preds = torch.LongTensor([x == correct_nonleak_gold_labels_[i] for i, x in enumerate(correct_nonleak_preds_)])
            correct_nonleak_gold_labels = torch.ones(len(correct_nonleak_gold_labels_)).long()

            correct_nonleak_perf_metrics = get_step_metrics(correct_nonleak_preds, correct_nonleak_gold_labels, model_class.perf_metrics)
            correct_nonleak_perf_metrics = get_epoch_metrics(model_class.perf_metrics)
            log_data_to_neptune(model_class, correct_nonleak_perf_metrics['acc'], 'correct_nonleak_pred_acc', 'metric', 'epoch', split)
            log_data_to_neptune(model_class, torch.tensor(len(correct_nonleak_preds_)), 'num_correct_nonleak_pred', 'metric', 'epoch', split)
        else:
            log_data_to_neptune(model_class, torch.tensor(0), 'num_correct_nonleak_pred', 'metric', 'epoch', split)

        # Log results for incorrect+leak preds
        incorrect_leak_preds_ = [preds_[i] for i in range(len(preds_)) if not correct_pred_label_[i] and leak_pred_label_[i]]
        if incorrect_leak_preds_:
            incorrect_leak_gold_labels_ = [gold_labels_[i] for i in range(len(gold_labels_)) if not correct_pred_label_[i] and leak_pred_label_[i]]

            incorrect_leak_preds = torch.LongTensor([x == incorrect_leak_gold_labels_[i] for i, x in enumerate(incorrect_leak_preds_)])
            incorrect_leak_gold_labels = torch.ones(len(incorrect_leak_gold_labels_)).long()

            incorrect_leak_perf_metrics = get_step_metrics(incorrect_leak_preds, incorrect_leak_gold_labels, model_class.perf_metrics)
            incorrect_leak_perf_metrics = get_epoch_metrics(model_class.perf_metrics)
            log_data_to_neptune(model_class, incorrect_leak_perf_metrics['acc'], 'incorrect_leak_pred_acc', 'metric', 'epoch', split)
            log_data_to_neptune(model_class, torch.tensor(len(incorrect_leak_preds_)), 'num_incorrect_leak_pred', 'metric', 'epoch', split)
        else:
            log_data_to_neptune(model_class, torch.tensor(0), 'num_incorrect_leak_pred', 'metric', 'epoch', split)

        # Log results for incorrect+nonleak preds
        incorrect_nonleak_preds_ = [preds_[i] for i in range(len(preds_)) if not correct_pred_label_[i] and not leak_pred_label_[i]]
        if incorrect_nonleak_preds_:
            incorrect_nonleak_gold_labels_ = [gold_labels_[i] for i in range(len(gold_labels_)) if not correct_pred_label_[i] and not leak_pred_label_[i]]

            incorrect_nonleak_preds = torch.LongTensor([x == incorrect_nonleak_gold_labels_[i] for i, x in enumerate(incorrect_nonleak_preds_)])
            incorrect_nonleak_gold_labels = torch.ones(len(incorrect_nonleak_gold_labels_)).long()

            incorrect_nonleak_perf_metrics = get_step_metrics(incorrect_nonleak_preds, incorrect_nonleak_gold_labels, model_class.perf_metrics)
            incorrect_nonleak_perf_metrics = get_epoch_metrics(model_class.perf_metrics)
            log_data_to_neptune(model_class, incorrect_nonleak_perf_metrics['acc'], 'incorrect_nonleak_pred_acc', 'metric', 'epoch', split)
            log_data_to_neptune(model_class, torch.tensor(len(incorrect_nonleak_preds_)), 'num_incorrect_nonleak_pred', 'metric', 'epoch', split)
        else:
            log_data_to_neptune(model_class, torch.tensor(0), 'num_incorrect_nonleak_pred', 'metric', 'epoch', split)