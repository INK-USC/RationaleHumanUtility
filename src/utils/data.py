dataset_info = {
    'strategyqa': {
        'train': ['train', 1648],
        'dev': ['dev', 184],
        'test': ['test', 458],
        'raw_test':['raw_test',490],
        'num_classes': 2,
        'classes': ['False', 'True']
    },
    'openbookqa': {
        'train': ['train', 4957],
        'dev': ['validation', 500],
        'test': ['test', 500],
        'num_classes': 4,
        'classes': ['A', 'B', 'C', 'D']
    },
    'qed': {
        'train': ['train', 4638],
        'dev': ['val', 516],
        'test': ['test', 1021],
    },
}

monitor_dict = {
    'strategyqa': 'dev_acc_metric_epoch',
    'openbookqa': 'dev_acc_metric_epoch',
    'qed': 'dev_acc_metric_epoch',
}

data_keys = ['item_idx', 'input_ids', 'attention_mask', 'labels', 'gold_label', 'prompt_text', 'output_label_text']