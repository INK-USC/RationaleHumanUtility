from transformers import get_scheduler

no_decay = ['bias', 'LayerNorm.weight']


def setup_optimizer_params(model, optimizer):
    optimizer_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': optimizer.weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        },
    ]

    return optimizer_parameters

def setup_scheduler(scheduler, total_steps, optimizer):
    if scheduler.warmup_updates > 1.0:
        warmup_steps = int(scheduler.warmup_updates)
    else:
        warmup_steps = int(total_steps *
                            scheduler.warmup_updates)
    print(
        f'\nTotal steps: {total_steps} with warmup steps: {warmup_steps}\n')

    scheduler = get_scheduler(
        "linear", optimizer=optimizer,
        num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    scheduler = {
        'scheduler': scheduler,
        'interval': 'step',
        'frequency': 1
    }
    return scheduler