# Are Machine Rationales (Not) Useful to Humans? Measuring and Improving Human Utility of Free-Text Rationales

This is the code and associated datasets for the paper titled 

>[Are Machine Rationales (Not) Useful to Humans? Measuring and Improving Human Utility of Free-Text Rationales. *Brihi Joshi\*, Ziyi Liu\*, Sahana Ramnath, Aaron Chan, Zhewei Tong, Shaoliang Nie, Qifan Wang, Yejin Choi, Xiang Ren*](https://aclanthology.org/2023.acl-long.392/)

accepted at [ACL 2023](https://2023.aclweb.org/).

If you end up using this code or the data, please cite our paper: 

```
@inproceedings{joshi-etal-2023-machine,
    title = "Are Machine Rationales (Not) Useful to Humans? Measuring and Improving Human Utility of Free-text Rationales",
    author = "Joshi, Brihi  and
      Liu, Ziyi  and
      Ramnath, Sahana  and
      Chan, Aaron  and
      Tong, Zhewei  and
      Nie, Shaoliang  and
      Wang, Qifan  and
      Choi, Yejin  and
      Ren, Xiang",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.392",
    pages = "7103--7128",
    abstract = "Among the remarkable emergent capabilities of large language models (LMs) is free-text rationalization; beyond certain scale, large LMs are capable of generating seemingly useful rationalizations, which in turn, can dramatically enhance their performances on leaderboards. This phenomenon raises a question: can machine generated rationales also be useful for humans, especially when lay humans try to answer questions based on those machine rationales? We observe that human utility of existing rationales is far from satisfactory and expensive to estimate with human studies. Existing metrics like task performance of the LM generating the rationales or similarity between generated and gold rationales are not good indicators of their human utility. While we observe that certain properties of rationales like conciseness and novelty are correlated with their human utility, estimating them without human involvement is challenging. We show that, by estimating a rationale{'}s helpfulness in answering similar unseen instances, we can measure its human utility to a better extent. We also translate this finding into an automated score, Gen-U, that we propose, which can help improve LMs{'} ability to generate rationales with better human utility, while maintaining most of its task performance. Lastly, we release all code and collected data with this project.",
}
```

---------

## Setting up the environment

Make a folder titled ```ftru``` and place this GitHub repo __inside__ that folder.

### Requirements Setup

```conda create -n ftru python=3.8.11```

```conda activate ftru```

```conda install pytorch==1.7.1 torchvision torchaudio cudatoolkit=10.1 -c pytorch```

```pip install -r requirements.txt```

## Running T5 Few Shot Rationalising Baselines

__promt_type__ are different prompt types for self-rationalization that are taken from the [FEB Benchmark](https://aclanthology.org/2022.findings-naacl.31/).

### Building Dataset

Build the dataset for settings which do not have datasets.

#### Building the training set

Note: There are two commands here. The first one builds the entire train set, and the second one selects 48 shots from the dataset. Both are required for the self-rationalizing setup.

```
python scripts/build_dataset.py \
    --dataset strategyqa \
    --gen_mode I-OR \
    --arch t5-base \
    --prompt_type infilling \
    --split train

python scripts/build_dataset.py \
    --dataset strategyqa \
    --gen_mode I-OR \
    --arch t5-base \
    --prompt_type infilling \
    --split train \
    --num_samples 48 \
    --seed 0
```

#### Building the dev set

This setting will generate the dev set in the given prompt format.

```
python scripts/build_dataset.py \
    --dataset strategyqa \
    --gen_mode I-OR \
    --arch t5-base \
    --prompt_type infilling \
    --split dev 
```

#### Building the test set

This setting will generate the test set with presampled 200 instances in the given prompt format.

```
python scripts/build_dataset.py \
    --dataset strategyqa \
    --gen_mode I-OR \
    --arch t5-base \
    --prompt_type infilling \
    --split test \
    --presample 200
```

### Training

Things that need to change - ```data.prompt_type``` and ```model.arch```

```
srun --gres=gpu:2080:1 -t 30 python main.py -m \
data=strategyqa \
data.gen_mode=I-OR \
data.incontext=None \
data.prompt_type=infilling \
data.num_train=48 \
data.num_train_seed=0 \
data.presample=200 \
setup.train_batch_size=4 \
setup.eval_batch_size=4 \
setup.num_workers=3 \
setup.accumulate_grad_batches=1 \
setup.eff_train_batch_size=4 \
model=lm \
model.arch=t5-base \
model.optimizer.lr=3e-5 \
trainer.max_epochs=25 \
trainer.log_every_n_steps=6 \
trainer.check_val_every_n_epoch=25 \
training.patience=25 \
seed=0 
```


