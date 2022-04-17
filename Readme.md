# NER-Transformers-ArmanPers

In this repository, a complete fine-tuning process of `HooshvareLab/roberta-fa-zwnj-base` over `ArmanPers` dataset
using `Transformers` library is presented. `ArmanPers` is a name entity recognition dataset which is
for Persian language and is publicly available through a GitHub repository. Although this is a public dataset,
Transformers's datasets library does not include it so a custom dataset is required which is provided in this repository
as well.

## Download & Prepare dataset

Download and unzip the dataset:

```commandline
wget https://raw.githubusercontent.com/HaniehP/PersianNER/master/ArmanPersoNERCorpus.zip
unzip ArmanPersoNERCorpus.zip
```

6 files with the following files will be generated in your directory:

- train_fold1.txt
- train_fold2.txt
- train_fold3.txt
- test_fold1.txt
- test_fold2.txt
- test_fold3.txt

As it is obvious, there are 3 folds for the train and the test datasets. Combine the folds for each dataset using the
following commands:

```commandline
 python prepare_dataset.py --in_files train_fold1.txt train_fold2.txt train_fold3.txt --out_path train.txt
 python prepare_dataset.py --in_files test_fold1.txt test_fold2.txt test_fold3.txt --out_path test.txt
```
To have a validation sample run the following code:
```commandline
python split_dataset.py --in_file train.txt --part_1 train.txt --part_2 val.txt
```

## Train
To train the model use the following notebook
```commandline
ArmanPers-NER.ipynb
```

## References
1. https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb#scrollTo=n9qywopnIrJH
2. https://github.com/HaniehP/PersianNER
3. 