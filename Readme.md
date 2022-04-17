# NER-Transformers-ArmanPers

In this repository, a complete fine-tuning process of `HooshvareLab/roberta-fa-zwnj-base` over `ArmanPers` dataset
using `Transformers` library is presented. `ArmanPers` is a name entity recognition dataset which is
for Persian language and is publicly available through a GitHub repository. Although this is a public dataset,
Transformers's datasets library does not include it so a custom dataset is required which is provided in this repository
as well.

## Install requirements
```commandline
pip install -r requirements.txt
```

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
 python combine_datasets.py --in_files train_fold1.txt train_fold2.txt train_fold3.txt --out_path train.txt
 python combine_datasets.py --in_files test_fold1.txt test_fold2.txt test_fold3.txt --out_path test.txt
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

## Results
```
+-------+---------------+--------------+-----------+--------+-------+----------+
| Epoch | Training Loss |  Validation  | Precision | Recall |  F1   | Accuracy |
|       |               |     Loss     |           |        |       |          |
+=======+===============+==============+===========+========+=======+==========+
| 1     | no log        | 0.095        | 0.710     | 0.708  | 0.709 | 0.970    |
+-------+---------------+--------------+-----------+--------+-------+----------+
| 2     | no log        | 0.067        | 0.788     | 0.805  | 0.797 | 0.979    |
+-------+---------------+--------------+-----------+--------+-------+----------+
| 3     | 0.139         | 0.054        | 0.840     | 0.856  | 0.848 | 0.984    |
+-------+---------------+--------------+-----------+--------+-------+----------+
| 4     | 0.139         | 0.047        | 0.861     | 0.888  | 0.874 | 0.987    |
+-------+---------------+--------------+-----------+--------+-------+----------+
| 5     | 0.139         | 0.043        | 0.892     | 0.895  | 0.893 | 0.989    |
+-------+---------------+--------------+-----------+--------+-------+----------+
| 6     | 0.030         | 0.040        | 0.900     | 0.913  | 0.907 | 0.990    |
+-------+---------------+--------------+-----------+--------+-------+----------+
| 7     | 0.030         | 0.038        | 0.912     | 0.921  | 0.916 | 0.991    |
+-------+---------------+--------------+-----------+--------+-------+----------+
| 8     | 0.013         | 0.039        | 0.921     | 0.923  | 0.922 | 0.992    |
+-------+---------------+--------------+-----------+--------+-------+----------+
| 9     | 0.013         | 0.038        | 0.922     | 0.926  | 0.924 | 0.992    |
+-------+---------------+--------------+-----------+--------+-------+----------+
| 10    | 0.013         | 0.038        | 0.921     | 0.928  | 0.924 | 0.992    |
+-------+---------------+--------------+-----------+--------+-------+----------+
```
## Test
```
example = "پویا محمدی در سازمان کشاورزی چه کاری میتواند بکند ؟"
tokens = tokenizer(example)
predictions, labels, _ = trainer.predict([tokens])
predictions = np.argmax(predictions, axis=2)
true_predictions = [[label_list[p] for p in prediction] for prediction in predictions]
for t, t_p in zip(tokenizer.decode(token_ids=tokens['input_ids']).split(" "), true_predictions[0]):
    print(t, " -> ", t_p)
```
test result:
```commandline
[CLS]  ->  O
پویا  ->  B-pers
محمدی  ->  I-pers
در  ->  O
سازمان  ->  B-org
کشاورزی  ->  I-org
چه  ->  O
کاری  ->  O
میتواند  ->  O
بکند  ->  O
؟  ->  O
[SEP]  ->  O
```
## References
1. https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb#scrollTo=n9qywopnIrJH
2. https://github.com/HaniehP/PersianNER