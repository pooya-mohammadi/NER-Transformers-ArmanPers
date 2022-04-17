import datasets
import os
from pathlib import Path
from datasets import ClassLabel, DownloadConfig

logger = datasets.logging.get_logger(__name__)

_CITATION = ""
_DESCRIPTION = """\

"""


class ArmanPersConfig(datasets.BuilderConfig):
    """BuilderConfig for ArmanPers"""

    def __init__(self, **kwargs):
        """BuilderConfig for ArmanPers.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(ArmanPersConfig, self).__init__(**kwargs)


class ArmanPers(datasets.GeneratorBasedBuilder):
    """ArmanPers dataset."""

    BUILDER_CONFIGS = [
        ArmanPersConfig(name="ArmanPers", version=datasets.Version("1.0.0"),
                        description="ArmanPers Name Entity Recognition dataset"),
    ]

    def __init__(self,
                 *args,
                 cache_dir=None,
                 train_file="train.txt",
                 val_file="val.txt",
                 test_file="test.txt",
                 ner_tags=('B-event',
                           'B-fac',
                           'B-loc',
                           'B-org',
                           'B-pers',
                           'B-pro',
                           'I-event',
                           'I-fac',
                           'I-loc',
                           'I-org',
                           'I-pers',
                           'I-pro',
                           'O'),
                 **kwargs):
        self._ner_tags = ner_tags
        self._train_file = train_file
        self._val_file = val_file
        self._test_file = test_file
        super(ArmanPers, self).__init__(*args, cache_dir=cache_dir, **kwargs)

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=sorted(list(self._ner_tags))
                        )
                    )
                }
            ),
            supervised_keys=None,
            homepage="",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": f"{self._train_file}",
            "dev": f"{self._val_file}",
            "test": f"{self._test_file}",
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = []
            ner_tags = []
            for line in f:
                if line == "" or line == "\n":
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "ner_tags": ner_tags,
                        }
                        guid += 1
                        tokens = []
                        ner_tags = []
                else:
                    # ArmanPers tokens are space separated
                    splits = line.split(" ")
                    tokens.append(splits[0])
                    ner_tags.append(splits[1].rstrip())
            # last example
            yield guid, {
                "id": str(guid),
                "tokens": tokens,
                "ner_tags": ner_tags,
            }


class HFArmanPersDataset:
    """
    """
    NAME = "HFArmanPersDataset"

    def __init__(self, train_path, val_path, test_path, cache_dir=None):
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
        self._dataset = ArmanPers(cache_dir=cache_dir,
                                  train_file=train_path,
                                  val_file=val_path,
                                  test_file=test_path)
        print("Cache1 directory: ", self._dataset.cache_dir)
        self._dataset.download_and_prepare(download_config=DownloadConfig(cache_dir=cache_dir))
        self._dataset = self._dataset.as_dataset()

    @property
    def dataset(self):
        return self._dataset

    @property
    def labels(self) -> ClassLabel:
        return self._dataset['train'].features['ner_tags'].feature.names

    @property
    def id2label(self):
        return dict(list(enumerate(self.labels)))

    @property
    def label2id(self):
        return {v: k for k, v in self.id2label.items()}

    def train(self):
        return self._dataset['train']

    def test(self):
        return self._dataset["test"]

    def validation(self):
        return self._dataset["validation"]


if __name__ == '__main__':
    dataset = HFArmanPersDataset(train_path="train.txt", val_path="val.txt", test_path="test.txt").dataset

    print(dataset['train'])
    print(dataset['test'])
    print(dataset['validation'])

    print("List of tags: ", dataset['train'].features['ner_tags'].feature.names)

    print("First sample: ", dataset['train'][0])
