# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The WDC Product Data Corpus and Gold Standard for Large-Scale Product Matching - Version 2.0."""

import json

import datasets

_CITATION = """\
@inproceedings{primpeli2019wdc,
  title={The WDC training dataset and gold standard for large-scale product matching},
  author={Primpeli, Anna and Peeters, Ralph and Bizer, Christian},
  booktitle={Companion Proceedings of The 2019 World Wide Web Conference},
  pages={381--386},
  year={2019}
}
"""

_DESCRIPTION = """\
Many e-shops have started to mark-up product data within their HTML pages using the schema.org vocabulary. The Web Data Commons project regularly extracts such data from the Common Crawl, a large public web crawl. The Web Data Commons Training and Test Sets for Large-Scale Product Matching contain product offers from different e-shops in the form of binary product pairs (with corresponding label "match" or "no match")

In order to support the evaluation of machine learning-based matching methods, the data is split into training, validation and test set. We provide training and validation sets in four different sizes for four product categories. The labels of the test sets were manually checked while those of the training sets were derived using shared product identifiers from the Web via weak supervision.

The data stems from the WDC Product Data Corpus for Large-Scale Product Matching - Version 2.0 which consists of 26 million product offers originating from 79 thousand websites.
"""

_HOMEPAGE = "http://webdatacommons.org/largescaleproductcorpus/v2/index.html"

_LICENSE = ""

_BASE_DATA_PAT_FORMAT_STR = "{category}/"

class Products2017Config(datasets.BuilderConfig):
    """The WDC Product Data Corpus and Gold Standard for Large-Scale Product Matching - Version 2.0."""

    def __init__(self, name, category: str, **kwargs):
        """BuilderConfig for WDC Products 2017.
        Args:
            category (str): The product category and training set size.
        """

        size = name.split('_')[1]
        # Initialize the base class.

        super(Products2017Config, self).__init__(
            name=name, **kwargs
        )

        # Additional attributes
        self.name = name
        self.category = category
        self.size = size
        self.base_data_path = _BASE_DATA_PAT_FORMAT_STR.format(
            category=category
        )

class Products2017(datasets.GeneratorBasedBuilder):
    """The WDC Product Data Corpus and Gold Standard for Large-Scale Product Matching - Version 2.0."""

    VERSION = datasets.Version("2.1.0")

    BUILDER_CONFIG_CLASS = Products2017Config

    BUILDER_CONFIGS = [
        Products2017Config(
            name='computers_xlarge',
            category='computers',
            version=VERSION,
            description="The computers xlarge dataset part of Products-2017"),
        Products2017Config(
            name='computers_large',
            category='computers',
            version=VERSION,
            description="The computers large dataset part of Products-2017"),
        Products2017Config(
            name='computers_medium',
            category='computers',
            version=VERSION,
            description="The computers medium dataset part of Products-2017"),
        Products2017Config(
            name='computers_small',
            category='computers',
            version=VERSION,
            description="The computers small dataset part of Products-2017"),
        Products2017Config(
            name='cameras_xlarge',
            category='cameras',
            version=VERSION,
            description="The cameras xlarge dataset part of Products-2017"),
        Products2017Config(
            name='cameras_large',
            category='cameras',
            version=VERSION,
            description="The cameras large dataset part of Products-2017"),
        Products2017Config(
            name='cameras_medium',
            category='cameras',
            version=VERSION,
            description="The cameras medium dataset part of Products-2017"),
        Products2017Config(
            name='cameras_small',
            category='cameras',
            version=VERSION,
            description="The cameras small dataset part of Products-2017"),
        Products2017Config(
            name='watches_xlarge',
            category='watches',
            version=VERSION,
            description="The watches xlarge dataset part of Products-2017"),
        Products2017Config(
            name='watches_large',
            category='watches',
            version=VERSION,
            description="The watches large dataset part of Products-2017"),
        Products2017Config(
            name='watches_medium',
            category='watches',
            version=VERSION,
            description="The watches medium dataset part of Products-2017"),
        Products2017Config(
            name='watches_small',
            category='watches',
            version=VERSION,
            description="The watches small dataset part of Products-2017"),
        Products2017Config(
            name='shoes_xlarge',
            category='shoes',
            version=VERSION,
            description="The shoes xlarge dataset part of Products-2017"),
        Products2017Config(
            name='shoes_large',
            category='shoes',
            version=VERSION,
            description="The shoes large dataset part of Products-2017"),
        Products2017Config(
            name='shoes_medium',
            category='shoes',
            version=VERSION,
            description="The shoes medium dataset part of Products-2017"),
        Products2017Config(
            name='shoes_small',
            category='shoes',
            version=VERSION,
            description="The shoes small dataset part of Products-2017"),
        
    ]

    DEFAULT_CONFIG_NAME = "computers_medium"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):

        features = datasets.Features(
            {
                "pair_id": datasets.Value("string"),
                "label": datasets.Value("int32"),
                "id_left": datasets.Value("int32"),
                "category_left": datasets.Value("string"),
                "cluster_id_left": datasets.Value("int32"),
                "brand_left": datasets.Value("string"),
                "title_left": datasets.Value("string"),
                "description_left": datasets.Value("string"),
                "price_left": datasets.Value("string"),
                "specTableContent_left": datasets.Value("string"),
                "id_right": datasets.Value("int32"),
                "category_right": datasets.Value("string"),
                "cluster_id_right": datasets.Value("int32"),
                "brand_right": datasets.Value("string"),
                "title_right": datasets.Value("string"),
                "description_right": datasets.Value("string"),
                "price_right": datasets.Value("string"),
                "specTableContent_right": datasets.Value("string"),
            }
        )

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        main_path = self.config.base_data_path
        size = self.config.size
        relevant_files = [f'{main_path}train_{size}.json.gz', f'{main_path}valid_{size}.json.gz', f'{main_path}test.json.gz']
        
        data_dir = dl_manager.download_and_extract(relevant_files)
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir[0],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir[2],
                    "split": "test"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir[1],
                    "split": "validation",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        with open(filepath, encoding="utf-8") as f:
            for key, row in enumerate(f):
                data = json.loads(row)
                yield key, {
                    "pair_id": data["pair_id"],
                    "label": data["label"],
                    "id_left": data["id_left"],
                    "category_left": data["category_left"],
                    "cluster_id_left": data["cluster_id_left"],
                    "brand_left": data["brand_left"],
                    "title_left": data["title_left"],
                    "description_left": data["description_left"],
                    "price_left": data["price_left"],
                    "specTableContent_left": data["specTableContent_left"],
                    "id_right": data["id_right"],
                    "category_right": data["category_right"],
                    "cluster_id_right": data["cluster_id_right"],
                    "brand_right": data["brand_right"],
                    "title_right": data["title_right"],
                    "description_right": data["description_right"],
                    "price_right": data["price_right"],
                    "specTableContent_right": data["specTableContent_right"]
                }