from engine.config import default
from engine.datasets import dataset_classes
import argparse

parser = argparse.ArgumentParser()

###########################
# Directory Config (modify if using your own paths)
###########################
parser.add_argument(
    "--data_dir",
    type=str,
    default=default.DATA_DIR,
    help="where the dataset is saved",
)
parser.add_argument(
    "--indices_dir",
    type=str,
    default=default.FEW_SHOT_DIR,
    help="where the (few-shot) indices is saved",
)
parser.add_argument(
    "--feature_dir",
    type=str,
    default=default.FEATURE_DIR,
    help="where to save pre-extracted features",
)
parser.add_argument(
    "--result_dir",
    type=str,
    default=default.RESULT_DIR,
    help="where to save experiment results",
)

###########################
# Dataset Config (few_shot_split.py)
###########################
parser.add_argument(
    "--dataset",
    type=str,
    default="",
    choices=dataset_classes.keys(),
    help="number of train shot",
)
parser.add_argument(
    "--train-shot",
    type=int,
    default="",
    help="number of train shot",
)
parser.add_argument(
    "--beta",
    type=float,
    default="",
    help="loss regularization",
)

parser.add_argument(
    "--seed",
    type=int,
    default="",
    help="seed number",
)

###########################
# Feature Extraction Config (features.py)
###########################

parser.add_argument(
    "--clip-encoder",
    type=str,
    default="RN50",
    choices=["ViT-B/16", "ViT-B/32", "RN50", "RN101", "RN50x4", "RN50x16"],
    help="specify the clip encoder to use",
)
parser.add_argument(
    "--image-layer-idx",
    type=int,
    default=0,
    choices=[0, 1, -1],
    help="specify how many image encoder layers to finetune. 0 means none. -1 means full finetuning.",
)
parser.add_argument(
    "--text-layer-idx",
    type=int,
    default=0,
    choices=[0, 1, -1],
    help="specify how many text encoder layers to finetune. 0 means none. -1 means full finetuning.",
)
parser.add_argument(
    "--text-augmentation",
    type=str,
    default='hand_crafted',
    choices=['hand_crafted', # tip_adapter selected
             'classname', # plain class name
             'vanilla', # a photo of a {cls}.
             'template_mining' # examples of best zero-shot templates for few-shot val set
             ],
    help="specify the text augmentation to use.",
)
parser.add_argument(
    "--image-augmentation",
    type=str,
    default='flip',
    choices=['none', # only a single center crop
             'flip', # add random flip view
             'randomcrop', # add random crop view
             ],
    help="specify the image augmentation to use.",
)
parser.add_argument(
    "--image-views",
    type=int,
    default=1,
    help="if image-augmentation is not None, then specify the number of extra views.",
)
parser.add_argument(
    "--test-batch-size",
    type=int,
    default=32,
    help="batch size for test (feature extraction and evaluation)",
)
parser.add_argument(
    "--num-workers",
    type=int,
    default=0,
    help="number of workers for dataloader",
)


###########################
# Training Config (train.py)
###########################

parser.add_argument(
    "--modality",
    type=str,
    default="cross_modal",
    choices=["cross_modal", # half batch image, half batch text
             "uni_modal", # whole batch image
    ],
    help="whether or not to perform cross-modal training (ie. half batch is image, half batch is text)",
)
parser.add_argument(
    "--classifier_head",
    type=str,
    default="linear",
    choices=["linear", # linear classifier
             "adapter", # 2-layer MLP with 0.2 residual ratio following CLIP-adapter + linear classifier
    ],
    help="classifier head architecture",
)
parser.add_argument(
    "--classifier_init",
    type=str,
    default="zeroshot",
    choices=["zeroshot", # zero-shot/one-shot-text-based initialization
             "random", # random initialization
    ],
    help="classifier head initialization",
)
parser.add_argument(
    "--logit",
    type=float,
    default=4.0, # CLIP's default logit scaling
    choices=[4.60517, # CLIP's default logit scaling
             4.0, # for partial finetuning
    ],
    help="logit scale (exp(logit) is the inverse softmax temperature)",
)
parser.add_argument(
    "--hyperparams",
    type=str,
    default="partial",
    choices=["linear", # linear hyper
             "adapter", # adapter hyper
             "partial", # partial hyper
    ],
    help="hyperparams sweep",
)