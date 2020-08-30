import argparse
import pickle
import yaml
from functools import partial
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TestTubeLogger  # pip install test-tube

from experiment import Experiment
from models import BiLSTMAttn
from utils import NewsDataset, collate_fn

import warnings

warnings.filterwarnings(action="ignore")


parser = argparse.ArgumentParser(description="Generic runner for BiLSTMAttn models")

parser.add_argument(
    "--config",
    "-c",
    dest="filename",
    metavar="FILE",
    help="path to the config file",
    default="./config.yaml",
)

args = parser.parse_args()

with open(args.filename, "r") as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


# ----------------
# DataLoader
# ----------------
data_path = config["exp_params"]["data_path"]
vocab_path = config["exp_params"]["vocab_path"]
labels_list = ["조선일보", "동아일보", "경향신문", "한겨레"]
labels_dict = {label: idx for idx, label in enumerate(labels_list)}

with open(vocab_path, "rb") as f:
    word_index = pickle.load(f)


dataset = NewsDataset(data_path)

train_loader = DataLoader(
    dataset=dataset,
    batch_size=64,
    shuffle=True,
    collate_fn=partial(collate_fn, word_index=word_index, labels_dict=labels_dict),
)

dev_loader = DataLoader(
    dataset=dataset,
    batch_size=64,
    shuffle=False,
    collate_fn=partial(collate_fn, word_index=word_index, labels_dict=labels_dict),
)

# ----------------
# SetUp Model
# ----------------
# vocab_size & num_class
config["model_params"]["vocab_size"] = len(word_index)
config["model_params"]["num_class"] = len(labels_list)

model = BiLSTMAttn(**config["model_params"])
experiment = Experiment(model, config["exp_params"])


# ----------------
# TestTubeLogger
# ----------------
tt_logger = TestTubeLogger(
    save_dir=config["logging_params"]["save_dir"],
    name=config["logging_params"]["name"],
    debug=False,
    create_git_tag=False,
)

# ----------------
# Checkpoint
# ----------------
checkpoint_callback = ModelCheckpoint(
    filepath="./checkpoints/BilstmAttn_{epoch:02d}_{val_loss:.2f}",
    monitor="val_loss",
    verbose=True,
    save_top_k=5,
)

early_stopping = EarlyStopping(monitor="val_loss", patience=5, verbose=True)


# ----------------
# Trainer
# ----------------
runner = Trainer(
    default_save_path=f"{tt_logger.save_dir}",
    min_epochs=1,
    logger=tt_logger,
    log_save_interval=100,
    train_percent_check=1.0,
    val_percent_check=1.0,
    num_sanity_val_steps=5,
    early_stop_callback=early_stopping,
    checkpoint_callback=checkpoint_callback,
    **config["trainer_params"],
)

# ----------------
# Start Train
# ----------------
runner.fit(experiment, train_loader, dev_loader)
