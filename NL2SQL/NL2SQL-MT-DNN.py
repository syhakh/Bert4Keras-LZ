import json
import os
import sys
import pandas as pd
import torch
from tempfile import TemporaryDirectory ##生成临时文件

from mtdnn.common.types import EncoderModelType
from mtdnn.configuration_mtdnn import MTDNNConfig
from mtdnn.data_builder_mtdnn import MTDNNDataBuilder
from mtdnn.modeling_mtdnn import MTDNNModel
from mtdnn.process_mtdnn import MTDNNDataProcess
from mtdnn.tasks.config import MTDNNTaskDefs # 自己要写
from mtdnn.tokenizer_mtdnn import MTDNNTokenizer


# define Configuration,Task,Model Objects
ROOT_DIR = TemporaryDirectory().name
OUTPUT_DIR = os.path.join(ROOT_DIR, 'checkpoint') # os.path.join() 路径进行拼接

DATA_DIR = "../../nl2sql_data/"  #数据集位置
DATA_SOURCE_DIR = os.path.join(DATA_DIR, "XXX") # os.path.join() 路径进行拼接 ../../nl2sql_data/MNLI
# Training parameters
BATCH_SIZE = 16
MULTI_GPU_ON = True
MAX_SEQ_LEN = 128
NUM_EPOCHS = 5

config = MTDNNConfig(batch_size=BATCH_SIZE, 
                     max_seq_len=MAX_SEQ_LEN, 
                     multi_gpu_on=MULTI_GPU_ON)
                     
tasks_params = {
    "NL2SQL": { # MNLI指的是数据集的名称
        "data_format": "PremiseAndOneHypothesis",
        "encoder_type": "BERT",
        "dropout_p": 0.3,
        "enable_san": True,
        "labels": ["", "AVG", "MAX", "MIN", "COUNT", "SUM", "不被select"],
        "metric_meta": ["ACC"],
        "loss": "CeCriterion", # 
        "kd_loss": "MseCriterion", # 
        "n_class": 7,
        "split_names": [
            "train",
            "dev",
            "test",
        ],
        "data_source_dir": DATA_SOURCE_DIR,
        "data_process_opts": {"header": True, "is_train": True, "multi_snli": False,},
        "task_type": "Classification",
    },
}

task_defs = MTDNNTaskDefs(tasks_params) # MTDNNTaskDefs

tokenizer = MTDNNTokenizer(do_lower_case=True)

## Load and build data
data_builder = MTDNNDataBuilder(
    tokenizer=tokenizer,
    task_defs=task_defs,
    data_dir=DATA_SOURCE_DIR,
    canonical_data_suffix="canonical_data", # 规范数据后缀
    dump_rows=True,
)

## Build data to MTDNN Format
## Iterable of each specific task and processed data
vectorized_data = data_builder.vectorize()

# Make the Data Preprocess step and update the config with training data updates
data_processor = MTDNNDataProcess(
    config=config, task_defs=task_defs, vectorized_data=vectorized_data
)

train_dataloader_list = data_processor.get_train_dataloader() # 
dev_dataloaders_list = data_processor.get_dev_dataloaders()
test_dataloaders_list = data_processor.get_test_dataloaders()
print(train_dataloader_list)
print(dev_dataloaders_list)
print(test_dataloaders_list)


decoder_opts = data_processor.get_decoder_options_list()
task_types = data_processor.get_task_types_list()
dropout_list = data_processor.get_tasks_dropout_prob_list()

loss_types = data_processor.get_loss_types_list()
kd_loss_types = data_processor.get_kd_loss_types_list()
tasks_nclass_list = data_processor.get_task_nclass_list()

num_all_batches = data_processor.get_num_all_batches()

model = MTDNNModel(
    config,
    task_defs,
    pretrained_model_name="bert-base-uncased",
    num_train_step=num_all_batches,
    decoder_opts=decoder_opts,
    task_types=task_types,
    dropout_list=dropout_list,
    loss_types=loss_types,
    kd_loss_types=kd_loss_types,
    tasks_nclass_list=tasks_nclass_list,
    multitask_train_dataloader=train_dataloader_list,
    dev_dataloaders_list=dev_dataloaders_list,
    test_dataloaders_list=test_dataloaders_list,
    output_dir=OUTPUT_DIR,
    log_dir=LOG_DIR 
)


model.fit(epochs=NUM_EPOCHS)
