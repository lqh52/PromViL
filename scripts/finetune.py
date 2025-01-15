from transformers import AutoProcessor, EarlyStoppingCallback, Kosmos2ForConditionalGeneration, Kosmos2Config
from peft import PeftModel, PeftConfig, AutoPeftModelForCausalLM
import os
import sys
import random
import pandas as pd
import numpy as np
import json
from transformers import TrainingArguments, BitsAndBytesConfig, default_data_collator, DataCollatorForLanguageModeling
from transformers.trainer import Trainer
from datasets import load_dataset, Dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
import bitsandbytes as bnb
import torch
from scripts.dataloader import Kosmos2DataCollator


SEED = 5020
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

DEVICE = "cuda"
BASE_MODEL = "microsoft/kosmos-2-patch14-224"
EPOCHS = 1
LORA_RANK = 64
WARMUP_RATIO = 0.1
TUNING_LAYER = sys.argv[1]
CHECKPOINT = None
if TUNING_LAYER == 'feed_forward':
    BATCH_SIZE = 8
    MICRO_BATCH_SIZE = 8
    LEARNING_RATE = 5e-4
else:
    BATCH_SIZE = 4
    MICRO_BATCH_SIZE = 1
    LEARNING_RATE = 1e-4
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
OUTPUT_DIR = f"./kosmos2-ep{EPOCHS}-lr{LEARNING_RATE}-lora{LORA_RANK}-bz{BATCH_SIZE}-mbz{MICRO_BATCH_SIZE}-{TUNING_LAYER}"

print("Logging model in: ", OUTPUT_DIR)

def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    return bnb_config

def create_peft_config(target_modules, modules_to_save, r, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"):
    """
    Creates Parameter-Efficient Fine-Tuning configuration for the model

    :param r: LoRA attention dimension
    :param lora_alpha: Alpha parameter for LoRA scaling
    :param modules: Names of the modules to apply LoRA to
    :param lora_dropout: Dropout Probability for LoRA layers
    :param bias: Specifies if the bias parameters should be trained
    """
    config = LoraConfig(
        r = r,
        lora_alpha = r/2,
        target_modules = target_modules,
        lora_dropout = lora_dropout,
        bias = bias,
        task_type = task_type,
        modules_to_save=modules_to_save
    )

    return config

def print_trainable_parameters(model, use_4bit = False):
    """
    Prints the number of trainable parameters in the model.

    :param model: PEFT model
    """

    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    if use_4bit:
        trainable_params /= 2

    print(
        f"All Parameters: {all_param:,d} || Trainable Parameters: {trainable_params:,d} || Trainable Parameters %: {100 * trainable_params / all_param}"
    )

def find_all_linear_names(peft_model, int4=False, int8=False):
    """Find all linear layer names in the model. reference from qlora paper."""
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            # last layer is not add to lora_module_names
            if 'lm_head' in name:
                continue
            if 'output_layer' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)

if CHECKPOINT is not None:
    model = Kosmos2ForConditionalGeneration.from_pretrained(BASE_MODEL, device_map="auto")
    peft_config = CHECKPOINT
    base_with_adapters_model = PeftModel.from_pretrained(model, peft_config)
    model = base_with_adapters_model.merge_and_unload()
    modules_to_save=None
else:
    model = Kosmos2ForConditionalGeneration.from_pretrained(BASE_MODEL, device_map="auto")
    modules_to_save=None

if TUNING_LAYER == 'feed_forward':
    target_modules = ['fc1', 'fc2']
elif TUNING_LAYER == 'all_linear':
    target_modules = ['fc1', 'fc2', 'k_proj', 'out_proj', 'q_proj', 'v_proj']

processor = AutoProcessor.from_pretrained(BASE_MODEL, device_map="auto")

peft_config = create_peft_config(target_modules, modules_to_save, LORA_RANK)
lora_model = get_peft_model(model, peft_config)
print_trainable_parameters(lora_model)

train_data = load_dataset("json", data_files='/path/to/VG/train', split='train')
val_data = load_dataset("json", data_files='/path/to/VG/val', split='train')

vsr_train_data = load_dataset("json", data_files='/path/to/VSR/train', split='train')
vsr_val_data = load_dataset("json", data_files='/path/to/VSR/val', split='train')

llava_data = load_dataset("json", data_files='/path/to/LLaVA', split='train').shuffle(seed=SEED)

vg_qa_data = load_dataset("json", data_files='/path/to/VG_VQA', split='train').shuffle(seed=SEED)
vg_qa_train_data = vg_qa_data.select(range(8000))
vg_qa_val_data = vg_qa_data.select(range(8000, 8500))

train_data = concatenate_datasets([train_data, vsr_train_data, llava_data, vg_qa_train_data])

val_data = concatenate_datasets([val_data, vsr_val_data, vg_qa_val_data])
train_data = train_data.shuffle(seed=SEED)
val_data = val_data.shuffle(seed=SEED)

data_collator = Kosmos2DataCollator(processor, phrase_type=PHRASE_TYPE)

callback = EarlyStoppingCallback(4)

training_args = TrainingArguments(
    remove_unused_columns=False,
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    per_device_eval_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_ratio=WARMUP_RATIO,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    load_best_model_at_end=True,
    logging_strategy="steps",
    logging_steps=1,
    optim="adamw_torch",
    evaluation_strategy="steps",
    eval_steps=14604,
    save_steps=14604,
    save_strategy="steps",
    output_dir=OUTPUT_DIR,
    label_names=["labels"],
    log_level="info",
    # report_to="wandb"
)

trainer = Trainer(
    model=lora_model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_data,
    eval_dataset=val_data,
    callbacks=[callback],
)
lora_model.config.use_cache = False
lora_model.is_parallelizable = True
lora_model.model_parallel = True

# Start training
trainer.train()
lora_model.save_pretrained(OUTPUT_DIR, save_embedding_layers=True)

