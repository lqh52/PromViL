# PromViL

<font size='5'>**Progressive Multi-granular Alignments for Grounded Reasoning in Large Vision-Language Models**</font>

## Getting Started
### Installation

**1. Environment**
```bash
pip install requirements.txt
```

**2. Data Generation**

Download Visual Genome dataset: https://homes.cs.washington.edu/~ranjay/visualgenome/api.html. Save it to dir: ./visual_genome/data/
Run preprocess VG data (find region with common object):
```bash
python data_generation_scripts/preprocess.py
```

Run data generation:
```bash
batch_size=4
max_new_token=64
python data_generation_scripts/LLM_infer.py $batch_size $max_new_token
```

Run bounding box assignment:
```bash
python data_generation_scripts/RE_labeling.py
```

**2. Model Finetuning**
See training data sample in: data/data_sample.json

Model Finetuning:
```bash
target_layer="all_linear"
python scripts/finetune.py $target_layer
```

**2. Model Evaluation**
Download Model Checkpoint: https://drive.google.com/file/d/1Gr6ZkKkP08GpxINWTSbWAoogJGmi39E-/view?usp=sharing

Referring Expression:
```bash
checkpoint=""
eval_file="eval_data/finetune_refcocog_test.txt"
buffer_size=10
batch_size=1
output_dir=""
python eval_scripts/evaluate.py $checkpoint $eval_file $buffer_size $batch_size > $output_dir
```

VQA:
```bash
checkpoint_path=""
eval_file="eval_data/gqa_testdev.txt"
buffer_size=10
batch_size=1
dataset="gqa"
split="testdev"
python eval_scripts/evaluate.py $checkpoint $eval_file $buffer_size $batch_size
```

Calculate metrics:
```bash
model_output=""
label_file="eval_data/finetune_refcocog_test.json"
output_dir=""
python eval_scripts/metrics_calc.py $model_output $label_file 100000 --quantized_size 32 > $output_dir
```


## Acknowledgement

Our implementation is based on: 
+ [Kosmos-2](https://github.com/microsoft/unilm/tree/master/kosmos-2)
+ [Huggingface-implementation] (https://github.com/huggingface/transformers/tree/v4.44.0/src/transformers/models/kosmos2) 

