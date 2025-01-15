from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

import pandas as pd
from itertools import islice
from tqdm import tqdm
import re
from torch.utils.data import Dataset

import sys
import time

model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
pipe = transformers.pipeline(
    "text-generation",
    device_map='auto',
    model=model_name,
    model_kwargs={"torch_dtype": torch.float16, "load_in_4bit": False},
)
pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id

class PromptDataset(Dataset):
    def __init__(self, prompt_list):
        super().__init__()
        self.prompt_list = prompt_list

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, i):
        return self.prompt_list[i]

def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch

def load_data():
    path = f'input/data.json'
    data = pd.read_json(path, orient='records', lines=True)
    return data

def extract_phrase(generated_text, gen_type):
    sign1 = f"<{gen_type}>"
    sign2 = f"</{gen_type}>"
    pattern = re.compile(re.escape(sign1) + "([\s\S]*?)" + re.escape(sign2))
    matches = pattern.findall(generated_text)
    matches = [text.replace("\n", "") for text in matches]
    return matches

def infer(batch_size, max_new_tokens):
    data = load_data()
    prompts = data['prompt'].to_list()

    results = []
    prompt_dataset = PromptDataset(prompts)
    for generated_output in tqdm(pipe(prompt_dataset, batch_size=batch_size, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_k=50, top_p=0.95, pad_token_id=tokenizer.eos_token_id), total=len(prompt_dataset)):
        output = generated_output[0]['generated_text']
        output = output.split('[/INST]')[-1].split('* Given Phrases:')[0].strip()
        gen_type = 'phrase'
        output = extract_phrase(output, gen_type=gen_type)
        results.append(output)
    
    output_column = "generated_phrase"
    data[output_column] = results
    data = data.explode(output_column)
    data.to_json(f'output_dir/phrase_data.json', orient='records', lines=True, index=False)

if __name__ == "__main__":
    # sys.argv[1] lets use call arguments from shell script
    # note we start from 1 instead of 0
    batch_size = int(sys.argv[3])
    max_new_tokens = int(sys.argv[4])
    print('File number:')
    start = time.time()
    results = infer(batch_size, max_new_tokens)
    end = time.time()
    print(end - start)   
