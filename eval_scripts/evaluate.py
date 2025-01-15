from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM, Kosmos2ForConditionalGeneration
from peft import PeftModel
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
import pandas as pd
from PIL import Image
import sys
import os
import re
import fileinput
from extract_np import process_text, pluralize_phrase

DEVICE = "cuda"

def clean_token(text, token_to_clean):
    if 'phrase' in token_to_clean:
        text = text.replace('<phrase>', '').replace('</phrase>', '')
    
    if 'grounding' in token_to_clean:
        text = text.replace('<grounding>', '')
    return text

def buffered_read(input, buffer_size):
    buffer = []
    with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer

def input_process(inputs):
    texts = []
    image_paths = []
    for input in inputs:
        text = input.split('<tab>')[-1]
        texts.append(text)
        image_paths.append(input.split('<tab>')[0].replace('[image]', ''))
    return texts, image_paths

def input_process_level_1(inputs):
    texts = []
    image_paths = []
    ids = []
    org_texts = []
    for inp in inputs:
        if '[level 1]' in inp:
            inp = inp.replace('[level 1]', '')
            text = f"{inp.split('<tab>')[-1]}"
            org_text = f"{inp.split('<tab>')[1]}"
            texts.append(text)
            org_texts.append(org_text)
            image_paths.append(inp.split('<tab>')[0].split('[image]')[-1])
            ids.append(inp.split('[image]')[0])
    return ids, org_texts, texts, image_paths

def collate_fn(batch_inputs):
    texts = []
    images = []
    for input in batch_inputs:
        with Image.open(input['image_path']) as image:
            images.append(image.convert("RGB"))
        texts.append(input['text'])
    return texts, images


def run(eval_locout, buffer_size, batch_size, checkpoint_path=None):
    BASE_MODEL = "microsoft/kosmos-2-patch14-224"
    if checkpoint_path=='None':
        model = Kosmos2ForConditionalGeneration.from_pretrained(BASE_MODEL, device_map='auto')
    else:
        model = Kosmos2ForConditionalGeneration.from_pretrained(BASE_MODEL, device_map="auto")
        model = PeftModel.from_pretrained(model=model, model_id=checkpoint_path, device_map="auto")
        model = model.merge_and_unload()
    processor = AutoProcessor.from_pretrained(BASE_MODEL, device_map="auto")
    for i, inputs in enumerate(buffered_read(eval_locout, buffer_size)):
        ids = [i*buffer_size + j for j in range(buffer_size)]
        texts, image_paths = input_process(inputs)
        infer(ids, texts, image_paths, batch_size, model, processor)

def prepare_input_for_model(extracted_phrases, images_list, level):
    text_inputs = []
    dict_keys = []
    img_inputs = []
    for i, (phrase_dict, img) in enumerate(zip(extracted_phrases, images_list)):
        _dict_keys = []
        _text_list = []
        if f'level {level}' in phrase_dict:
            for k, v in phrase_dict[f'level {level}'].items():
                if level == 1:
                    prompt = f"<grounding>{v['prompt']}<object>"
                    prompt = clean_token(prompt, ['noun', 'rel'])
                    _text_list.append(prompt)
                    _dict_keys.append([i, k])
                else:
                    prompt = f'<grounding>'
                    add_request = False
                    for anchor in v['anchors']['others']:
                        if level == 2:
                            if '<Cannot find>' not in phrase_dict[f'level {level - 1}'][anchor]['output']:
                                anchor_text = f"<phrase><noun>{anchor}</noun></phrase>{phrase_dict[f'level {level - 1}'][anchor]['output']}; "
                            else:
                                anchor_text = ''
                        else:
                            if '<Cannot find>' not in phrase_dict[f'level {level - 1}'][anchor]['output']:
                                anchor_text = f"{phrase_dict[f'level {level - 1}'][anchor]['prompt']}{phrase_dict[f'level {level - 1}'][anchor]['output']}, "
                            else:
                                anchor_text = ''
                        if (anchor_text != '') & (not add_request):
                            prompt += 'We can see in the image: '
                            add_request = True
                        prompt += anchor_text
                    if prompt != '<grounding>':
                        prompt = prompt[:-2] + f". Base on that, we can detect: {v['prompt']}<object>"
                    else:
                        prompt += f"{v['prompt']}<object>"
                    prompt = clean_token(prompt, ['noun', 'rel'])
                    _text_list.append(prompt)
                    _dict_keys.append([i, k])
        text_inputs.extend(_text_list)
        img_inputs.extend([img]*len(_text_list))
        dict_keys.extend(_dict_keys)
    return text_inputs, img_inputs, dict_keys

def process_output(extracted_phrases, text_inputs, generated_texts, dict_keys, level, P=32):
    for key, inp_text, out_text in zip(dict_keys, text_inputs, generated_texts):
        model_output = out_text.split('</image>')[-1]
        if 'Base on that, we can detect:' in model_output:
            model_output = model_output.split('Base on that, we can detect:')[1]
        obj_out_ls = list(re.finditer(r"<object>((?:<patch_index_\d+><patch_index_\d+></delimiter_of_multi_objects/>)*<patch_index_\d+><patch_index_\d+>)</object>", string=model_output))
        if len(obj_out_ls) > 0:
            obj_out = obj_out_ls[0].group()
            patch_index_pairs = obj_out.replace('<object>', '').replace('</object>', '').split('</delimiter_of_multi_objects/>')
            out = []
            for pair in patch_index_pairs:  
                # Extract the xxxx and yyyy values from the patch_index pair  
                x = re.search(r'<patch_index_(\d+)>', pair)  
                y = re.search(r'<patch_index_(\d+)>', pair[1:])                
                if x and y:
                    _x = int(x.group(1))
                    _y = int(y.group(1))
                    ul_x = _x % P  
                    ul_y = _x // P  
                    lr_x = _y % P  
                    lr_y = _y // P
                    if (lr_x-ul_x)*(lr_y-ul_y) / P**2 > 0.8:
                        continue
                    else:
                        out.append(f'<patch_index_{x.group(1)}><patch_index_{y.group(1)}>')
            if len(out) > 0:
                out = '</delimiter_of_multi_objects/>'.join(out)
                obj_out = f"<object>{out}</object>"
            else:
                obj_out = f"<Cannot find>. Output: {model_output}"
        else:
            obj_out = f"<Cannot find>. Output: {model_output}"
        idx, key = key
        extracted_phrases[idx][f'level {level}'][key]['output'] = obj_out
    return extracted_phrases
        
        
def infer(ids, texts, image_paths, batch_size, model, processor, model_base=None):
    inputs = Dataset.from_dict({'image_path': image_paths, 'text': texts})
    dataloader = DataLoader(inputs, batch_size=batch_size, collate_fn=collate_fn)
    output_ls = []
    for batch_inputs in dataloader:
        texts, images = batch_inputs
        extracted_phrases = process_text(texts, eval='ref')
        max_level_in_batch = max([len(phrase_dict) for phrase_dict in extracted_phrases])
        for level in range(1, max_level_in_batch+1):
            if (level == 1) & (model_base is not None):
                text_inputs, img_inputs, dict_keys = prepare_input_for_model(extracted_phrases, images, level)
                tmp_inputs = processor(text=text_inputs, images=img_inputs, return_tensors="pt")
                generated_ids = model_base.generate(
                    pixel_values=tmp_inputs["pixel_values"].to(DEVICE),
                    input_ids=tmp_inputs["input_ids"].to(DEVICE),
                    attention_mask=tmp_inputs["attention_mask"].to(DEVICE),
                    image_embeds=None,
                    image_embeds_position_mask=tmp_inputs["image_embeds_position_mask"].to(DEVICE),
                    use_cache=True,
                    max_new_tokens=32
                )
            else:
                text_inputs, img_inputs, dict_keys = prepare_input_for_model(extracted_phrases, images, level)
                tmp_inputs = processor(text=text_inputs, images=img_inputs, return_tensors="pt")
                generated_ids = model.generate(
                    pixel_values=tmp_inputs["pixel_values"].to(DEVICE),
                    input_ids=tmp_inputs["input_ids"].to(DEVICE),
                    attention_mask=tmp_inputs["attention_mask"].to(DEVICE),
                    image_embeds=None,
                    image_embeds_position_mask=tmp_inputs["image_embeds_position_mask"].to(DEVICE),
                    use_cache=True,
                    max_new_tokens=32
                )
            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            extracted_phrases = process_output(extracted_phrases, text_inputs, generated_texts, dict_keys, level)
        output_ls.extend(extracted_phrases)

    for id, extracted_phrase in zip(ids, output_ls):
        max_level = len(list(extracted_phrase.keys()))
        phrase = list(extracted_phrase[f'level {max_level}'].keys())[0]
        output = extracted_phrase[f'level {max_level}'][phrase]['output']
        print(f"S-{id} {extracted_phrase}")
        print(f"H-{id} <phrase>{phrase}</phrase>{output}")
        
if __name__ == "__main__":
    # sys.argv[1] lets use call arguments from shell script
    # note we start from 1 instead of 0
    checkpoint_path = sys.argv[1]
    eval_file = sys.argv[2]
    buffer_size = int(sys.argv[3])
    batch_size = int(sys.argv[4])
    run(eval_file, buffer_size, batch_size, checkpoint_path)
