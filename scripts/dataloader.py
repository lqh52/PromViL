from typing import List
from datasets import load_dataset, Dataset
from PIL import Image
import torch
import pandas as pd
import os
import json
import re
import random


data_dir = 'path/to/original_VG/data/'
image_data_dir = os.path.join(data_dir, 'images/')
attr_data = json.load(open(os.path.join(data_dir, 'attributes.json')))
image_data = json.load(open(os.path.join(data_dir, 'image_data.json')))
image_data = pd.DataFrame.from_records(image_data)

def _mask_prompt_tokens_and_get_labels_and_inputs_ids(labels, phrase_type, datasets):
    """
    Make labels_ids as a copy of the input_ids. Then mask (set to -100) the tokens until the text/label separator. Return the input_ids until the text/label separator.
    """
    masked_label_ids = []
    assert len(labels.numpy().tolist()) == len(datasets)
    for label_id, dataset in zip(labels.numpy().tolist(), datasets):
        if phrase_type in ['multi_phrase']:
            sep_token = [20032, 55]
            start_index = next((i for i, x in enumerate(label_id) if label_id[i:i+len(sep_token)] == sep_token), None)
        elif phrase_type == 'full_phrase':
            if dataset in ['vg', 'vsr']:
                sep_token = [64007]
                start_index = len(label_id) - label_id[::-1].index(sep_token[0]) - 1
            else:
                sep_token = [20032, 55]
                start_index = next((i for i, x in enumerate(label_id) if label_id[i:i+len(sep_token)] == sep_token), None) + 2
        elif phrase_type == 'small_phrase':
            if dataset in ['llava', 'vg_qa']:
                sep_token = [20032, 55]
                start_index = next((i for i, x in enumerate(label_id) if label_id[i:i+len(sep_token)] == sep_token), None) + 2
            else:
                sep_token = [64007]
                start_index = len(label_id) - label_id[::-1].index(sep_token[0]) - 1


        # If the indices are found
        if start_index is not None:
            masked_label_id = torch.tensor([[-100]*(start_index) + label_id[start_index:]]).to(labels.device)
        else:
            raise ValueError("Cannot fine 'Answer:'" )
        masked_label_ids.append(masked_label_id)
    return torch.stack(masked_label_ids).squeeze(1)

class Kosmos2DataCollator:
    def __init__(self, processor, phrase_type='multi_phrase'):
        self.processor = processor
        self.phrase_type = phrase_type
    
    def __call__(self, examples: List[dict]):
        texts = []
        bboxes_lses = []
        image_paths = []
        box_obj_matchs = []
        datasets = []
        for example in examples:
            texts.extend(example['text'])
            if example['bboxes'] is not None:
                bboxes_lses.extend(example['bboxes'])
                box_obj_matchs.extend(example['box_obj_match'])
            else:
                bboxes_lses.append(None)
                box_obj_matchs.append(None)
            image_paths.extend([example['image_path']]*len(example['text']))
            datasets.extend([example['dataset']]*len(example['text']))
        bboxes = []
        for boxes_ls, box_obj_match_ls in zip(bboxes_lses, box_obj_matchs):
            tmp_bboxes = []
            if boxes_ls is None:
                bboxes.append(None)
            elif len(box_obj_match_ls) == 0:
                bboxes.append([[tuple(box) for box in boxes_ls]])
            else:
                for i in list(set(box_obj_match_ls)):
                    tmp_ls = []
                    for _i, box in enumerate(boxes_ls):
                        if box_obj_match_ls[_i] == i:
                            tmp_ls.append(tuple(box))
                    tmp_bboxes.append(tmp_ls)
                bboxes.append(tmp_bboxes)
        images = []
        for image_path in image_paths:
            with Image.open(image_path) as image:
                images.append(image.convert('RGB'))
        try:
            inputs = self.processor(text=texts, images=images, bboxes=bboxes, return_tensors="pt")
            labels = inputs['input_ids'].clone()
            labels[inputs['input_ids'] == 1] = -100
            inputs['labels'] = _mask_prompt_tokens_and_get_labels_and_inputs_ids(labels, phrase_type=self.phrase_type, datasets=datasets)
        except Exception as e:
            print(e)
            print(texts)
            print(bboxes)
            print(examples)
            print(datasets)
        return inputs