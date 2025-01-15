import copy
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

import torch
import torch.utils.data
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from prettytable import PrettyTable

import re
import json

from utils.box_ops import box_iou
from utils.decode_string import decode_bbox_from_caption

def clean_answer(answer):
    answer = answer.replace('<phrase>', '')
    answer = answer.strip()
    token_idx = answer.find("</phrase>")
    if token_idx == -1:
        answer = answer.strip()
    else:
        answer = answer[:token_idx]
    stop_words = ['a ', 'an ', 'the ', 'on the ', 'in the ', 'his ', 'her ', 'their ', 'at the ', 'to the ']
    for w in stop_words:
        if answer.startswith(w):
            answer = answer[len(w):].strip()
            return answer
    return answer

def extract_answer(text):
    marker = "Short Answer:"
    start_index = text.find(marker)
    if start_index == -1:
        return ''
    end_marker = '.'
    end_index = text.find(end_marker, start_index)
    answer = text[start_index+len(marker):end_index].strip()
    return clean_answer(answer)


class RefExpEvaluatorFromTxt(object):
    def __init__(self, refexp_gt_path, k=(1, -1), thresh_iou=0.5):
        assert isinstance(k, (list, tuple))
        with open(refexp_gt_path, 'r') as f:
            self.refexp_gt = json.load(f)
        self.img_ids = [item['id'] for item in self.refexp_gt['images']]
        print(f"Load {len(self.img_ids)} images")
        print(f"Load {len(self.refexp_gt['annotations'])} annotations")
        self.k = k
        self.thresh_iou = thresh_iou

    def summarize(self,
                  prediction_file: str,
                  quantized_size: int = 32,
                  line_number: int = 100000,
                  verbose: bool = False,):
        
        # get the predictions
        with open(prediction_file, 'r', encoding='utf-8') as f:
            predict_all_lines = f.readlines()
        # filter the invaild lines for predict_all_lines
        filter_prediction_lines = []
        count = 0
        for line in predict_all_lines:
            if line.startswith('H-'):
                line_id = int(line.split(' ')[0].replace('H-', ''))
                filter_prediction_lines.append(line)
                if count == line_number:
                        break
                else:
                    count += 1
                
        
        predict_all_lines = filter_prediction_lines
        predict_index = 0
        
        dataset2score = {
            "refcoco": {k: 0.0 for k in self.k},
            "refcoco+": {k: 0.0 for k in self.k},
            "refcocog": {k: 0.0 for k in self.k},
        }
        dataset2count = {"refcoco": 0.0, "refcoco+": 0.0, "refcocog": 0.0}
        for item_img, item_ann in tqdm(zip(self.refexp_gt['images'], self.refexp_gt['annotations'])):
            # quit when evaluating all predictions
            if predict_index == len(predict_all_lines):
                break
                
            if item_img['id'] != item_ann['image_id']:
                raise ValueError(f"Ann\n{item_ann} \nis not matched\n {item_img}")
            
            dataset_name = item_img['dataset_name']
            img_height = item_img['height']
            img_width = item_img['width']
            caption = item_img['caption']
            target_bbox = item_ann["bbox"]
            converted_bbox = [
                target_bbox[0],
                target_bbox[1],
                target_bbox[2] + target_bbox[0],
                target_bbox[3] + target_bbox[1],
            ]
            target_bbox = torch.as_tensor(converted_bbox).view(-1, 4)
            
            
            prediction_line = predict_all_lines[predict_index]
            predict_index += 1
            
            collect_entity_location = decode_bbox_from_caption(prediction_line, quantized_size=quantized_size, verbose=verbose)

            predict_boxes = []
            for (p_pred, p_x1, p_y1, p_x2, p_y2) in collect_entity_location:
                pred_box = [p_x1 * img_width, p_y1 * img_height, p_x2 * img_width, p_y2 * img_height]
                predict_boxes.append(pred_box)
            
            if len(predict_boxes) == 0:
                print(f"Can't find valid bbox for the given phrase {caption}, \n{collect_entity_location}")
                print(f"We set a 0-area box to calculate result")
                predict_boxes = [[0., 0., 0., 0.]]
                
            predict_boxes = torch.as_tensor(predict_boxes).view(-1, 4)
            
            iou, _ = box_iou(predict_boxes, target_bbox)
            mean_iou, _ = box_iou(predict_boxes.mean(0).view(-1, 4), target_bbox)
            for k in self.k:
                if k == 'upper bound':
                    if max(iou) >= self.thresh_iou:
                        dataset2score[dataset_name][k] += 1.0
                    else:
                        print("Caption: ", caption)
                        print("Prompt: ", prediction_line.strip())
                        print("Target box: ", target_bbox)
                elif k == 'mean':
                    if max(mean_iou) >= self.thresh_iou:
                        dataset2score[dataset_name][k] += 1.0
                else:
                    if max(iou[0, :k]) >= self.thresh_iou:
                        dataset2score[dataset_name][k] += 1.0

            dataset2count[dataset_name] += 1.0

        for key, value in dataset2score.items():
            for k in self.k:
                try:
                    value[k] /= dataset2count[key]
                except:
                    pass
        
        print(dataset2score[dataset_name])
        return

class GQAEvaluatorFromTxt(object):
    def __init__(self, vqa_gt_path):
        with open(vqa_gt_path, 'r') as f:
            self.vqa_gt = json.load(f)
        if isinstance(self.vqa_gt, list):
            self.answers = {i:data_['answer'] for i, data_ in enumerate(self.vqa_gt)}
        else:
            self.answers = {i:data_['answer'] for i, (key, data_) in enumerate(self.vqa_gt.items())}
    
    def summarize(self,
                  prediction_file: str,
                  line_number: int = 100000,
                  verbose: bool = False,):
        
        # get the predictions
        with open(prediction_file, 'r', encoding='utf-8') as f:
            predict_all_lines = f.readlines()
        # filter the invaild lines for predict_all_lines
        filter_prediction_lines = []
        count = 0
        for line in predict_all_lines:
            if line.startswith('H-'):
                line_id = int(line.split(' ')[0].replace('H-', ''))
                filter_prediction_lines.append(line)
                if count == line_number:
                        break
                else:
                    count += 1
        predict_all_lines = filter_prediction_lines
        predict_index = 0
        score = 0

        sample_count = 0
        for i, answer in self.answers.items():
            if predict_index == len(predict_all_lines):
                break
            prediction_line = predict_all_lines[predict_index]
            predict_index += 1
            predicted_answer = extract_answer(prediction_line)
            if predicted_answer == answer:
                score += 1
            else:
                print(f"Full Pred: {prediction_line.strip()}")
                print(f"Pred: {predicted_answer}")
                print(f"Ground Truth: {answer}\n")
            sample_count += 1
        accuracy = score/sample_count
        print(f"Accuracy: {accuracy}")
        return

def add_missing_object_tag(text):
    # Split the text into segments
    segments = re.split(r'(</phrase>)', text)
    
    result = []
    for i in range(len(segments)):
        result.append(segments[i])
        
        # Check if this segment is </phrase> and it's not the last segment
        if segments[i] == '</phrase>' and i + 1 < len(segments):
            # If the next segment doesn't start with <object>, add it
            if not segments[i+1].strip().startswith('<object>'):
                result.append('<object>')
    
    return ''.join(result)

class NLVR2EvaluatorFromTxt(object):
    def __init__(self, nlvr_gt_path):
        with open(nlvr_gt_path, 'r') as f:
            self.nlvr_gt = [json.loads(line) for line in f]
        if isinstance(self.nlvr_gt, list):
            self.answers = {i:data_['label'] for i, data_ in enumerate(self.nlvr_gt)}
        else:
            self.answers = {i:data_['label'] for i, (key, data_) in enumerate(self.nlvr_gt.items())}
    
    def summarize(self,
                  prediction_file: str,
                  line_number: int = 100000,
                  verbose: bool = False,):
        
        # get the predictions
        with open(prediction_file, 'r', encoding='utf-8') as f:
            predict_all_lines = f.readlines()
        # filter the invaild lines for predict_all_lines
        filter_prediction_lines = []
        count = 0
        for line in predict_all_lines:
            if line.startswith('H-'):
                line_id = int(line.split(' ')[0].replace('H-', ''))
                filter_prediction_lines.append(line)
                if count == line_number:
                        break
                else:
                    count += 1
        predict_all_lines = filter_prediction_lines
        predict_index = 0
        score = 0

        sample_count = 0
        for i, answer in self.answers.items():
            if predict_index == len(predict_all_lines):
                break
            prediction_line = predict_all_lines[predict_index]
            predict_index += 1
            predicted_answer = extract_answer(prediction_line)
            if predicted_answer == answer.lower():
                score += 1
            else:
                print(f"Full Pred: {prediction_line.strip()}")
                print(f"Pred: {predicted_answer}")
                print(f"Ground Truth: {answer.lower()}\n")
            sample_count += 1
        accuracy = score/sample_count
        print(f"Accuracy: {accuracy}")
        return

def add_missing_object_tag(text):
    # Split the text into segments
    segments = re.split(r'(</phrase>)', text)
    
    result = []
    for i in range(len(segments)):
        result.append(segments[i])
        
        # Check if this segment is </phrase> and it's not the last segment
        if segments[i] == '</phrase>' and i + 1 < len(segments):
            # If the next segment doesn't start with <object>, add it
            if not segments[i+1].strip().startswith('<object>'):
                result.append('<object>')
    
    return ''.join(result)

class Visual7WEvaluatorFromTxt(object):
    def __init__(self, v7w_gt_path):
        self.v7w_gt = []
        with open(v7w_gt_path, 'r') as f:
            for line in f:
                self.v7w_gt.append(json.loads(line))
        self.bboxes_info = json.load(open('path/to/visual7w/bboxes.json', 'r'))
        self.labels = {i:{'choices': data_['multiple_choice'], 'answer': data_['answer'], 'height': data_['image_height'], 'width': data_['image_width']} for i, data_ in enumerate(self.v7w_gt)}
    
    def summarize(self,
                  prediction_file: str,
                  line_number: int = 100000,
                  verbose: bool = False):
        
        # get the predictions
        with open(prediction_file, 'r', encoding='utf-8') as f:
            predict_all_lines = f.readlines()
        # filter the invaild lines for predict_all_lines
        filter_prediction_lines = []
        count = 0
        for line in predict_all_lines:
            if line.startswith('H-'):
                line_id = int(line.split(' ')[0].replace('H-', ''))
                filter_prediction_lines.append(line)
                if count == line_number:
                        break
                else:
                    count += 1
        predict_all_lines = filter_prediction_lines
        predict_index = 0
        score = 0

        sample_count = 0
        for i, item_img in self.labels.items():
            img_height = item_img['height']
            img_width = item_img['width']
            target_bboxes = []
            for choice in item_img['choices']:
                choice = str(choice)
                box = [self.bboxes_info[choice]['x'], self.bboxes_info[choice]['y'], self.bboxes_info[choice]['w'], self.bboxes_info[choice]['h']]
                converted_bbox = [
                    box[0],
                    box[1],
                    box[2] + box[0],
                    box[3] + box[1],
                ]
                target_bboxes.append(converted_bbox)
            correct_box = target_bboxes[-1]
            target_bboxes = torch.as_tensor(target_bboxes).view(-1, 4)
            if predict_index == len(predict_all_lines):
                break
            prediction_line = predict_all_lines[predict_index]
            predict_index += 1
            
            prediction_line = add_missing_object_tag(prediction_line)
            collect_entity_location = decode_bbox_from_caption(prediction_line, quantized_size=32, verbose=verbose)

            predict_boxes = []
            for (p_pred, p_x1, p_y1, p_x2, p_y2) in collect_entity_location:
                pred_box = [p_x1 * img_width, p_y1 * img_height, p_x2 * img_width, p_y2 * img_height]
                predict_boxes.append(pred_box)
            
            if len(predict_boxes) == 0:
                print(f"Can't find valid bbox for the given phrase, \n{collect_entity_location}")
                print(f"We set a 0-area box to calculate result")
                predict_boxes = [[0., 0., 0., 0.]]
            
            predict_box = predict_boxes[0]
            predict_box = torch.as_tensor(predict_box).view(-1, 4)

            iou, _ = box_iou(target_bboxes, predict_box)
            model_choice = torch.argmax(iou, dim=0)
            
            if model_choice == 3:
                score += 1
            else:
                print(f"Full Pred: {prediction_line.strip()}")
                print(f"Ground Truth: {correct_box}\n")
            sample_count += 1
        accuracy = score/sample_count
        print(f"Accuracy: {accuracy}")
        return



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('prediction_file', help='prediction_file')
    parser.add_argument('annotation_file', default='/path/to/mdetr_processed_json_annotations', help='annotation_file')
    parser.add_argument('line_number', default=10000, type=int)
    parser.add_argument('--type', default='ref', type=str)
    parser.add_argument('--quantized_size', default=32, type=int)
    
    args = parser.parse_args()

    if args.type == 'ref':
        evaluator = RefExpEvaluatorFromTxt(
            refexp_gt_path=args.annotation_file, 
            k=(1, 'mean', 'upper bound'), 
            thresh_iou=0.5,
        )
        
        evaluator.summarize(args.prediction_file, args.quantized_size, args.line_number, verbose=False)
    elif args.type == 'vqa':
        evaluator = GQAEvaluatorFromTxt(
            vqa_gt_path=args.annotation_file, 
        )
        evaluator.summarize(args.prediction_file, args.line_number, verbose=False)
    elif args.type == 'v7w':
        evaluator = Visual7WEvaluatorFromTxt(
            v7w_gt_path=args.annotation_file, 
        )
        evaluator.summarize(args.prediction_file, args.line_number, verbose=False)
    elif args.type == 'nlvr2':
        evaluator = NLVR2EvaluatorFromTxt(
            nlvr_gt_path=args.annotation_file, 
        )
        evaluator.summarize(args.prediction_file, args.line_number, verbose=False)
        