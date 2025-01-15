import json
import benepar
import spacy
import pandas as pd
import numpy as np
import os
import sys
import warnings
import time
import nltk
from data_generation_scripts.box_utils import nms, remove_outlier, iou_calc
from nltk.tree import Tree
from nltk.corpus import wordnet
from rapidfuzz import fuzz
from tqdm import tqdm
from gensim.downloader import load
from utils.abstract_noun import ABSTRACT_NOUN
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from pattern.en import pluralize


import re
warnings.filterwarnings("ignore")
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'True'

gpu = spacy.prefer_gpu()
nlp = spacy.load("en_core_web_md")

model = SentenceTransformer('avsolatorio/GIST-Embedding-v0')

def get_sentence_transformer_embeddings(texts):
    embeddings = model.encode(texts, convert_to_tensor=True)
    return embeddings

def long_phrase_similarity(phrase1, phrase2, score_cutoff):
    vecs = get_sentence_transformer_embeddings([phrase1, phrase2])
    score = util.cos_sim(vecs[0], vecs[1])
    character_similarity_score = fuzz.ratio(phrase1, phrase2) / 100
    score = (score + character_similarity_score) / 2
    if score > score_cutoff:
        return score.item()
    else:
        return 0

data_dir = './visual_genome_python_driver/visual_genome/data/'
image_data_dir = os.path.join(data_dir, 'images/')

obj_data = json.load(open(os.path.join(data_dir, 'objects.json')))
attr_data = json.load(open(os.path.join(data_dir, 'attributes.json')))
reg_gr_data = json.load(open(os.path.join(data_dir, 'region_graphs.json')))

#Load Pretrained Glove Embedding
glove_wiki = load("glove-twitter-100")

def identify_main_object_phrase(text):
    doc = nlp(text)
    
    def find_noun_chunk(token):
        for chunk in doc.noun_chunks:
            if token in chunk:
                return chunk
        return None

    root = [token for token in doc if token.head == token][0]
    
    if root.pos_ in ["NOUN", "PRON"]:
        return find_noun_chunk(root)
    
    for token in root.rights:
        if token.pos_ in ["NOUN", "PRON"]:
            return find_noun_chunk(token)
    
    for token in root.lefts:
        if token.pos_ in ["NOUN", "PRON"]:
            return find_noun_chunk(token)
    
    return None

def normalize_bbox(bbox, img_size):
    ymin, xmin, ymax, xmax = bbox
    h, w = img_size
    return [ymin / h, xmin / w, ymax / h, xmax / w]

def convert_bbox(box):
    x, y, h, w = box
    # Calculate the coordinates of the corners
    x1 = x
    y2 = y + h
    x2 = x + w
    y1 = y
    # Return the converted bounding box
    return (float(x1), float(y1), float(x2), float(y2))

def pluralize_phrase(phrase):
    phrase = phrase.lower().strip()
    if phrase.startswith('a '):
        phrase = phrase[2:]
    elif phrase.startswith('an '):
        phrase = phrase[3:]
    elif phrase.startswith('the '):
        phrase = phrase[4:]
    return pluralize(phrase)

def get_word_vector(word):
    try:
        return glove_wiki[word]
    except KeyError:
        return np.zeros(glove_wiki.vector_size)  # Return zero vector for out-of-vocabulary words

def tokenize(text):
    # Define regular expression pattern to split text into words
    pattern = r"\w+"
    # Use the findall method to extract words from the text
    tokens = re.findall(pattern, text)
    return tokens

def get_phrase_vector(phrase):
    tokens = tokenize(phrase)
    word_vectors = np.array([get_word_vector(token.lower()) for token in tokens])
    return np.mean(word_vectors, axis=0)

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity

def similarity(phrase1, phrase2, score_cutoff):
    vec1 = get_phrase_vector(phrase1)
    vec2 = get_phrase_vector(phrase2)
    semantic_similarity_score = cosine_similarity(vec1, vec2)
    character_similarity_score = fuzz.token_set_ratio(phrase1, phrase2) / 100
    score = (semantic_similarity_score + character_similarity_score * 5) / 6
    if score > score_cutoff:
        return score
    else:
        return 0

def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return synonyms

def clean_token(text, token_to_clean):
    if 'phrase' in token_to_clean:
        text = text.replace('<phrase>', '').replace('</phrase>', '')
    
    if 'grounding' in token_to_clean:
        text = text.replace('<grounding>', '')
    return text

def clean_text(text):
    cleaned_text = re.sub(r"-[a-zA-Z]+-", " ", text)
    cleaned_text = re.sub(r"\s+(?=\'s)", "", cleaned_text)
    cleaned_text = re.sub("\s+(-|/)\s+", "\\1", cleaned_text)
    punctuations = ['?', ',', '.', ';', '!']
    pattern = re.compile(r'\s+(?=[{}])'.format(''.join(re.escape(p) for p in punctuations)))
    # Replace spaces before specified punctuations with an empty string
    cleaned_text = re.sub(pattern, '', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text

def _is_abs_noun(w):
    return w.lower().strip() in ABSTRACT_NOUN

def get_np_depth(tree):
    if not isinstance(tree, nltk.Tree):
        return 0
    # If the root is an NP, calculate the maximum level of its subtrees
    if tree.label() == 'NP':
        subtree_levels = [get_np_depth(subtree) for subtree in tree]
        return 1 + max(subtree_levels, default=0)
    # If not an NP, calculate the maximum level from the subtrees
    subtree_levels = [get_np_depth(subtree) for subtree in tree]
    return max(subtree_levels, default=0)

def _valid_noun(tree):
    nn_words = []
    for child in tree:
        if get_np_depth(child)>=2:
            for _child in child:
                tmp = ' '.join(_child.leaves())
                nn_words.append(tmp)
        else:
            tmp = ' '.join(child.leaves())
            nn_words.append(tmp)
    contain_abs_noun = any(_is_abs_noun(w) for w in nn_words)
    if not contain_abs_noun:
        return True
    else:
        return False

def check_and_or(tree, level):
    nn_words = []
    for child in tree:
        tmp = ' '.join(child.leaves())
        if (child.label() in ['NN', 'NNS', 'NP']) | (tmp=='and') | (tmp=='or'):
            nn_words.append(tmp)
    if (('and' in nn_words) | ('or' in nn_words)) & (level <= 2):
        return True, [w for w in nn_words if ((w!='and') & (w!='or'))]
    else:
        return False, None

def _extract_np_phrases(tree, np_phrases):
    if isinstance(tree, Tree):
        if tree.label() == 'NP':
            if _valid_noun(tree):
                level = get_np_depth(tree)
                if f'level {level}' in np_phrases:
                    if level >= 2:
                        have_and_or, out = check_and_or(tree, level)
                        if not have_and_or:
                            out = ' '.join(tree.leaves())
                            out = [out]
                    else:
                        have_and_or, out = check_and_or(tree, level)
                        if not have_and_or:
                            out = ' '.join(tree.leaves())
                            out = [out]
                    if have_and_or:
                        for res in out:
                            res = clean_text(res)
                            if res not in np_phrases['level 1']:
                                np_phrases['level 1'].append(res)
                    else:
                        for res in out:
                            res = clean_text(res)
                            np_phrases[f'level {level}'].append(res)
                else:
                    if level >= 2:
                        have_and_or, out = check_and_or(tree, level)
                        if not have_and_or:
                            out = ' '.join(tree.leaves())
                            out = [out]
                    else:
                        have_and_or, out = check_and_or(tree, level)
                        if not have_and_or:
                            out = ' '.join(tree.leaves())
                            out = [out]
                    if len(out) > 0:
                        np_phrases[f'level {level}'] = []
                    if 'level 1' not in np_phrases:
                        np_phrases[f'level 1'] = []
                    if have_and_or:
                        for res in out:
                            res = clean_text(res)
                            if res not in np_phrases['level 1']:
                                np_phrases['level 1'].append(res)
                    else:
                        for res in out:
                            res = clean_text(res)
                            np_phrases[f'level {level}'].append(res)
        for child in tree:
           _extract_np_phrases(child, np_phrases)
    return np_phrases

def process_np(np_dict):
    tmp_np_dict = {}
    np_dict = {k:v for k, v in sorted(np_dict.items(), key=lambda x: x[0], reverse=True) if len(v)>0}
    for i, level in enumerate(list(np_dict.keys())):
        tmp_np_dict[f'level {len(list(np_dict.keys()))-i}'] = np_dict[level]
    np_dict = tmp_np_dict
    max_level = len(list(np_dict.keys()))
    results = []
    for i in range(1, max_level+1):
        for np in np_dict[f'level {i}']:
            if i==1:
                results.append([np])
            else:
                for ls in results:
                    if ls[0] == identify_main_object_phrase(np):
                        ls_id = results.index(ls)
                        results[ls_id].append(np)
    return results

def extract_np(captions, model):
    try:
        if isinstance(captions, list):
            docs = model.pipe(captions)
            np_ls = []
            for doc in docs:
                doc_output = []
                for sent in doc.sents:
                    parse_str = sent._.parse_string
                    tree = Tree.fromstring(parse_str)
                    np_phrases = {}
                    np_dict = _extract_np_phrases(tree, np_phrases)
                    doc_output.extend(process_np(np_dict))
                np_ls.append(doc_output)
            return np_ls
        else:
            doc = model(captions)
            np_ls = []
            for sent in doc.sents:
                parse_str = sent._.parse_string
                tree = Tree.fromstring(parse_str)
                np_phrases = {}
                np_dict = _extract_np_phrases(tree, np_phrases)
                np_ls.extend(process_np(np_dict))
            return np_ls
    except Exception as e:
        print(e)
        return None, 0

def process_text():
    nlp.add_pipe("benepar", config={"model": "benepar_en3_large"})
    
    results = pd.DataFrame()
    file_path = f'input/phrase_data.json'
    unlabeled_data = pd.read_json(file_path, lines=True)
    unlabeled_data = unlabeled_data[['image_id', 'regions', 'generated_phrase']].drop_duplicates(subset=['image_id', 'generated_phrase'])#.iloc[:100,:]
    results_dict = []
    _log_results_dict = []
    count = {'1': 0, '2': 0, '3': 0}
    for _i, (_image_id, _regions, _generated_phrase) in tqdm(unlabeled_data.iterrows()):
        _phrases = [v['phrase'] for k, v in _regions.items()]
        all_phrases = list(set([_generated_phrase] + [_phrase.lower() for _phrase in _phrases]))
        all_phrases = [_phrase for _phrase in all_phrases if (_image_id, _phrase) not in _log_results_dict]
        if True:
            for _iter, iter_phrase in enumerate(all_phrases):
                if iter_phrase is None:
                    continue
                iter_phrase = iter_phrase.strip().strip('.')
                extracted_np = extract_np(iter_phrase, nlp)
                try:
                    max_level = max([len(x) for x in extracted_np])
                except (ValueError, TypeError) as e:
                    continue
                img_object_box = {}
                for obj in attr_data[int(_image_id)]['attributes']:
                    tmp_name_ls = obj['names']
                    tmp_name_ls = list(set(tmp_name_ls))
                    name_ls = []
                    if 'attributes' in obj:
                        for attr in obj['attributes']:
                            name_ls.extend([attr + ' ' + name for name in tmp_name_ls])
                    else:
                        name_ls = tmp_name_ls
                    for name in name_ls:
                        if name.lower() in img_object_box:
                            box = [obj['x'], obj['y'], obj['h'], obj['w']]
                            box = convert_bbox(box)
                            if box not in img_object_box[name.lower()]:
                                img_object_box[name.lower()].append(box)
                        else:
                            img_object_box[name.lower()] = []
                            box = [obj['x'], obj['y'], obj['h'], obj['w']]
                            box = convert_bbox(box)
                            img_object_box[name.lower()].append(box)
                phrase_box = {}
                for k, v in _regions.items():
                    box = v['box']
                    phrase = v['phrase']
                    box = [box[0], box[1], box[3], box[2]]
                    box = convert_bbox(box)
                    phrase_box[phrase] = [box]
                # filter the reion, make a dictionary with object name and bbox correspond to that region
                # assign the bbox to the object name then extend to phrase (in a region can be sure that it's rarely have multiple object of the same class)
                object_dict = {obj['object_id']: {'names': obj['names'], 'box': [obj['x'], obj['y'], obj['h'], obj['w']]} for obj in attr_data[int(_image_id)]['attributes']}
                img_region = {reg['region_id']: reg for reg in reg_gr_data[int(_image_id)]['regions']}
                _object_dict_ls = []
                for reg_id in list(_regions.keys()):
                    reg_obj_dict = {}
                    reg_object = [object_dict[obj['object_id']] for obj in img_region[int(reg_id)]['objects']]
                    for obj in reg_object:
                        name_ls = obj['names']
                        name_ls = list(set(name_ls))
                        for name in name_ls:
                            if name.lower() in reg_obj_dict:
                                box = obj['box']
                                box = convert_bbox(box)
                                if box not in reg_obj_dict[name.lower()]:
                                    reg_obj_dict[name.lower()].append(box)
                            else:
                                reg_obj_dict[name.lower()] = []
                                box = obj['box']
                                box = convert_bbox(box)
                                reg_obj_dict[name.lower()].append(box)
                    reg_obj_dict = {k: remove_outlier(v) for k, v in reg_obj_dict.items()}
                    _object_dict_ls.append(reg_obj_dict)
                
                region_object_box = {}
                for obj_dict in _object_dict_ls:
                    for name, boxes in obj_dict.items():
                        if name.lower() in region_object_box:
                            region_object_box[name.lower()].extend(boxes)
                        else:
                            region_object_box[name.lower()] = []
                            region_object_box[name.lower()].extend(boxes)

                q_tmp = {}
                for q_nps in extracted_np:
                    img_best_match = None
                    reg_best_match = None
                    phrase_best_match = None
                    try:
                        img_obj_sim = {img_obj: similarity(img_obj, q_nps[0].lower(), score_cutoff=0.8) for img_obj in list(img_object_box.keys())}
                    except:
                        continue
                    if any(np.array(list(img_obj_sim.values()))!=0):
                        img_best_match = max(list(img_obj_sim.keys()), key=lambda x: img_obj_sim[x])
                        s = 0
                        e = int(10e9)
                        box_level_0 = None
                        for level, q_np in enumerate(sorted(q_nps, key=lambda x: len(x), reverse=True)):
                            cleaned_q_np = clean_text(q_np) 
                            try:
                                start_idx = iter_phrase.lower().index(cleaned_q_np.lower(), s, e)
                            except:
                                print("Phrase: ", iter_phrase)
                                print("NP dict: ", q_np)
                                print("Cleaned: ", cleaned_q_np)
                                continue
                            end_idx = start_idx + len(q_np)

                            if abs(level - len(q_nps)) > 1:
                                box_level_0 = []

                                region_box_ls = []
                                phrase_sim = {phrase: long_phrase_similarity(phrase, clean_token(cleaned_q_np, ['grounding', 'phrase']), score_cutoff=0.85) for phrase in list(phrase_box.keys())}
                                if any(np.array(list(phrase_sim.values()))!=0):
                                    phrase_best_match = max(list(phrase_sim.keys()), key=lambda x: phrase_sim[x])
                                    region_box_ls = nms(remove_outlier(phrase_box[phrase_best_match], 'iou', 0.5)) if len(phrase_box[phrase_best_match]) > 0 else []
                                        
                                box_reg_best_match = []
                                reg_obj_sim = {reg_obj: similarity(reg_obj, q_nps[0].lower(), score_cutoff=0.8) for reg_obj in list(region_object_box.keys())}
                                if any(np.array(list(reg_obj_sim.values()))!=0):
                                    reg_best_match = max(list(reg_obj_sim.keys()), key=lambda x: reg_obj_sim[x])
                                    box_reg_best_match = remove_outlier(region_object_box[reg_best_match], 'area', 0.8) if len(region_object_box[reg_best_match]) > 0 else []
                                
                                if (len(box_reg_best_match) > 0) & (len(region_box_ls) > 0):
                                    for region_box in region_box_ls:
                                        box = max(box_reg_best_match, key=lambda x: iou_calc(x, region_box))
                                        if box not in box_level_0:
                                            box_level_0.append(box)
                                    count['1'] += 1
                                elif len(box_reg_best_match) > 0:
                                    box_level_0 = box_reg_best_match
                                    count['2'] += 1
                                else:
                                    box_level_0 = region_box_ls
                                    count['3'] += 1
                                
                                if len(box_level_0) > 0:   
                                    q_tmp[cleaned_q_np] = {"string_index": [start_idx, end_idx], "box": list(set(box_level_0)), "level": abs(level - len(q_nps))}
                                else:
                                    continue
                                                   
                            else:
                                if box_level_0 is not None:
                                    _box = box_level_0 + img_object_box[img_best_match]
                                else:
                                    _box = img_object_box[img_best_match]
                                _box = remove_outlier(_box, 'area', 0.8)
                                q_tmp[cleaned_q_np] = {"string_index": [start_idx, end_idx], "box": _box, "level": abs(level - len(q_nps))}
                            s = start_idx
                            e = end_idx
                    else:
                        continue

                if len(list(q_tmp.keys()))==0:
                    continue
                else:
                    max_level = max([q_tmp[a]['level'] for a in list(q_tmp.keys())])
                    if max_level <= 1:
                        continue
                    else:
                        clean_key = [clean_token(k, ['rel']) for k in list(q_tmp.keys())]
                        if iter_phrase not in clean_key:
                            phrase_max_level = max(list(q_tmp.keys()), key=lambda x: q_tmp[x]['level'])
                            if (iter_phrase.startswith(clean_token(phrase_max_level, ['rel']))) & (len(q_tmp[phrase_max_level]['box'])==1):
                                new_q_tmp = {iter_phrase: {"box": q_tmp[phrase_max_level]['box'], "level": q_tmp[phrase_max_level]['level']+1}}
                                for k, v in q_tmp.items():
                                    new_q_tmp[k] = v
                                q_tmp = new_q_tmp
                output = {"image_id": _image_id, "phrase": iter_phrase, "q_np_with_box": q_tmp}
                for done_phrase in list(q_tmp.keys()):
                    done_phrase = clean_token(done_phrase, ['noun', 'rel', 'grounding', 'phrase'])
                    _log_results_dict.append((_image_id, done_phrase))
                results_dict.append(output)

        results = pd.DataFrame.from_records(results_dict) 
        results.to_json(f'output/data.json', orient='records', lines=True, index=False)
        print(count)

if __name__ == "__main__":
    start = time.time()
    results = process_text()
    end = time.time()
