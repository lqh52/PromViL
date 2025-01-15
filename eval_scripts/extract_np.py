import benepar
import spacy
import nltk
from nltk.tree import Tree
from utils.abstract_noun import ABSTRACT_NOUN
import fileinput
from PIL import Image
import sys
import re
from pattern.en import pluralize

gpu = spacy.prefer_gpu()
nlp = spacy.load("en_core_web_md")

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

def clean_token(text):
    return text.replace('<phrase>', '').replace('</phrase>', '').replace('<grounding>', '')

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

def pluralize_phrase(phrase):
    phrase = phrase.lower().strip()
    if phrase.startswith('a '):
        phrase = phrase[2:]
    elif phrase.startswith('an '):
        phrase = phrase[3:]
    elif phrase.startswith('the ') | phrase.startswith('its ') | phrase.startswith('his ') | phrase.startswith('her '):
        phrase = phrase[4:]
    return pluralize(phrase)
    
def _is_abs_noun(w, dataset=None):
    if dataset=='nlvr2':
        abs_noun_ls = ABSTRACT_NOUN + ['image', 'images']
        return w.lower().strip() in abs_noun_ls
    else:
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

def _valid_noun(tree, dataset=None):
    nn_words = []
    if get_np_depth(tree) >=2:
        for child in tree:
            if get_np_depth(child)==1:
                nn_words.extend(child.leaves())
    else:
        nn_words.extend(tree.leaves())
    contain_abs_noun = any(_is_abs_noun(w, dataset) for w in nn_words)
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

def _extract_np_phrases(tree, np_phrases, dataset=None):
    if isinstance(tree, Tree):
        if tree.label() == 'NP':
            if _valid_noun(tree, dataset):
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

def process_np(np_dict, cap, eval='vqa'):
    tmp_np_dict = {}
    np_dict = {k:v for k, v in sorted(np_dict.items(), key=lambda x: x[0], reverse=True) if len(v)>0}
    for i, level in enumerate(list(np_dict.keys())):
        tmp_np_dict[f'level {len(list(np_dict.keys()))-i}'] = np_dict[level]
    np_dict = tmp_np_dict
    max_level = len(list(np_dict.keys()))
    if max_level > 0:
        if eval != 'vqa':
            if cap not in [clean_token(k) for k in np_dict[f"level {max_level}"]]:
                np_dict[f"level {max_level+1}"] = [cap]
                max_level += 1
        results = {k: {} for k in list(np_dict.keys())}
        for i in range(1, max_level+1):
            for np in np_dict[f'level {i}']:
                cleaned_np = clean_token(np)
                if i==1:
                    results[f'level {i}'][cleaned_np] = {'prompt': f"<phrase>{pluralize_phrase(np)}</phrase>", 'output': ''}
                else:
                    prompt = f"<phrase>{np}</phrase>"
                    results[f'level {i}'][cleaned_np] = {'anchors': {'main_obj': '', 'others': []}, 'output': ''}
                    for small_np in np_dict['level 1']:
                        if small_np == identify_main_object_phrase(np):
                            results[f'level {i}'][cleaned_np]['anchors']['main_obj'] = clean_token(small_np)
                    for small_np in results[f'level {i-1}']:
                        if small_np in cleaned_np:
                            if small_np != results[f'level {i}'][cleaned_np]['anchors']['main_obj']:
                                results[f'level {i}'][cleaned_np]['anchors']['others'].append(small_np)
                    results[f'level {i}'][cleaned_np]['prompt'] = prompt
        return results
    else:
        return None

def extract_np(captions, model, eval='vqa', dataset=None):
    docs = model.pipe(captions)
    np_ls = []
    for doc, cap in zip(docs, captions):
        doc_output = []
        for sent in doc.sents:
            parse_str = sent._.parse_string
            tree = Tree.fromstring(parse_str)
            np_phrases = {}
            np_dict = _extract_np_phrases(tree, np_phrases, dataset)
            results = process_np(np_dict, cap, eval)
            if results is None:
                results = {'level 1': {}}
                if eval=='vqa':
                    results[f'level 1'][cap] = {'prompt': f"{clean_token(cap)}", 'output': ''}
                else:
                    results[f'level 1'][cap] = {'prompt': f"<phrase>{clean_token(cap)}</phrase>", 'output': ''}
            doc_output.append(results)
        if len(doc_output) == 1:
            doc_output = doc_output[0]
        else:
            print(f"E - {doc_output}")
            doc_output = doc_output[0]
        np_ls.append(doc_output)
    return np_ls

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
        text = input.split('<tab>')[-1].replace('<grounding><phrase>', '').replace('</phrase>', '')
        texts.append(text)
        image_paths.append(input.split('<tab>')[0].replace('[image]', ''))
    return texts, image_paths

def collate_fn(batch_inputs):
    texts = []
    images = []
    for input in batch_inputs:
        with Image.open(input['image_path']) as image:
            images.append(image.convert("RGB"))
        texts.append(input['text'])
    return texts, images

def process_text(texts, eval='vqa', dataset=None):
    texts = [clean_token(text) for text in texts]
    nlp.add_pipe("benepar", config={"model": "benepar_en3_large"})
    return extract_np(texts, nlp, eval, dataset)


