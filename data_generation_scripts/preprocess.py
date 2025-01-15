import json
import os
import random
import re
from tqdm import tqdm
import pandas as pd
import os
import sys
import time
import itertools

random.seed(52)

def extract_phrase(generated_text, gen_type):
    sign1 = f"<{gen_type}>"
    sign2 = f"</{gen_type}>"
    pattern = re.compile(re.escape(sign1) + "([\s\S]*?)" + re.escape(sign2))
    matches = pattern.findall(generated_text)
    matches = [text.replace("\n", "") for text in matches]
    return matches

def make_prompt_1(phrases):
    pattern = "[^A-Za-z0-9()]+$"
    phrases = re.sub (pattern, "", phrases)
    phrases = phrases .lower()
    prompt = f"""
    You are a helpful linguistic assistance. The following phrases refer a same object, but maybe in different words. Your task is combining the given phrases to make long compound nouns in between two tag <phrase> </phrase>. Do not repeat above phrases, no explanation needed. Use only the information from the given phrases, do not invent any connections between the objects. Generate phrases as much natural as possible.
    * Given Phrases:  spoon on top brown placemat; red paper napkin under placemat (In this context: brown placemat, placemat refer to the same object) <end>
    Result: <phrase>spoon on top the placemat above the paper napkin</phrase>
    * Given Phrases:  black cat with a red hat; black cat is on the wall (In this context: black cat refer to the same object) <end>
    Result: <phrase>black cat with a red hat on the wall</phrase>
    * Given Phrases: a man with black suit and blue tie; a man with black suit (In this context: a man refer to the same object) <end>
    Result: <phrase>the tie of a man with black suit</phrase>
    * Given Phrases: a woman with flower; a woman with a black dress and a blue hat (In this context: a woman refer to the same object) <end>
    Result: <phrase>the flower of a woman with black dress and blue hat</phrase>
    * Given Phrases: a brown cow on the mountain; flowers on the mountain (In this context: the moutain refer to the same object) <end>
    Result: <phrase>a brown cow on the mountain with flowers</phrase>
    * Given Phrases: a man with white tshirt and blue hat; player holding a bat and a glove (In this context: player, man refer to the same object) <end>
    Result: <phrase>player with white tshirt holding a bat</phrase>
    * Given Phrases: a big cat on the desk; a green apple on the table (In this context: desk, table refer to the same object) <end>
    Result: <phrase>a big cat on the table with a green apple</phrase>
    * Given Phrases: a man holding a cake; a little child reaching for the birthday cake (In this context: cake, birthday cake refer to the same object) <end>
    Result: <phrase>a little child reaching for the cake that the man holding</phrase>
    * Given Phrases: a man wearing a black hat; the shirt of the man (In this context: the man refer to the same object) <end>
    Result: <phrase>the shirt of the man wearing a black hat</phrase>
    * Given Phrases: the ear of the monkey; a monkey with a banana (In this context: the monkey refer to the same object) <end>
    Result: <phrase>the ear of the monkey with a banana</phrase>
    * Given Phrases: the man next to a woman; a woman holding a bag (In this context: a woman refer to the same object) <end>
    Result: <phrase>the man next to a woman holding a bag</phrase>
    * Given Phrases: the car behind a bicycle; the red bicycle on the sidewalk (In this context: bicycle refer to the same object) <end>
    Result: <phrase>the car behind the red bicycle on the sidewalk</phrase>
    * Given Phrases: {phrases} <end>
    """
    prompt = "[INST] " + prompt.strip() + " [/INST]"
    return prompt.strip()

data_dir = './visual_genome/data/'
image_data_dir = os.path.join(data_dir, 'images/')

attr_data = json.load(open(os.path.join(data_dir, 'attributes.json')))
reg_gr_data = json.load(open(os.path.join(data_dir, 'region_graphs.json')))

REGION_LIMIT = 100

def is_sub_list(list1, list2):
    return all(element in list2 for element in list1)

def check_obj(list_obj, list_attr):
    return any(element in list_attr for element in list_obj)

def valid_region(region, relationships, filtered_obj):
    if ('cloud' in region['phrase']) | ('sky' in region['phrase']):
        return False

    obj_id_reg = [a['object_id'] for a in region['objects']]
    if len(region['relationships']) < 1:
        return False
    elif not check_obj(obj_id_reg, filtered_obj):
        return False
    else:
        return True

def elements_appearing_more_than_twice(lst):
    counts = {}  # Dictionary to store counts of elements
    result = []  # List to store elements appearing more than twice

    # Count occurrences of each element in the list
    for item in lst:
        counts[item] = counts.get(item, 0) + 1

    # Iterate over the counts dictionary to find elements appearing more than twice
    for item, count in counts.items():
        if count >= 2:
            result.append(item)

    return result


def preprocess():
    results = pd.DataFrame()
    for id in tqdm(len(attr_data)):
        object_ls = []
        for a in attr_data[id]['attributes']:
            object_ls.append(a['object_id'])
        filtered_obj = list(set(object_ls))
        regions = []
        object_ids = []
        regs_with_rel = []
        relationships = []
        for reg in reg_gr_data[id]['regions']:
            if valid_region(reg, relationships, filtered_obj):
                regs_with_rel.append(reg)
                relationships.extend(relationships)

        region_ids = [reg['region_id'] for reg in regs_with_rel]
        region_ids = random.sample(region_ids, min(REGION_LIMIT, len(region_ids)))
        for reg in regs_with_rel:
            if reg['region_id'] in region_ids:
                regions.append(reg)
                object_ids.extend(list(set([a['object_id'] for a in reg['objects']])))
        
        multi_appear = elements_appearing_more_than_twice(object_ids)
        object_dict = {obj['object_id']: {'object_id': obj['object_id'],'names': obj['names'], 'box': [obj['x'], obj['y'], obj['h'], obj['w']]} for obj in attr_data[id]['attributes']}
        if len(multi_appear) > 0:
            combine_reg_phrase_by_obj = {}

            for reg in sorted(regions, key=lambda x: x['relationships'][0]['subject_id']):
                reg_object = []
                for obj in reg['objects']:
                    obj_dict = object_dict[obj['object_id']]
                    obj_dict['region_id'] = reg['region_id']
                    reg_object.append(obj_dict)
                for obj in reg['objects']:
                    if (obj['object_id'] in multi_appear) & (object_dict[obj['object_id']]['names'][0] in reg['phrase']):
                        key = obj['object_id']
                        if key in combine_reg_phrase_by_obj:
                            new_objs = [obj for obj in reg_object if obj not in combine_reg_phrase_by_obj[key]['objects']]
                            if len(new_objs) < 1:
                                continue
                            else:
                                combine_reg_phrase_by_obj[key]['regions'].update({reg['region_id']: {'box': [reg['x'], reg['y'], reg['width'], reg['height']], 'phrase': reg['phrase']}})
                                combine_reg_phrase_by_obj[key]['objects'].extend(new_objs)
                        else:
                            combine_reg_phrase_by_obj[key] = {'regions': {}}
                            combine_reg_phrase_by_obj[key]['common_objects'] = object_dict[key]
                            combine_reg_phrase_by_obj[key]['regions'].update({reg['region_id']: {'box': [reg['x'], reg['y'], reg['width'], reg['height']], 'phrase': reg['phrase']}})
                            combine_reg_phrase_by_obj[key]['objects'] = reg_object

            unlabeled_data = list(combine_reg_phrase_by_obj.values())
            to_generate_data = []
            for data in unlabeled_data:
                tmp = {k: v for k, v in data.items() if k != 'phrases'}
                obj_names_str = ", ".join(data['common_objects']['names'])
                obj_names_str = f"In this context: {obj_names_str} refer to the same object"
                phrase_ls = [a['phrase'] for _, a in data['regions'].items()]
                if len(list(set(phrase_ls))) >= 2:
                    region_pair_ls = [list(a) for a in list(itertools.combinations(list(data['regions'].keys()), 2))]
                    for region_pair in region_pair_ls:
                        phrase_pair = [data['regions'][r]['phrase'] for r in region_pair]
                        prompt = "; ".join(phrase_pair)
                        prompt = prompt + f"; ({obj_names_str})"
                        prompt = make_prompt_1(prompt)
                        tmp['prompt'] = prompt
                        tmp["image_id"] = id
                        to_generate_data.append(tmp)
            results = pd.concat([results, pd.DataFrame.from_records(to_generate_data)], ignore_index=True) 
            results = results.drop_duplicates(subset=['prompt', 'image_id']) 

        else:
            continue 

        results.to_json(f'output/data.json', orient='records', lines=True, index=False)

if __name__ == "__main__":
    # sys.argv[1] lets use call arguments from shell script
    # note we start from 1 instead of 0
    preprocess()

