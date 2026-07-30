# import re
from typing import Dict, List, Tuple
# from neo4j import GraphDatabase
from io import BytesIO
from typing import Optional, Tuple
from PIL import Image
import yaml
# import os
from pathlib import Path
try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None
# import json
try:
    from IPython.display import display
except ImportError:
    display = None
# from json_to_netlist import *
import numpy as np
import shutil
# import math
# import os
# import shutil
from pathlib import Path
from PIL import Image
try:
    import cv2
except ImportError:
    cv2 = None
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import numpy as np
# import cv2 as cv
# from graph import *
import os, json

def load_lists(files_dir):
    list_dict = {}
    try: 
        if os.path.exists(f'{files_dir}/accurate_list.json'):
            with open(f'{files_dir}/accurate_list.json', 'r') as f:
                loaded = json.load(f)
                # Handle legacy list format by converting to dict, and convert string keys to int
                if isinstance(loaded, dict):
                    list_dict['accurate_list'] = {int(k): v for k, v in loaded.items()}
                else:
                    list_dict['accurate_list'] = {idx: "legacy" for idx in loaded}

        if os.path.exists(f'{files_dir}/accurate_list_old.json'):
            with open(f'{files_dir}/accurate_list_old.json', 'r') as f:
                loaded = json.load(f)
                # Handle legacy list format by converting to dict, and convert string keys to int
                if isinstance(loaded, dict):
                    list_dict['accurate_list_old'] = {int(k): v for k, v in loaded.items()}
                else:
                    list_dict['accurate_list_old'] = {idx: "legacy" for idx in loaded}
        
        if os.path.exists(f'{files_dir}/baseline_1_list.json'):
            with open(f'{files_dir}/baseline_1_list.json', 'r') as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    list_dict['baseline_1_list'] = {int(k): v for k, v in loaded.items()}
                else:
                    list_dict['baseline_1_list'] = {idx: "legacy" for idx in loaded}
        
        if os.path.exists(f'{files_dir}/baseline_1_accurate_list.json'):
            with open(f'{files_dir}/baseline_1_accurate_list.json', 'r') as f:
                loaded = json.load(f)
                # Convert to list of ints if it's a list, otherwise keep as is
                list_dict['baseline_1_accurate_list'] = [int(x) for x in loaded] if isinstance(loaded, list) else loaded
        
        if os.path.exists(f'{files_dir}/baseline_2_list.json'):
            with open(f'{files_dir}/baseline_2_list.json', 'r') as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    list_dict['baseline_2_list'] = {int(k): v for k, v in loaded.items()}
                else:
                    list_dict['baseline_2_list'] = {idx: "legacy" for idx in loaded}
        
        if os.path.exists(f'{files_dir}/baseline_2_accurate_list.json'):
            with open(f'{files_dir}/baseline_2_accurate_list.json', 'r') as f:
                loaded = json.load(f)
                list_dict['baseline_2_accurate_list'] = [int(x) for x in loaded] if isinstance(loaded, list) else loaded
        
        if os.path.exists(f'{files_dir}/baseline_3_list.json'):
            with open(f'{files_dir}/baseline_3_list.json', 'r') as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    list_dict['baseline_3_list'] = {int(k): v for k, v in loaded.items()}
                else:
                    list_dict['baseline_3_list'] = {idx: "legacy" for idx in loaded}
        
        if os.path.exists(f'{files_dir}/baseline_3_accurate_list.json'):
            with open(f'{files_dir}/baseline_3_accurate_list.json', 'r') as f:
                loaded = json.load(f)
                list_dict['baseline_3_accurate_list'] = [int(x) for x in loaded] if isinstance(loaded, list) else loaded
        
        if os.path.exists(f'{files_dir}/baseline_4_list.json'):
            with open(f'{files_dir}/baseline_4_list.json', 'r') as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    list_dict['baseline_4_list'] = {int(k): v for k, v in loaded.items()}
                else:
                    list_dict['baseline_4_list'] = {idx: "legacy" for idx in loaded}
        
        if os.path.exists(f'{files_dir}/baseline_4_accurate_list.json'):
            with open(f'{files_dir}/baseline_4_accurate_list.json', 'r') as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    list_dict['baseline_4_accurate_list'] = {int(k): v for k, v in loaded.items()}
                else:
                    list_dict['baseline_4_accurate_list'] = {int(x): "legacy" for x in loaded}

        if os.path.exists(f'{files_dir}/baseline_5_list.json'):
            with open(f'{files_dir}/baseline_5_list.json', 'r') as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    list_dict['baseline_5_list'] = {int(k): v for k, v in loaded.items()}
                else:
                    list_dict['baseline_5_list'] = {idx: "legacy" for idx in loaded}

        if os.path.exists(f'{files_dir}/baseline_5_accurate_list.json'):
            with open(f'{files_dir}/baseline_5_accurate_list.json', 'r') as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    list_dict['baseline_5_accurate_list'] = {int(k): v for k, v in loaded.items()}
                else:
                    list_dict['baseline_5_accurate_list'] = {int(x): "legacy" for x in loaded}
        if os.path.exists(f'{files_dir}/baseline_6_list.json'):
            with open(f'{files_dir}/baseline_6_list.json', 'r') as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    list_dict['baseline_6_list'] = {int(k): v for k, v in loaded.items()}
                else:
                    list_dict['baseline_6_list'] = {idx: "legacy" for idx in loaded}
        
        if os.path.exists(f'{files_dir}/baseline_6_accurate_list.json'):
            with open(f'{files_dir}/baseline_6_accurate_list.json', 'r') as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    list_dict['baseline_6_accurate_list'] = {int(k): v for k, v in loaded.items()}
                else:
                    list_dict['baseline_6_accurate_list'] = {int(x): "legacy" for x in loaded}
        
        if os.path.exists(f'{files_dir}/total_list.json'):
            with open(f'{files_dir}/total_list.json', 'r') as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    list_dict['total_list'] = {int(k): v for k, v in loaded.items()}
                else:
                    list_dict['total_list'] = {idx: "legacy" for idx in loaded}
    except Exception as e:
        print(f"Error loading lists: {e}")
        list_dict['accurate_list'] = {}
        list_dict['accurate_list_old'] = {}
        list_dict['baseline_1_list'] = {}
        list_dict['baseline_2_list'] = {}
        list_dict['baseline_3_list'] = {}
        list_dict['baseline_4_list'] = {}
        list_dict['baseline_5_list'] = {}
        list_dict['baseline_6_list'] = {}
        list_dict['baseline_1_accurate_list'] = {}
        list_dict['baseline_2_accurate_list'] = {}
        list_dict['baseline_3_accurate_list'] = {}
        list_dict['baseline_4_accurate_list'] = {}
        list_dict['baseline_5_accurate_list'] = {}
        list_dict['baseline_6_accurate_list'] = {}
        list_dict['total_list'] = {}
    return list_dict
     


def save_lists(files_dir, list_dict):
    # save all dicts to files using JSON (convert int keys to strings for JSON compatibility)
    with open(f'{files_dir}/accurate_list.json', 'w') as f:
        json.dump({str(k): v for k, v in list_dict.get('accurate_list', {}).items()}, f, indent=4)
    
    # with open(f'{files_dir}/baseline_1_list.json', 'w') as f:
    #     json.dump({str(k): v for k, v in list_dict.get('baseline_1_list', {}).items()}, f, indent=4)
    # with open(f'{files_dir}/baseline_1_accurate_list.json', 'w') as f:
    #     json.dump(list_dict.get('baseline_1_accurate_list', []), f, indent=4)
    
    # with open(f'{files_dir}/baseline_2_list.json', 'w') as f:
    #     json.dump({str(k): v for k, v in list_dict.get('baseline_2_list', {}).items()}, f, indent=4)
    # with open(f'{files_dir}/baseline_2_accurate_list.json', 'w') as f:
    #     json.dump(list_dict.get('baseline_2_accurate_list', []), f, indent=4)
    
    # with open(f'{files_dir}/baseline_3_list.json', 'w') as f:
    #     json.dump({str(k): v for k, v in list_dict.get('baseline_3_list', {}).items()}, f, indent=4)
    # with open(f'{files_dir}/baseline_3_accurate_list.json', 'w') as f:
    #     json.dump(list_dict.get('baseline_3_accurate_list', []), f, indent=4)
    
    with open(f'{files_dir}/baseline_4_list.json', 'w') as f:
        json.dump({str(k): v for k, v in list_dict.get('baseline_4_list', {}).items()}, f, indent=4)
    with open(f'{files_dir}/baseline_4_accurate_list.json', 'w') as f:
        json.dump({str(k): v for k, v in list_dict.get('baseline_4_accurate_list', {}).items()}, f, indent=4)

    with open(f'{files_dir}/baseline_5_list.json', 'w') as f:
        json.dump({str(k): v for k, v in list_dict.get('baseline_5_list', {}).items()}, f, indent=4)
    with open(f'{files_dir}/baseline_5_accurate_list.json', 'w') as f:
        json.dump({str(k): v for k, v in list_dict.get('baseline_5_accurate_list', {}).items()}, f, indent=4)

    with open(f'{files_dir}/baseline_6_list.json', 'w') as f:
        json.dump({str(k): v for k, v in list_dict.get('baseline_6_list', {}).items()}, f, indent=4)
    with open(f'{files_dir}/baseline_6_accurate_list.json', 'w') as f:
        json.dump({str(k): v for k, v in list_dict.get('baseline_6_accurate_list', {}).items()}, f, indent=4)
    
    # Only save total_list if it's explicitly provided and not empty
    if 'total_list' in list_dict and list_dict['total_list']:
        with open(f'{files_dir}/total_list.json', 'w') as f:
            json.dump({str(k): v for k, v in list_dict['total_list'].items()}, f, indent=4)

def save_answer(files_dir, index, answer):
    # open an existing dictionary which as index to answer mapping, and update with the new answer or add if not present
    answers = {}
    if os.path.exists(f'{files_dir}/answers.json'):
        with open(f'{files_dir}/answers.json', 'r') as f:
            answers = json.load(f)
    answers[str(index)] = answer
    with open(f'{files_dir}/answers.json', 'w') as f:
        json.dump(answers, f, indent=4)


def to_pil_and_bytes(img_field):
    if isinstance(img_field, dict) and "bytes" in img_field:
        img_bytes = img_field["bytes"]
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        return img, img_bytes
    if hasattr(img_field, "convert"):
        img = img_field.convert("RGB")
        return img, None
    img = Image.open(BytesIO(img_field)).convert("RGB")
    return img, img_field

def load_select_dataset():
    config = load_config()
    dset = load_dataset(config['dataset'])
    dset = dset[config['split']]
    # indices = [150, 200, 0, 2, 3, 9, 11, 15, 16, 20, 22, 35, 37, 38, 39] 
    indices = [0,2,6,9,11,15,16,22,23,25,35,37,38,39,46,48,51,55,64,68,79,87,106,112,113,114,124,129,131,138,139,150,151,163,164,164,191]
    new_ds = dset.select(indices)

    return new_ds

def load_select_dataset_indices(indices):
    config = load_config()
    dset = load_dataset(config['dataset'])
    dset = dset[config['split']]
    new_ds = dset.select(indices)

    return new_ds

def load_full_dataset():
    config = load_config()
    dset = load_dataset(config['dataset'])
    dset = dset[config['split']]
    return dset


def load_config(config_path=None):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str, optional): Path to config file. 
                                   If None, looks for config.yaml in ../configs/ relative to this file.
    
    Returns:
        dict: Configuration dictionary
    """
    if config_path is None:
        # Get the directory of the current script
        current_dir = Path(__file__).parent
        # Go up one level and into configs directory
        config_path = current_dir.parent / "configs" / "config.yaml"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Config file not found at: {config_path}")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML config: {e}")
        return {}





# ENTITY_RE = re.compile(r'id=([^;|\s]+);\s*type=([^;|\s]+)', re.IGNORECASE)
# PROPS_RE = re.compile(r'props\((.*?)\)', re.IGNORECASE)


# def extract_after_assistant(text: str) -> str:
#     """
#     Return everything after the last occurrence of the word 'assistant'
#     (case-insensitive). Accepts optional colon or whitespace after the word.
#     If 'assistant' is not found, returns the original text.
#     """
#     # Find all matches of 'assistant' as a whole word, optional ':' and trailing whitespace/newline
#     pattern = re.compile(r'(?is)\bassistant\b\s*:?\s*')
#     last_match = None
#     for m in pattern.finditer(text):
#         last_match = m
#     if last_match is None:
#         return text
#     return text[last_match.end():].strip()


# def _parse_entity(ent: str) -> Tuple[str, str, Dict[str, str]]:
#     """
#     Parse an entity segment like:
#       'SUBJECT id=R1; type=resistor; props(resistance=470_Ohm,label=R_series)'
#       'id=ground; type=ground'
#     Returns (id, type, props_dict)
#     """
#     # Normalize by removing leading SUBJECT/OBJECT tokens if present
#     ent = ent.strip()
#     ent = re.sub(r'^(SUBJECT|OBJECT)\s+', '', ent, flags=re.IGNORECASE)

#     m = ENTITY_RE.search(ent)
#     if not m:
#         raise ValueError(f"Cannot parse entity id and type from: {ent}")
#     eid, etype = m.group(1).strip(), m.group(2).strip()

#     props: Dict[str, str] = {}
#     mprops = PROPS_RE.search(ent)
#     if mprops:
#         inner = mprops.group(1).strip()
#         if inner:
#             # Split on commas not inside quotes (we do not expect quotes here)
#             pairs = [p.strip() for p in inner.split(',') if p.strip()]
#             for p in pairs:
#                 if '=' in p:
#                     k, v = p.split('=', 1)
#                     props[k.strip()] = v.strip()
#                 else:
#                     # key with no value, store as empty string
#                     props[p.strip()] = ""
#     return eid, etype, props

# def parse_triplet_lines(text: str) -> Tuple[Dict[str, Dict], List[Dict]]:
#     """
#     Parse pipe format lines into nodes and relationships.
#     Returns:
#       nodes_map: {id: {"id": id, "type": type, "props": {...}}}
#       rels: [{"from_id": idA, "to_id": idB, "predicate": pred}]
#     """
#     nodes_map: Dict[str, Dict] = {}
#     rels: List[Dict] = []

#     for raw_line in text.splitlines():
#         line = raw_line.strip()
#         if not line or line.lower().startswith("system") or line.lower().startswith("user"):
#             continue

#         # Expect exactly two pipes
#         parts = [p.strip() for p in line.split('|')]
#         if len(parts) != 3:
#             # Skip anything that is not a triplet line
#             continue

#         left_str, pred_str, right_str = parts

#         # Predicate token may be like 'PREDICATE connected_to' or just 'connected_to'
#         pred_str_norm = pred_str.replace('PREDICATE', '').strip()
#         predicate = pred_str_norm if pred_str_norm else 'connected_to'

#         sid, stype, sprops = _parse_entity(left_str)
#         oid, otype, oprops = _parse_entity(right_str)

#         # Merge or insert subject node
#         if sid not in nodes_map:
#             nodes_map[sid] = {"id": sid, "type": stype, "props": {}}
#         # Keep the first non empty type or prefer a more specific one
#         if not nodes_map[sid].get("type") and stype:
#             nodes_map[sid]["type"] = stype
#         # Merge props
#         nodes_map[sid]["props"].update(sprops)

#         # Merge or insert object node
#         if oid not in nodes_map:
#             nodes_map[oid] = {"id": oid, "type": otype, "props": {}}
#         if not nodes_map[oid].get("type") and otype:
#             nodes_map[oid]["type"] = otype
#         nodes_map[oid]["props"].update(oprops)

#         # Relationship
#         rels.append({"from_id": sid, "to_id": oid, "predicate": predicate})

#     return nodes_map, rels

# def build_cypher_create(nodes_map, rels):
#     """
#     Build a single Cypher CREATE statement like:
#     CREATE (v1:voltage_source {id:'V1', ...}),
#            (r1:resistor {id:'R1', ...}),
#            (v1)-[:CONNECTED_TO {predicate:'connected_to'}]->(r1), ...
#     Skips any empty string or None values in properties.
#     """
#     def clean_props(props):
#         # Remove empty or None values
#         return {k: v for k, v in props.items() if v not in ("", None)}

#     # 1. Node text
#     node_strs = []
#     for n in nodes_map.values():
#         label = n.get("type", "component")
#         props = {"id": n["id"], **(n.get("props") or {})}
#         props = clean_props(props)
#         prop_text = ", ".join(f"{k}:'{v}'" for k, v in props.items())
#         node_strs.append(f"({n['id']}:{label} {{{prop_text}}})")

#     # 2. Relationship text
#     rel_strs = []
#     for r in rels:
#         pred = r.get("predicate", "connected_to")
#         if pred not in ("", None):
#             rel_strs.append(
#                 f"({r['from_id']})-[:CONNECTED_TO {{predicate:'{pred}'}}]->({r['to_id']})"
#             )
#         else:
#             rel_strs.append(
#                 f"({r['from_id']})-[:CONNECTED_TO]->({r['to_id']})"
#             )

#     return "CREATE " + ",\n       ".join(node_strs + rel_strs)


# In terminal:
# ```
# user@host:~$ sudo docker run -it --rm \
# >   -p7474:7474 -p7687:7687 \
# >   -e NEO4J_AUTH=neo4j/testpass \
# >   neo4j:5.20
# ```


# In local terminal:
# ```
# ssh -L 7474:localhost:7474 -L 7687:localhost:7687 user@host
# ```


# def build_graph_from_output(output: str):
#     assistant_output = extract_after_assistant(output)
#     nodes_map, rels = parse_triplet_lines(assistant_output)
#     cypher = build_cypher_create(nodes_map, rels)
        
#     driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j","testpass"))
#     with driver.session() as session:
#         session.run(cypher)
#     driver.close()


# MATCH (n) RETURN n LIMIT 100;
# MATCH (n) DETACH DELETE n;


# def print_output(model_name=None, method="baseline"):
#     """
#     Evaluate model results.
    
#     Args:
#         filename (str): Name of the results file
#     """
#     # Load configuration
#     config = load_config()
#     if model_name is None:
#         model_name = config['model']
#     output_path = f"{config['results_dir']}/{config['dataset_name']}_{model_name}_{method}.jsonl"

#     dset = load_full_dataset()


#     results = []
#     with open(output_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             if line.strip():  # Skip empty lines
#                 results.append(json.loads(line))
                

        
#     for i, item in enumerate(results):
        
#         idx = item['index']
#         question = dset[idx]['problem']
#         image = dset[idx]['image']
#         gt = str(item['solution']).strip()
#         if 'model_output' in item:
#             mo = str(item['model_output']).strip()
#         elif 'ir' in item:
#             mo = str(item['ir']).strip()
#         else:
#             mo = None


#         print(f"\nDataset Index: {idx}:")
#         print(f"Question: {question}")
#         display(image)
#         print(f"Ground Truth: {gt}")
#         print(f"Model Output: {mo}")
        

# def convert_ir_to_cir(method="netlist"):
#     config = load_config()
#     input_path = f"{config['results_dir']}/{config['dataset_name']}_{config['model']}_{method}.jsonl"

#     with open(input_path, "r", encoding="utf-8") as f:
#         input = [json.loads(line) for line in f if line.strip()]

#     output = []
#     for item in input:
#         netlist = item.get("ir", "")
#         # get string inside ```spice ... ```
#         m = re.search(r"```spice(.*?)```", netlist, re.DOTALL | re.IGNORECASE)
#         if m:
#             netlist = m.group(1).strip()
#         item["cir"] = netlist
#         output.append(item)

#     with open(f"{config['results_dir']}/{config['dataset_name']}_{config['model']}_cir.jsonl", "w", encoding="utf-8") as f:
#         for item in output:
#             f.write(json.dumps(item, ensure_ascii=False) + "\n")
#     print(f"Converted IR to CIR and saved to {config['results_dir']}/{config['dataset_name']}_{config['model']}_cir.jsonl")


# def convert_schema_to_netlist():
#     config = load_config()
#     input_path = f"{config['results_dir']}/{config['dataset_name']}_{config['model']}_schema.jsonl"

#     with open(input_path, "r", encoding="utf-8") as f:
#         input = [json.loads(line) for line in f if line.strip()]

#     output = []
#     for idx,item in enumerate(input):
#         schema = item.get("ir", "")
#         netlist = convert_ir_json_to_ngspice_netlist(schema,f"item_{idx}")
#         item["ir"] = netlist
#         output.append(item)

#     with open(f"{config['results_dir']}/{config['dataset_name']}_{config['model']}_netlist.jsonl", "w", encoding="utf-8") as f:
#         for item in output:
#             f.write(json.dumps(item, ensure_ascii=False) + "\n")
#     print(f"Converted IR to Netlist and saved to {config['results_dir']}/{config['dataset_name']}_{config['model']}_netlist.jsonl")



def crop_from_bbox(img, bbox, pad=8):
    """
    Crop a bbox from the image and return a square, white-padded version
    so that CLIP sees the symbol centered.
    Works with grayscale or BGR images.
    """
    x, y, w, h = bbox
    x2, y2 = x + w, y + h

    # step 1: extract crop
    crop = img[y:y2, x:x2]

    # step 2: ensure 3-channel RGB (CLIP expects RGB input)
    if len(crop.shape) == 2:  # grayscale
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    else:  # convert BGR to RGB
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    # step 3: make square canvas
    h, w, _ = crop.shape
    size = max(h, w) + 2 * pad  # extra white border around
    square = np.ones((size, size, 3), dtype=np.uint8) * 255  # white background

    # step 4: center the crop
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    square[y_off:y_off+h, x_off:x_off+w] = crop

    return square

def save_crop(crop_np, path):
    if crop_np.ndim == 3 and crop_np.shape[2] == 3:
        pil = Image.fromarray(crop_np[..., ::-1])  # BGR to RGB
    else:
        pil = Image.fromarray(crop_np)
    pil.save(path)

def save_bboxes_as_crops(merged_boxes, img, idx):
    img = np.asarray(img)
    config = load_config()
    crop_dir = Path(config["crops_dir"]) / f"img{idx}"
    if crop_dir.exists():
        shutil.rmtree(crop_dir)
    crop_dir.mkdir(parents=True, exist_ok=True)

    for i, (x1, y1, x2, y2) in enumerate(merged_boxes):
        bbox = (x1, y1, x2 - x1, y2 - y1)
        crop_np = crop_from_bbox(img, bbox)
        save_crop(crop_np, str(crop_dir / f"{i}.png"))