import json
from argparse import ArgumentParser
from copy import deepcopy
from shutil import copy

parser = ArgumentParser(description="Converts class IDs in DEST annotation file to match class IDs in SOURCE")
parser.add_argument("source", help="source annotation file")
parser.add_argument("dest", help="destination annotation file, will be modified")
args = parser.parse_args()

source_path = args.source
dest_path = args.dest

with open(source_path, mode='r', encoding="UTF-8") as source_file:
    source_cats = {cat["name"]: cat["id"] for cat in json.load(source_file)["categories"]}

with open(dest_path, mode='r', encoding="UTF-8") as dest_file:
    dest_cats = {cat["name"]: cat["id"] for cat in json.load(dest_file)["categories"]}

new_dest_cats = deepcopy(dest_cats)

for cat, cid in source_cats.items():
    if cat in dest_cats:
        swap_name = list(new_dest_cats.keys())[list(new_dest_cats.values()).index(cid)]
        temp = new_dest_cats[cat]
        new_dest_cats[cat] = cid
        new_dest_cats[swap_name] = temp

output_cat_dicts = []

for cat, cid in new_dest_cats.items():
    output_cat_dicts.append({"supercategory": "none", "id": cid, "name": cat})

output_cat_dicts.sort(key=lambda cat: cat["id"])

with open(dest_path, mode='r', encoding="UTF-8") as dest_file:
    copy(dest_path, "{}.bak".format(dest_path))
    dest_json = json.load(dest_file)
    dest_json["categories"] = output_cat_dicts

with open(dest_path, mode='w', encoding="UTF-8", newline='\n') as dest_file:
    json.dump(dest_json, dest_file)

# # Print new label map
# ordered_dict = {}
# reverse_dict = {v: k for k, v in new_dest_cats.items()}
#
# for v in sorted(reverse_dict):
#     ordered_dict[v] = reverse_dict[v]
#
# print(ordered_dict)
