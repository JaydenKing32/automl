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
    source_cats = {cat["id"]: cat["name"] for cat in json.load(source_file)["categories"]}

with open(dest_path, mode='r', encoding="UTF-8") as dest_file:
    dest_cats = {cat["id"]: cat["name"] for cat in json.load(dest_file)["categories"]}

new_dest_cats = deepcopy(dest_cats)

for cid, cat in source_cats.items():
    if cat in dest_cats.values():
        if cid in new_dest_cats:
            orig_cid = list(new_dest_cats.keys())[list(new_dest_cats.values()).index(cat)]
            temp = new_dest_cats[cid]
            new_dest_cats[cid] = cat
            new_dest_cats[orig_cid] = temp
        else:
            orig_cid = list(new_dest_cats.keys())[list(new_dest_cats.values()).index(cat)]
            new_dest_cats.pop(orig_cid, None)
            new_dest_cats[cid] = cat

# # Fill empty class IDs
# for i in range(max(new_dest_cats)):
#     if i not in new_dest_cats:
#         new_dest_cats[i] = source_cats[i]

output_cat_dicts = []

for cid, cat in new_dest_cats.items():
    output_cat_dicts.append({"supercategory": "none", "id": cid, "name": cat})

output_cat_dicts.sort(key=lambda cat: cat["id"])

with open(dest_path, mode='r', encoding="UTF-8") as dest_file:
    copy(dest_path, "{}.bak".format(dest_path))  # Make backup of dest_file
    dest_json = json.load(dest_file)
    dest_json["categories"] = output_cat_dicts

with open(dest_path, mode='w', encoding="UTF-8", newline='\n') as dest_file:
    json.dump(dest_json, dest_file)

# Print new label map
# print(new_dest_cats)
