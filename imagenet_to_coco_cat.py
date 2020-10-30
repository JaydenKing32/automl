import json
from copy import deepcopy

with open("coco_cats.json") as coco_file:
    coco_cats = {cat["name"]: cat["id"] for cat in json.load(coco_file)["categories"]}
    # coco_cats = {cat["name"]: {"supercategory": cat["supercategory"], "id": cat["id"]}
    #              for cat in json.load(coco_file)["categories"]}

with open("imagenet_cats.json") as imagenet_file:
    imagenet_cats = {cat["name"]: cat["id"] for cat in json.load(imagenet_file)["categories"]}

new_imagenet_cats = deepcopy(imagenet_cats)

for cat, cid in coco_cats.items():
    if cat in imagenet_cats:
        swap_name = list(new_imagenet_cats.keys())[list(new_imagenet_cats.values()).index(cid)]
        temp = new_imagenet_cats[cat]
        new_imagenet_cats[cat] = cid
        new_imagenet_cats[swap_name] = temp

output_json_dicts = []

for cat, cid in new_imagenet_cats.items():
    output_json_dicts.append({"supercategory": "none", "id": cid, "name": cat})

output_json_dicts.sort(key=lambda cat: cat["id"])

print(json.dumps({"categories": output_json_dicts}))

ordered_dict = {}
reverse_dict = {v: k for k, v in new_imagenet_cats.items()}

for v in sorted(reverse_dict):
    ordered_dict[v] = reverse_dict[v]

print(ordered_dict)
