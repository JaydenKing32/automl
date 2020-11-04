import json
from argparse import ArgumentParser

parser = ArgumentParser(description="Converts GTSDB ground-truth file to COCO object annotations file")
parser.add_argument("gt", help="ground-truth file")
gt_path = parser.parse_args().gt

sign_id = 13

categories = {
    "supercategory": "outdoor",
    "id": sign_id,
    "name": "stop sign"
}

images = [{
    "file_name": f"{i:05d}.jpg",
    "width": 1360,
    "height": 800,
    "id": i
} for i in range(900)]

with open(gt_path, mode='r', encoding="UTF-8") as file:
    annotations = []
    i = 0

    for line in file:
        image = line.split(';')
        image_id = int(image[0].split('.')[0])
        x1 = int(image[1])
        y1 = int(image[2])
        x2 = int(image[3])
        y2 = int(image[4])

        width = abs(x2 - x1)
        height = abs(y2 - y1)

        annotations.append({
            "segmentation": [
                x1, y1,
                x1, y2,
                x2, y2,
                x2, y1
            ],
            "area": width * height,
            "iscrowd": 0,
            "image_id": image_id,
            "bbox": [x1, y1, width, height],
            "category_id": sign_id,
            "id": i,
            "ignore": 0
        })
        i += 1

annotation_dict = {
    "type": "instances",
    "categories": [categories],
    "images": images,
    "annotations": annotations
}

with open("FullIJCNN2013.json", mode='w', encoding="UTF-8", newline='\n') as annotation_file:
    json.dump(annotation_dict, annotation_file, indent=2)
