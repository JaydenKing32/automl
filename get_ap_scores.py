import csv
import json
import re
from argparse import ArgumentParser
from collections import defaultdict

parser = ArgumentParser(description="Extracts AP scores from logs")
parser.add_argument("log", help="log file")
log_path = parser.parse_args().log

with open("coco_cats.json") as coco_file:
    coco_cats = [cat["name"] for cat in json.load(coco_file)["categories"]]

with open("imagenet_cats.json") as imagenet_file:
    imagenet_cats = [cat["name"] for cat in json.load(imagenet_file)["categories"]]

shared_cats = defaultdict(lambda: False)
for cat in coco_cats:
    shared_cats[cat] = cat in imagenet_cats

with open(log_path) as log_file:
    for line in log_file:
        if line.startswith("INFO:tensorflow:Saving dict for global step"):
            results = re.findall(r" ([\w\-_/ ]+) = (\d+\.[\de\-)]+),?", line)
            break

ap_scores = list(filter(lambda a: a[0].startswith("AP_/"), results))
metrics = list(filter(lambda m: m not in ap_scores, results))

with open("ap_scores.csv", mode='w', newline='') as outfile:
    writer = csv.writer(outfile)

    writer.writerow(["Metric", "Value", ''])
    for metric in metrics:
        metric_name = metric[0]
        metric_value = float(metric[1])
        writer.writerow([metric_name, "{:.8f}".format(metric_value), ''])
    writer.writerow(['', '', ''])

    writer.writerow(["Class", "AP", "Shared"])
    for ap in ap_scores:
        class_name = ap[0].split('/')[1]
        ap_score = float(ap[1])
        is_shared = shared_cats[ap[0]]
        writer.writerow([class_name, "{:.8f}".format(ap_score), shared_cats[ap[0]]])
        # print("{:<22}: {:.8f}".format(class_name, ap_score))
