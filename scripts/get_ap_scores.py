import csv
import json
import re
from argparse import ArgumentParser
from collections import defaultdict

parser = ArgumentParser(description="Extracts AP scores from logs")
parser.add_argument("log", help="log file")
parser.add_argument("annotation", help="annotation file used to produce log")
args = parser.parse_args()

log_path = args.log
ann_path = args.annotation

with open("coco_categories.json") as coco_file:
    coco_cats = [cat["name"] for cat in json.load(coco_file)["categories"]]

with open(ann_path) as ann_file:
    log_cats = [cat["name"] for cat in json.load(ann_file)["categories"]]

shared_cats = defaultdict(lambda: False)
for cat in coco_cats:
    shared_cats[cat] = cat in log_cats

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

    heading = ["Class", "AP", "Shared"]
    writer.writerow(heading)
    print("{:24} {:10} {}".format(*heading))

    for ap in ap_scores:
        class_name = ap[0].split('/')[1]
        ap_score = float(ap[1])
        is_shared = shared_cats[class_name]
        writer.writerow([class_name, "{:.8f}".format(ap_score), is_shared])
        print("{:<22}: {:.8f}, {}".format(class_name, ap_score, is_shared))
