import json

images = [{
    "file_name": f"{i:05d}.jpg",
    "width": 1360,
    "height": 800,
    "id": i
} for i in range(900)]

with open("efficientdet/data/FullIJCNN2013/gt.txt", mode='r', encoding="UTF-8") as file:
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
            "category_id": 13,
            "id": i,
            "ignore": 0
        })
        i += 1
print(json.dumps(images))
print(json.dumps(annotations))
