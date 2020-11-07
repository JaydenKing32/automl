- Instructions assume working directory is efficientdet
- Install requirements file with `pip install -r requirements.txt`
- Download [Efficient-D0 model](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d0.tar.gz)
- Download dataset and annotations:
    - COCO 2017: [dataset](http://images.cocodataset.org/zips/val2017.zip) (val2017),
    [annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
    - [GTSDB](https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/FullIJCNN2013.zip) (FullIJCNN2013)
    - [VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) (VOCdevkit),
    [backup](http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar)
    - ImageNet 2014: [request download](http://image-net.org/signup.php) (ILSVRC2013),
    [coco annotations](https://s3.amazonaws.com/images.cocodataset.org/external/external_ILSVRC2014.zip)
- Dataset preparation:
    - Convert GTSDB files from PPM to JPG with `magick mogrify -format jpg -quality 75 *.ppm`
    - Run gtsdb_to_coco.py on GTSDB ground-truth file to create COCO annotation file,
    e.g. `python gtsdb_to_coco.py gt.txt` will use gt.txt to create FullIJCNN2013.json
    - Delete ILSVRC2013_val_00004542.JPEG from ILSVRC2013 and remove corresponding entry from ILSVRC2014_val.json
    (JSON object starting with `"file_name": "ILSVRC2013_val_00004542.JPEG"`)
    - Modify ImageNet and VOC annotation files to use COCO class IDs with convert_classes.py,
    e.g. `python convert_classes.py ILSVRC2014_val.json` and `python convert_classes.py json_pascal.json`
- Assuming models are stored in models/, datasets in data/, and annotations in data/annotations/, run the following
commands to execute EfficientDet:
    - COCO 2017 validation:
        - `python dataset/create_coco_tfrecord.py --image_dir=data/val2017 --caption_annotations_file=data/annotations/captions_val2017.json --output_file_prefix=tfrecord/val`
        - `python main.py --mode=eval --model_name=efficientdet-d0 --model_dir=models/efficientdet-d0/ --validation_file_pattern=tfrecord/val* --val_json_file=data/annotations/instances_val2017.json`
    - COCO 2017 test (upload detections_test-dev2017_test_results.json to
    [COCO evaluation server](https://competitions.codalab.org/competitions/20794#participate) to obtain results):
        - `python dataset/create_coco_tfrecord.py --image_dir=data/test2017 --image_info_file=data/annotations/image_info_test-dev2017.json --output_file_prefix=tfrecord/testdev`
        - `python main.py --mode=eval --model_name=efficientdet-d0 --model_dir=models/efficientdet-d0/ --validation_file_pattern=tfrecord/testdev* --eval_samples=20288 --testdev_dir='testdev_output'`
    - GTSDB:
        - `python dataset/create_coco_tfrecord.py --image_dir=data/FullIJCNN2013 --object_annotations_file=data/annotations/FullIJCNN2013.json --output_file_prefix=tfrecord/FullIJCNN2013`
        - `python main.py --mode=eval --model_name=efficientdet-d0 --model_dir=models/efficientdet-d0/ --validation_file_pattern=tfrecord/FullIJCNN2013* --eval_samples=900 --val_json_file=data/annotations/FullIJCNN2013.json`
    - ImageNet 2014:
        - `python dataset/create_coco_tfrecord.py --image_dir=data/ILSVRC2013 --image_info_file=data/annotations/ILSVRC2014_val.json --output_file_prefix=tfrecord/ILSVRC2014`
        - `python main.py --mode=eval --model_name=efficientdet-d0 --model_dir=models/efficientdet-d0/ --validation_file_pattern=tfrecord/ILSVRC2014* --eval_samples=20120 --val_json_file=data/annotations/ILSVRC2014_val.json`
    - VOC 2012:
        - `python ./dataset/create_pascal_tfrecord.py --data_dir=data/VOCdevkit --year=VOC2012 --output_path=tfrecord/pascal --set=val`
        - `python main.py --mode=eval --model_name=efficientdet-d0 --model_dir=models/efficientdet-d0 --validation_file_pattern=tfrecord/pascal*.tfrecord --eval_samples=5823 --val_json_file=tfrecord/json_pascal.json`
- Obtain per-class AP scores by running get_ap_scores.py on an evaluation log and the annotation file used to
produce said log, e.g. `python get_ap_scores.py ILSVRC2014.log ILSVRC2014_val.json`
