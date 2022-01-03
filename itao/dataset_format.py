
CLASSIFICATION = """\
* File Structure
.
|--dataset_root:
    |--train
        |--class_1:
            |--1.jpg
            |--2.jpg
        |--class_2:
            |--01.jpg
            |--02.jpg
    |--val
        |--class_1:
            |--3.jpg
            |--4.jpg
        |--class_2:
            |--03.jpg
            |--04.jpg
    |--test
        |--class_1:
            |--5.jpg
            |--6.jpg
        |--class_2:
            |--05.jpg
            |--06.jpg
"""

OBJECT = """\
* File Structure
.
|--dataset root
  |-- images
      |-- 000000.jpg
      |-- 000001.jpg
            .
            .
      |-- xxxxxx.jpg
  |-- labels
      |-- 000000.txt
      |-- 000001.txt
            .
            .
      |-- xxxxxx.txt
  |-- kitti_seq_to_map.json ( Optional )

* Label Format

** KITTI ( only requires the class name and bbox coordinates fields to be populated )
```
car 0.00 0 0.00 587.01 173.33 614.12 200.12 0.00 0.00 0.00 0.00 0.00 0.00 0.00
cyclist 0.00 0 0.00 665.45 160.00 717.93 217.99 0.00 0.00 0.00 0.00 0.00 0.00 0.00
pedestrian 0.00 0 0.00 423.17 173.67 433.17 224.03 0.00 0.00 0.00 0.00 0.00 0.00 0.00
```

** COCO
```
annotation{
"id": int,
"image_id": int,
"category_id": int,
"bbox": [x,y,width,height],
"area": float,
"iscrowd": 0 or 1,
}

image{
"id": int,
"width": int,
"height": int,
"file_name": str,
"license": int,
"flickr_url": str,
"coco_url": str,
"date_captured": datetime,
}

categories[{
"id": int,
"name": str,
"supercategory": str,
}]
```

"""

INSTANCE = """\
* File Structure
.
|--dataset root
  |-- images
      |-- 000000.jpg
      |-- 000001.jpg
            .
            .
      |-- xxxxxx.jpg
  |-- labels
      |-- 000000.txt
      |-- 000001.txt
            .
            .
      |-- xxxxxx.txt
  |-- kitti_seq_to_map.json ( Optional )

* Label Format

annotation{
"id": int,
"image_id": int,
"category_id": int,
"segmentation": RLE or [polygon],
"area": float,
"bbox": [x,y,width,height],
"iscrowd": 0 or 1,
}

image{
"id": int,
"width": int,
"height": int,
"file_name": str,
"license": int,
"flickr_url": str,
"coco_url": str,
"date_captured": datetime,
}

categories[{
"id": int,
"name": str,
"supercategory": str,
}]

"""

SEMANTIC = """\

* File Structure
.
/Dataset_01
    /images
      /train
        0000.png
        0001.png
        ...
        ...
        N.png
      /val
        0000.png
        0001.png
        ...
        ...
        N.png
      /test
        0000.png
        0001.png
        ...
        ...
        N.png
    /masks
      /train
        0000.png
        0001.png
        ...
        ...
        N.png
      /val
        0000.png
        0001.png
        ...
        ...
        N.png

* Image and Mask Format

** images_source1.txt

/home/user/workspace/exports/images_final/00001.jpg
/home/user/workspace/exports/images_final/00002.jpg

** labels_source1.txt

/home/user/workspace/exports/masks_final/00001.png
/home/user/workspace/exports/masks_final/00002.png

"""

DSET_FMT_LS = {
        "CLASSIFICATION":CLASSIFICATION, 
        "OBJECT":OBJECT, 
        "INSTANCE":INSTANCE, 
        "SEMANTIC":SEMANTIC 
}

def get_dset_format(task='classification'):
    task = task.upper()
    return DSET_FMT_LS[task]