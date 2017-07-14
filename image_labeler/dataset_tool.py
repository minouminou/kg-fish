"""
Make dataset in Microsoft COCO format
"""
import os, shutil
import json
import cv2


def im_array_width_height(input):
    width = input.shape[1]
    height = input.shape[0]
    return width, height


def imread(img_path):
    if not os.path.exists(img_path):
        "ERROR: image path does not exist."
        error
    img = cv2.imread(img_path)
    if img is None:
        print("ERROR: cannot load image " + img_path)
        error
    return img


def getbbox(bbox):
    x = min(bbox[0::2])
    y = min(bbox[1::2])
    x2 = max(bbox[0::2])
    y2 = max(bbox[1::2])
    w = x2 - x
    h = y2 - y
    return([x, y, w, h])


def getarea(bbox):
    return(1000.0)


def dump_json(out_file, images, annotations):
    a = dict()
    a["info"] = {"description": "The Nature Conservancy Fisheries Monitoring",
                 "url": "https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/data",
                 "version": "1.0", "year": 2016, "contributor": "Alex", "date_created": "2016-11-14 00:00:00.000000"}
    a["images"] = images
    a["licenses"] = [{"url": "https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/data", "id": 1,
                      "name": "The Nature Conservancy Fisheries Monitoring"}]
    a["annotations"] = annotations
    a["categories"] = [{"supercategory": "fish", "id": 1, "name": "ALB"},
                       {"supercategory": "fish", "id": 2, "name": "BET"},
                       {"supercategory": "fish", "id": 3, "name": "DOL"},
                       {"supercategory": "fish", "id": 4, "name": "LAG"},
                       {"supercategory": "fish", "id": 5, "name": "OTHER"},
                       {"supercategory": "fish", "id": 6, "name": "SHARK"},
                       {"supercategory": "fish", "id": 7, "name": "YFT"}]

    with open(out_file, 'w') as outfile:
        json.dump(a, outfile)


def coco_desc(img_dir, category_id, nmin, nmax, out_dir, images, annotations):
    img_filenames = [f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")]
    img_filenames.sort()
    img_filenames = img_filenames[nmin:nmax]
    resize = True
    for img_filename_index, img_filename in enumerate(img_filenames):
        print(img_filename_index, img_filename)

        img_path = os.path.join(img_dir, img_filename)
        ann_path = os.path.splitext(os.path.join(img_dir, img_filename))[0] + ".segm.txt"
        lbl_path = os.path.splitext(os.path.join(img_dir, img_filename))[0] + ".lbl.txt"

        img = imread(img_path)
        if resize:
            img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            cv2.imwrite(os.path.join(out_dir, img_filename), img)
        else:
            shutil.copy(img_path, out_dir)

        img_width, img_height = im_array_width_height(img)
        print(img_width, img_height)
        image_id = len(images) + 1
        images.append({"license": 1, "file_name": img_filename,
                       "coco_url": "https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/data",
                       "height": img_height, "width": img_width,
                       "date_captured": "2016-11-14 00:00:00",
                       "flickr_url": "https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/data",
                       "id": image_id})

        labels = []
        with open(lbl_path, "r") as ins:
            for line in ins:
                line = line.rstrip("\n")
                labels.append(line)
        classes = {"ALB":1, "BET":2, "DOL":3, "LAG":4, "OTHER":5, "SHARK":6, "YFT":7}

        with open(ann_path, "r") as ins:
            for rowno, line in enumerate(ins):
                merged = [float(x) for x in line.split(',') if x]

                if resize:
                    merged = [ x/2.0 for x in merged]
                    merged = [x / 2.0 for x in merged]

                segm_id = len(annotations) + 1
                categ_id = classes[labels[rowno]]
                annotations.append({"image_id": image_id,
                                    "category_id": categ_id,
                                    "segmentation": [merged],
                                    "area": getarea(merged),
                                    "iscrowd": 0,
                                    "bbox": getbbox(merged),
                                    "id": segm_id})

#new (k=0.65)
#class total test  train
#ALB   300   0:196 196:300
#BET   200   0:131 131:201
#DOL   117   0:77  77:118
#LAG   67    0:45  45:68
#NoF   465   -     -
#OTHER 299   0:196 196:300
#SHARK 176   0:116 116:177
#YFT   734   0:479 479:735

in_dir = '~/image_labeler/fish_label/'
out_dir = '~/image_labeler/fish_dataset/fish_dataset_quarter/'


ds_dir = out_dir + 'train2014/'
ds_file = out_dir +'annotations/instances_train2014.json'
images = []
annotations = []
coco_desc(in_dir + 'ALB/', 1, 0, 196, ds_dir, images, annotations)
coco_desc(in_dir + 'BET/', 2, 0, 131, ds_dir, images, annotations)
coco_desc(in_dir + 'DOL/', 3, 0, 77, ds_dir, images, annotations)
coco_desc(in_dir + 'LAG/', 4, 0, 45, ds_dir, images, annotations)
coco_desc(in_dir + 'OTHER/', 5, 0, 196, ds_dir, images, annotations)
coco_desc(in_dir + 'SHARK/', 6, 0, 116, ds_dir, images, annotations)
coco_desc(in_dir + 'YFT/', 7, 0, 479, ds_dir, images, annotations)
dump_json(ds_file, images, annotations)

ds_dir = out_dir + 'val2014/'
ds_file = out_dir +'annotations/instances_val2014.json'
images = []
annotations = []
coco_desc(in_dir + 'ALB/', 1, 196, 300, ds_dir, images, annotations)
coco_desc(in_dir + 'BET/', 2, 131, 201, ds_dir, images, annotations)
coco_desc(in_dir + 'DOL/', 3, 77, 118, ds_dir, images, annotations)
coco_desc(in_dir + 'LAG/', 4, 45, 68, ds_dir, images, annotations)
coco_desc(in_dir + 'OTHER/', 5, 196, 300, ds_dir, images, annotations)
coco_desc(in_dir + 'SHARK/', 6, 116, 177, ds_dir, images, annotations)
coco_desc(in_dir + 'YFT/', 7, 479, 735, ds_dir, images, annotations)
dump_json(ds_file, images, annotations)
