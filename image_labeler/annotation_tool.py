"""
Annotation tool
"""
import os
import sys
import itertools
import cv2


img_dir = "~/label/alb/"
ann_dir = "~/label/alb/"
show_annotations = False


def to_integers(list1d):
    return [int(float(x)) for x in list1d]


def imread(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("ERROR: cannot load image " + img_path)
        error
    return img


def draw_mask(img, mask, color=(0, 255, 0), thickness=1):
    for point in mask:
        pt1 = tuple(to_integers(point[0:2]))
        if mask.index(point) == 0:
            prev_pt = pt1
        cv2.circle(img, pt1, 1, color, thickness)
        cv2.line(img, pt1, prev_pt, color, thickness)
        prev_pt = pt1


def draw_masks(img, masks, color=(0, 255, 0), thickness=1):
    for mask in masks:
        draw_mask(img, mask, color, thickness)


def event_cv2_draw_masks(event, x, y, flags, param):
    global global_image
    global global_bboxes
    global global_masks
    img_copy = global_image.copy()
    draw_mask(img_copy, global_bboxes, (0, 255, 0))
    draw_masks(img_copy, global_masks, (255, 0, 0))
    if event == cv2.EVENT_LBUTTONDOWN:
        pt = (x,y)
        cv2.circle(img_copy, pt, 1, (255, 255, 0), 1)
        global_bboxes.append(pt)
    elif event == cv2.EVENT_RBUTTONDOWN:
        global_masks.append(global_bboxes)
        global_bboxes = []
    else:
        pass
    cv2.imshow("AnnotationWindow", img_copy)


def write_masks(output_file, table):
    f = open(output_file, 'w')
    for row in table:
        merged = list(itertools.chain(*row))
        merged = list(map(float, merged))
        merged = [str(x) for x in merged]
        line = ",".join(merged)
        f.write("%s\n" % line)
    f.close()

print("========================================================================")
print("Image annotation tool")
print("img_dir: " + img_dir)
print("ann_dir: " + img_dir)
print("show: " + str(show_annotations))
print("Keyboard: n - next image, s - skip image, u - undo/remove last point, q - quit")
print("Mouse: Left button - new point, Right button - finish mask")
print("========================================================================")
img_filenames = [f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")]
img_filenames.sort()
for img_filename_index, img_filename in enumerate(img_filenames):
    img_path = os.path.join(img_dir, img_filename)
    ann_path = os.path.splitext(os.path.join(ann_dir, img_filename))[0] + ".segm.txt"

    global_bboxes = []
    global_masks = []
    global_image = imread(img_path)

    cv2.namedWindow("AnnotationWindow")
    if show_annotations:
        print(ann_path)
        if not os.path.exists(ann_path):
            print ("Skipping image {0} since annotation not exists".format(img_filename))
            continue
        else:
            print ("Showing image {0} of {1}: {2}".format(img_filename_index + 1, len(img_filenames), img_path))
        with open(ann_path, "r") as ins:
            for line in ins:
                merged = [int(float(x)) for x in line.split(',') if x]
                global_masks.append(merged)
        global_masks = [list(zip(x[::2], x[1::2])) for x in global_masks]
        draw_masks(global_image, global_masks, (255, 0, 0))
        cv2.imshow("AnnotationWindow", global_image)
        while True:
            #key = chr(cv2.waitKey())
            key = cv2.waitKey()
            if key > 255: # bug see http://code.opencv.org/issues/2910
                key = key % 256
            key = chr(key)
            if key == "n":
                break
            elif key == "s":
                break
            elif key == "q":
                sys.exit()
    else:
        if os.path.exists(ann_path):
            print ("Skipping image {0} since annotation already exists".format(img_filename))
            continue
        else:
            print ("Processing image {0} of {1}: {2}".format(img_filename_index+1, len(img_filenames), img_path))
        cv2.setMouseCallback("AnnotationWindow", event_cv2_draw_masks)
        cv2.imshow("AnnotationWindow", global_image)
        while True:
            #key = chr(cv2.waitKey())
            key = cv2.waitKey()
            if key > 255:  # bug see http://code.opencv.org/issues/2910
                key = key % 256
            key = chr(key)
            # undo/remove last point
            if key == "u":
                if len(global_bboxes) >= 1:
                    global_bboxes = global_bboxes[:-1]
                    img_copy = global_image.copy()
                    draw_mask(img_copy, global_bboxes)
                    draw_masks(img_copy, global_masks, (255, 0, 0))
                    cv2.imshow("AnnotationWindow", img_copy)
            # skip image
            elif key == "s":
                if os.path.exists(ann_path):
                    print("Skipping image and deleting existing annotation file: " + ann_path)
                    os.remove(ann_path)
                break
            # next image
            elif key == "n":
                write_masks(ann_path, global_masks)
                break
            # quit
            elif key == "q":
                sys.exit()

cv2.destroyAllWindows()
