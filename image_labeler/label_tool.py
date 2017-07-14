"""
Label tool
"""
import os
#from Tkinter import *
from tkinter import *
from PIL import ImageTk, Image
import cv2


img_dir = "~/label_auto/alb/"
ann_dir = "~/label_auto/alb/"
lbl_dir = "~/label_auto/alb/"
classes = ("ALB", "BET", "DOL", "LAG", "OTHER", "SHARK", "YFT", "NoF", "SKIP")


def button_pressed_callback(s):
    global global_last_button_pressed
    global_last_button_pressed = s


def imconvert_cv2_pil(img):
    cv2_im = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2_im)


def imread(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("ERROR: cannot load image " + img_path)
        error
    return img


def to_integers(list1d):
    return [int(float(x)) for x in list1d]


def draw_mask(img, mask, color=(0, 255, 0), thickness=1):
    for point in mask:
        pt1 = tuple(to_integers(point[0:2]))
        if mask.index(point) == 0:
            prev_pt = pt1
        cv2.circle(img, pt1, 1, color, thickness)
        cv2.line(img, pt1, prev_pt, color, thickness)
        prev_pt = pt1


def write_labels(output_file, table):
    f = open(output_file, 'w')
    for row in table:
        line = row
        f.write("%s\n" % line)
    f.close()


print("========================================================================")
print("Image label tool")
print("img_dir: " + img_dir)
print("ann_dir: " + img_dir)
print("lbl_dir: " + lbl_dir)
print("========================================================================")
object_names = list(classes)
tk = Tk()
box_width = 10
box_height = 2
w = Canvas(tk, width=len(object_names) * box_width, height=len(object_names) * box_height, bd = box_width, bg = 'white')
w.grid(row = len(object_names), column = 0, columnspan = 2)
for object_index, object_name in enumerate(object_names):
    b = Button(width=box_width, height=box_height, text=object_name, command=lambda s = object_name: button_pressed_callback(s))
    b.grid(row = object_index, column = 0)

img_filenames = [f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")]
img_filenames.sort()
for img_index, img_filename in enumerate(img_filenames):
    img_path = os.path.join(img_dir, img_filename)
    lbl_path = os.path.splitext(os.path.join(lbl_dir, img_filename))[0] + ".lbl.txt"
    ann_path = os.path.splitext(os.path.join(ann_dir, img_filename))[0] + ".segm.txt"
    if os.path.exists(lbl_path):
        print ("Skipping image {:3} ({}) since label file already exists: {}".format(img_index, img_filename, lbl_path))
        continue
    print ("Processing image {0} of {1}: {2}".format(img_index + 1, len(img_filenames), img_path))
    img = imread(img_path)
    masks = []
    with open(ann_path, "r") as ins:
        for line in ins:
            merged = [int(float(x)) for x in line.split(',') if x]
            masks.append(merged)
    masks = [list(zip(x[::2], x[1::2])) for x in masks]
    labels = []
    skip = 0
    for mask_index, mask in enumerate(masks):
        imgCopy = img.copy()
        draw_mask(imgCopy,mask)
        imgTk = imgCopy
        imgTk = ImageTk.PhotoImage(imconvert_cv2_pil(imgTk))
        label = Label(tk, image=imgTk)
        label.grid(row=0, column=1, rowspan=1000)
        tk.update_idletasks()
        tk.update()
        global_last_button_pressed = None
        while not global_last_button_pressed:
            tk.update_idletasks()
            tk.update()
        print("Label: " + global_last_button_pressed)
        if global_last_button_pressed == "SKIP":
            skip = 1
            break
        labels.append(global_last_button_pressed)
    
    if skip == 0:
        write_labels(lbl_path, labels)
    skip = 0

tk.destroy()
print("Done")