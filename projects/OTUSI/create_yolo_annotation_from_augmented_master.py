# Copyright 2023 (C) antillia.com. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#
# create_augmented_master_512x512.py
# 2023/04/12 Antillia.com Toshiyuki Arai
# 2023/04/13 Modified to pass a parameter augment_all to create_augmented_master_512x512 function.
# 
# 1 This splits the original Dataset_BUSI_with_GT dataset 
# to three subsets train, test and valid. 
# 
# 2 Resize each image to 512x512
#
# 3 Rotate each image in train dataset by angle in ANGLES = [0, 90, 180, 270]
#
# 4 Save each rotated image as a jpg file,
#   In this way, the orignal image dataset has been augumented.
#    
import sys
import os
import glob
import random
import shutil

import traceback
import cv2
from PIL import Image, ImageOps, ImageEnhance


def read_classes_file(classes_file):
  filename_classid = {}
  with open(classes_file, "r") as f:
    lines = f.readlines()
    for line in lines:
       line = line.strip()
       array             = line.split(" ")
       print("--- array {}".format(array))
       filename, classid = line.split("  ")
       filename_classid[filename] = classid
       print(" --- filename: {}  classid: {}".format(filename, classid))
  return filename_classid


def create_yolo_annotation(classes_file, images_dir, masks_dir, 
                                    targets,   # train, valid, test
                                    output_dir, 
                                    debug=False):
    classes_definition  = read_classes_file(classes_file)
    
    #targets = ["train", "valid"] or ["test"]
    SP = " "
    NL = "\n"
    for target in targets: 
      pattern = images_dir + "/" + target + "/*.JPG"
      print("--- pattern {}".format(pattern))
      image_files = glob.glob(pattern)
      print("len {}".format(len(image_files)))
      output_subdir = os.path.join(output_dir , target)
      if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)

      for image_file in image_files:
        basename  = os.path.basename(image_file)
        name      = basename.split(".")[0]
        mask_filename   = basename.split(".")[0] + ".PNG"
        masks_subdir    = os.path.join(masks_dir, target)
        mask_file = os.path.join(masks_subdir, mask_filename)
        print("--- mask_file {}".format(mask_file))
        mask_img  = cv2.imread(mask_file)
        mask_img  = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        H, W = mask_img.shape[:2]
       
        contours, hierarchy = cv2.findContours(mask_img, 
           cv2.RETR_EXTERNAL, 
           cv2.CHAIN_APPROX_SIMPLE)
       
        contours = max(contours, key=lambda x: cv2.contourArea(x))
        x, y, w, h = cv2.boundingRect(contours)
        print("---x {} y {} w {} h {}".format(x, y, w, h))
              
        cx = x + w/2
        cy = y + h/2
        #Convert to relative coordinates for YOLO annotations
        rcx = round(cx / W, 5)
        rcy = round(cy / H, 5)
        rw  = round( w / W, 5)
        rh  = round( h / H, 5)
 
        if debug:
          _non_mask_img = cv2.imread(image_file)

          output_dir_annotated = os.path.join(output_dir, "annotated")
          if not os.path.exists(output_dir_annotated):
            os.makedirs(output_dir_annotated)
          #non_mask_img = cv2.imread(org_image, cv2.COLOR_BGR2RGB)
          _non_mask_img = cv2.rectangle(_non_mask_img , (x, y), (x+w, y+h), (255, 255, 0), 3)

          ouput_image_file_annotated = os.path.join(output_dir_annotated, basename)
          cv2.imwrite(ouput_image_file_annotated, _non_mask_img)
          print("--- create a annotated image file {}".format(ouput_image_file_annotated))

        xbasename = basename.lower()
        img_file_path = os.path.join(output_subdir, xbasename)

        non_mask_img = cv2.imread(image_file) #cv2.BGR2RGB)
      
        cv2.imwrite(img_file_path, non_mask_img) #, [cv2.IMWRITE_JPEG_QUALITY, 95])
      
        DELIMITER = "--"
        filename = basename.split(DELIMITER)[1]
        class_id = classes_definition[filename]

        #shutil.copy2(org_image, output_sub_dir)
        annotation = str(class_id ) + SP + str(rcx) + SP + str(rcy) + SP + str(rw) + SP + str(rh) 
        annotation_file = name + ".txt"
        annotation_file_path = os.path.join(output_subdir, annotation_file)
        with open(annotation_file_path, "w") as f:
          f.writelines(annotation + NL)
          print("---Created annotation file {}".format(annotation_file_path))
          print("---YOLO annotation {}".format(annotation))

"""
Input:
./OTU_Images_master_512x512/
├─test/
├─train/
└─valid/

"""

"""
Output:
./YOLO/
├─test/
├─train/
└─valid/
      
"""

"""
categories [0, 1, 2, 3, 4, 5, 6, 7]
 
"Chocolate cyst"        = 0
"Serous cystadenoma"    = 1
"Teratoma"              = 2
"Theca cell tumor"      = 3
"Symple cyst"           = 4
"Normal ovary"          = 5
"Mucinous cystadenoma"  = 6
"High grade serous"     = 7



"""
if __name__ == "__main__":
  try:      
    # create Ovrian UltraSound Images OUS_augmented_master_512x512 dataset train, valid 
    # from the orignal Dataset_.

    # 1 Create train and valid dataset
    train_classes_file = "./train_cls.txt"
    images_dir         = "./images"
    masks_dir          = "./annotations"
  
    #OTU = Ovarian_Tumor_Ultrasound Images
    images_dir  = "./Augmented_OTUSI_master_512x512/images"
    masks_dir   = "./Augmented_OTUSI_master_512x512/masks"
    output_dir  = "./YOLO-Augmented"
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    targets = ["train", "valid"]

    create_yolo_annotation(train_classes_file, images_dir, masks_dir, 
                                    targets,
                                    output_dir,
                                    debug=False)
    
    # 2 Create test dataset
    test_classes_file = "./val_cls.txt"
    targets = ["test"]
    create_yolo_annotation(test_classes_file, images_dir, masks_dir, 
                           targets,
                                    output_dir,
                                    debug=False)
    
  except:
    traceback.print_exc()
    