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
import numpy as np

import traceback
import cv2
from PIL import Image, ImageOps, ImageEnhance


class OvarianTumorImageAugmentor:
  def __init__(self, W=512, H=512):
    self.W = W
    self.H = H

  def read_classes_file(self, classes_file):
    filename_classid = {}
    with open(classes_file, "r") as f:
      lines = f.readlines()
      for line in lines:
        line = line.strip()
        print("--- line {}".format(line))
        array             = line.split(" ")
        print("--- array {}".format(array))
        filename, classid = line.split("  ")
        filename_classid[filename] = classid
        print(" --- {} = {}".format(filename, classid))
    return filename_classid



  def get_mask_filepaths(self, annotations_dir):
    pattern = annotations_dir + "/*.png"
    print("--- pattern {}".format(pattern))
    all_files  = glob.glob(pattern)
    mask_filepaths = []
    for file in all_files:
      basename = os.path.basename(file)
      if basename.find("_") == -1:
        mask_filepaths.append(file)
    return mask_filepaths


  def  get_train_images_filepaths(self, train_classes_definition, image_filepaths):
    train_images_filepaths = []
    for image_filepath in image_filepaths:
      print("--- {}".format(image_filepath))
      basename = os.path.basename(image_filepath) 
      # *.JPG
      name     = basename.split(".")[0]  
      print("---get_train_images_file image_filepath:{} name:{} ".format(image_filepath, name))
      try:
        class_id = train_classes_definition[basename] 
        train_images_filepaths.append(image_filepath)
        #input("----------------{} ".format(image_filepath))
      except:
        pass
    
    return train_images_filepaths

  def  get_splitted_mask_filepaths(self, masks_dir, train_filepaths):
    masks_filepaths = []
    for train_filepath in train_filepaths:
      # JPG file
      basename = os.path.basename(train_filepath)
      name     = basename.split(".")[0]
      mask_filepath = os.path.join(masks_dir, name + ".PNG")
      masks_filepaths.append(mask_filepath)
    return masks_filepaths


  def create_master(self, train_classes_file, images_dir, masks_dir, 
                                    output_images_dir, output_masks_dir, 
                                    istrain=True, debug=False):
    train_classes_definition  = self.read_classes_file(train_classes_file)

    mask_filepaths  = self.get_mask_filepaths(masks_dir)
    
    pattern = images_dir + "/*.JPG"
    image_filepaths  = glob.glob(pattern)
    #print("---- image_filepaths {}".format(image_filepaths))
    
    train_images_filepaths = self.get_train_images_filepaths(train_classes_definition, image_filepaths)
    #print("--- train_images_filepths {}".format(train_images_filepaths))
    #input("----------HIT")
    print("--- pattern {}".format(pattern))

    num_files   = len(train_images_filepaths)
    # 1 shuffle mask_files
    random.shuffle(train_images_filepaths)
    
    #input("--------------HIT")
    
    # 2 Compute the number of images to split
    # train= 0.8 valid=0.2
    if istrain:
      num_train = int(num_files * 0.5)
      num_valid = int(num_files * 0.5)

      train_filepaths  = train_images_filepaths[0        : num_train]
      valid_filepaths  = train_images_filepaths[num_train: num_train + num_valid]

      print("=== number of train_files {}".format(len(train_filepaths)))
      print("=== number of valid_files {}".format(len(valid_filepaths)))
      #input("--------------HIT")
      self.create_resized_images(train_filepaths, output_images_dir, "train", augment=True,  mask=False)
      self.create_resized_images(valid_filepaths, output_images_dir, "valid", augment=False, mask=False)

      mask_train_filepaths = self.get_splitted_mask_filepaths(masks_dir, train_filepaths)
      mask_valid_filepaths = self.get_splitted_mask_filepaths(masks_dir, valid_filepaths)

      self.create_resized_images(mask_train_filepaths, output_masks_dir, "train", augment=True,  mask=True)
      self.create_resized_images(mask_valid_filepaths, output_masks_dir, "valid", augment=False, mask=True)
    else :
      num_test = num_files
      test_filepaths  = train_images_filepaths[0        : num_files]

      print("=== number of test_files {}".format(len(test_filepaths)))
      self.create_resized_images(test_filepaths, output_images_dir, "test",     augment=False, mask=False)

      mask_test_filepaths = self.get_splitted_mask_filepaths(masks_dir, test_filepaths)

      self.create_resized_images(mask_test_filepaths, output_masks_dir, "test", augment=False, mask=True)


  def create_resized_images(self, image_filepaths, output_dir, target, augment=False, mask=False):

    for image_filepath in image_filepaths:
      img = Image.open(image_filepath)
      print("---create_resized_512x512_images {}".format(image_filepath))
      basename = os.path.basename(image_filepath)
      array    = basename.split(".")
      name     = array[0]
      image_format = array[1]

      w, h = img.size
      max = w
      if h > w:
        max = h
      if max < self.W:
        max = self.W
      # 1 Create a black background image
      background = Image.new("RGB", (max, max), (0, 0, 0))

      # 2 Paste the originral img to the background image at (x, y)
      print(img.format, img.size, img.mode)
      print(background.format, background.size, background.mode)

      x = int( (max - w)/2 )
      y = int( (max - h)/2 )
      background.paste(img, (x, y))

      output_subdir   = os.path.join(output_dir, target)
      if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)

      output_filepath = os.path.join(output_subdir, basename)

      # 3 Resize the backtround to 512x512 

      background_512x512 = background.resize((self.W, self.H))
      if mask:
        background_512x512 = self.convert2WhiteMask(background_512x512)
        #background_512x512.show()
        #input("HIT")
    
      self.augment_image(background_512x512, name, image_format, output_subdir, augment=augment)



  def augment_image(self, img, nameonly, image_format, output_dir, augment=False):
    W, H = img.size
    resized_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    ANGLES = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    #ANGLES = [0, 50, 100, 150, 200, 250, 300, 350]
    #ANGLES = [0, 60, 120, 180, 240, 300]
    DELIMITER = "--"
    if augment==False:
      ANGLES = [0]
    for angle in ANGLES:
      center = (self.W/2, self.H/2)
      rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)

      # 1 Rotate the resize_img by angle
      rotated_resized_image    = cv2.warpAffine(src=resized_img, M=rotate_matrix, dsize=(self.W, self.H))
      rotated_resized_filename = "rotated-" + str(angle) +  DELIMITER + nameonly + "." + image_format
      rotated_resized_filepath = os.path.join(output_dir, rotated_resized_filename)

      # 2 Write the rotated_resized_img as a jpg file.
      cv2.imwrite(rotated_resized_filepath, rotated_resized_image)
      print("Saved {} ".format(rotated_resized_filepath))

    if augment== False:
      return
    
    FLIPCODES = [0, 1]
    for flipcode in FLIPCODES:
      # 3 Flip the resized_mask_img by flipcode
      flipped_resized_img = cv2.flip(resized_img, flipcode)
      # Save flipped mask_filename is jpg
      save_flipped_img_filename = "flipped-" + str(flipcode) +  DELIMITER + nameonly + "." + image_format
      flipped_resized_img_filepath = os.path.join(output_dir, save_flipped_img_filename )

      # 4 Write the flipped_resized_mask_img as a jpg file.
      cv2.imwrite(flipped_resized_img_filepath, flipped_resized_img)
      print("Saved {} ".format(flipped_resized_img_filepath))

    
  def convert2WhiteMask(self, image):
    w, h = image.size
    for y in range(h):
      for x in range(w):
        pixel = image.getpixel((x, y))
        if pixel != (0, 0, 0):
          pixel = (255, 255, 255) #White
          image.putpixel((x, y), pixel)
    return image




"""
Output:
./Augmented_OTUSI_master_512x512/images"
├─images/
└─masks/
 
      
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
    output_images_dir  = "./Augmented_OTUSI_master_512x512/images"
    output_masks_dir   = "./Augmented_OTUSI_master_512x512/masks"

    if os.path.exists(output_images_dir):
      shutil.rmtree(output_images_dir)
    if not os.path.exists(output_images_dir):
      os.makedirs(output_images_dir)

    if os.path.exists(output_masks_dir):
      shutil.rmtree(output_masks_dir)
    if not os.path.exists(output_masks_dir):
      os.makedirs(output_masks_dir)

    augmentor = OvarianTumorImageAugmentor(W=512, H=512)

    augmentor.create_master(train_classes_file, images_dir, masks_dir, 
                                    output_images_dir, output_masks_dir, 
                                    istrain =True,
                                    debug=False)
    
    # 2 Create test dataset
    test_classes_file = "./val_cls.txt"

    augmentor.create_master(test_classes_file, images_dir, masks_dir, 
                                    output_images_dir, output_masks_dir, 
                                    istrain=False,
                                    debug=False)
    
  except:
    traceback.print_exc()
    