import os
import zipfile

os.system('wget http://nlp.stanford.edu/data/glove.6B.zip')  ##GLOVE ENCODINGS
os.system('wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip')  ##COCO_TRAINVAL2014 ANNOTATIONS DATASET
os.system('wget http://images.cocodataset.org/zips/train2014.zip')  ##COCO_TRAIN2014 DATASET
os.system('wget http://images.cocodataset.org/zips/val2014.zip')  ##COCO_VAL2014 DATASET
os.system('wget http://images.cocodataset.org/zips/test2014.zip')  ##COCO_TEST2014 DATASET

def extract_zip(path_to_zip,path_to_extract='./'):
  zip_ref = zipfile.ZipFile(path_to_zip, 'r')
  zip_ref.extractall(path_to_extract)
  zip_ref.close()

print("Extracting Glove Encodings")
extract_zip('glove.6B.zip','./dataset/glove_data/')
print("Extracted Glove Encodings")

print("Extracting Train Dataset")
extract_zip('train2014.zip','./dataset/')
print("Extracted Train Dataset")

print("Extracting Test Dataset")
extract_zip('test2014.zip','./dataset/')
print("Extracted Test Dataset")

print("Extracting Validation Dataset")
extract_zip('val2014.zip','./dataset/')
print("Extracted Validation Dataset")

print("Extracting Annotations Dataset")
extract_zip('annotations_trainval2014.zip','./dataset/')
print("Extracted Annotations Dataset")
