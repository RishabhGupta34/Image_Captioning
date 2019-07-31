import json

def read_captions(file_path,opt=1): ## opt==1 train, opt==2 val
  data={1:"train",2:"val"}
  with open(file_path,'r') as f:
    a=f.readline()
  p=json.loads(a)
  all_annot={}
  for i in p["annotations"]:
    key="COCO_{}2014_{:012d}".format(data[opt],i['image_id'])
    if(key in all_annot):
      all_annot[key].append(i['caption'])
    else:
      all_annot[key]=[i['caption']]
  return all_annot