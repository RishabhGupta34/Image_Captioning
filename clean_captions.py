import re
def clean_text(text):
  text=text.lower()
  text=re.sub("[^a-zA-Z]"," ",text)
  text="startseq "+text+" endseq"
  text=text.split()
  text=[w for w in text if len(w)>1]
  return " ".join(text)

def clean_captions(image_captions):
  keys=list(image_captions.keys())
  for i in keys:
    clean_cap=[clean_text(cap) for cap in image_captions[i]]
    image_captions[i]=clean_cap
  return image_captions