import os
from seg_module import ai_segmentation, learning_based_lung_cropping
from cls_module import ai_classification
import SimpleITK as sitk
import heapq
import argparse
import time

class TestOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="test options")
        self.parser.add_argument("--id", type=int, default=1)

    def parse(self):
        return self.parser.parse_args()

if __name__ == "__main__":
    opt = TestOptions()
    opt = opt.parse()

    reader = sitk.ImageSeriesReader()
    img_names = reader.GetGDCMSeriesFileNames(os.path.join('data', str(opt.id))) # change to load your own ct data here
    reader.SetFileNames(img_names)
    image = reader.Execute()
    CT = sitk.GetArrayFromImage(image)
   
    CT = learning_based_lung_cropping(CT)
    
    detect_conf, seg_mask = ai_segmentation(CT) # the detection confidence and the segmentation mask
    disease_pred = ai_classification(CT, seg_mask) # the classification probability vector
    
    top3_list = heapq.nlargest(3, range(len(disease_pred)), disease_pred.take)
    disease_list = ['thymoma', 'benign cyst', 'thymic carcinoma', 'germ cell tumor', 'other soft tissue tumor', 'neuroendocrine tumor', \
        'thymic hyperplasia', 'lymphoma', 'lymphadenosis', 'ectopicthyroidgland', 'granulomatous inflammation', 'neurogenic tumor']
    print(f'top3 prediction: {disease_list[top3_list[0]]}, {disease_list[top3_list[1]]}, {disease_list[top3_list[2]]}')