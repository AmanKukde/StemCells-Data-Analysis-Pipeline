import tifffile as tf
import os 
from re import search
from PIL import Image
import numpy as np
from tqdm import tqdm
from natsort import natsorted

# path = "/Users/akukde/Desktop/2022_06_01_test3_ablation_pauline"
# path = "/Users/aman/Desktop/pasteur/2022-06-22-test-control-10-30/"
path = "/Users/akukde/Desktop/pasteur/2022-06-22-test-control-10-30/"

files_list = os.listdir(path)
files_list.sort()
print(len(files_list))
# files_list[:10]


correct_files_list = []
for file in files_list:
    # if file[-4:] != '.TIF':
    if ".DS_" in file or file.__contains__(".nd") or file.__contains__(".rgn") or file.__contains__("_thumb_"):
        # print(file)
        pass
    else:    
        correct_files_list.append(file)
# print(len(files_list))

list1 = []
for file in correct_files_list:
    try:
        list1.append(file.split("_")[1])
    except IndexError:
        pass
list1 = list(dict.fromkeys(list1))

final = []
for ele in tqdm(list1):
    trial_i = [s for s in correct_files_list if f"_{ele}_" in s]
    trial_i = natsorted(trial_i)
    for t in tqdm(trial_i):
    
        im = Image.open(os.path.join(path, t)) # fname = "im_{ele}_w1CSU488A_t3.TIF"
        imarray = np.array(im)
        final.append(imarray)
    final_new = np.asarray(final)
    # im = Image.fromarray(final_new, mode='F') # float32
    # im.save(f"{path}{ele}.tif", "TIFF") 
    # im.save(f"/Users/akukde/Desktop/pasteur/data_2/{ele}.tif", "TIFF") 
    tf.imsave(f"/Users/akukde/Desktop/pasteur/data_2/{ele}.tif",final_new)
    final = []

