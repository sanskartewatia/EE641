import pickle
import pandas as pd
import os
import numpy as np

pkl_file_path = '../testing_annots.pkl'
val_KNN_dir = '../data/UCF101/UCF101-24_Videos/val_random'
indices_dir = os.path.join(val_KNN_dir, 'indices')

with open(pkl_file_path, 'rb') as file:
    data = pickle.load(file)

new_val_anns = []
for i in range(len(data)):
    v_name, anns = data[i]
    avi_name = v_name.split('/')[1]
    indices = pd.read_csv(f'{indices_dir}/{avi_name}.csv', header=None)[0].to_numpy()
    new_anns = [] 
    for ann in anns:
        select_bboxes, action_frames = [], []
        start_frame, end_frame, label, bboxes = ann[0], ann[1], ann[2], ann[3]
        if bboxes.shape[0] != end_frame - start_frame + 1:
            print("not match")
        
        for i, id in enumerate(indices):
            if start_frame <= id <= end_frame:
                x, y, w, h = bboxes[id - start_frame]
                select_bboxes.append([x,y,w,h])
                action_frames.append(i)
        if len(action_frames):
            new_sf = action_frames[0]
            new_ef = action_frames[-1]
            new_anns.append((new_sf, new_ef, label, select_bboxes, action_frames))
    if len(new_anns):
        new_val_anns.append((avi_name, new_anns))       
                
fp = '../testing_annots_random_20percent.pkl'
with open(fp,'wb') as wid:
    pickle.dump(new_val_anns, wid, protocol=4)
print("Saved at ", fp)