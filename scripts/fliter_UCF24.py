import os
import pickle
import shutil

pkl_file_path = 'training_annots.pkl'

with open(pkl_file_path, 'rb') as file:
    data = pickle.load(file)

src_root = '../data/UCF101/UCF101_Videos'
dst_root = '../data/UCF101/UCF101-24_Videos/train'

if not os.path.exists(dst_root):
    os.makedirs(dst_root)

for i, video in enumerate(data):
    v_name = video[0]
    v_path = os.path.join(src_root, '%s.avi' % v_name)
    shutil.copy(v_path, dst_root)
    print(f'No.{i + 1} video: {v_name} has been copied.')

