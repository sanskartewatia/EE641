import pickle

pkl_file_path = '../testing_annots.pkl'

with open(pkl_file_path, 'rb') as file:
    data = pickle.load(file)


new_annts = []
for i in range(len(data)):
    if data[i][0].split('/')[1] == 'v_Surfing_g07_c01':
        new_annts.append(data[i])
        
fp = '../visual_annots.pkl'
with open(fp,'wb') as wid:
    pickle.dump(new_annts, wid, protocol=4)
print("Saved at ", fp)