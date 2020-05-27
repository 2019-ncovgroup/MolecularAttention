import numpy as np
import pandas as pd
import os
import sys
from rdkit import Chem
import multiprocessing
import argparse
from tqdm import tqdm

# Takes a raw input csv file with a smiles column and dock scores for each target column                                                 
# and a .npy file of images generated from generate_images.py for all valid smiles in the raw input file                                 
# Creates directories of the format DIR.ml.<target_name>.images for each target containing:                                              
# (1) smiles .csv file for all valid smiles with valid dock scores (not nan) for the given target                                        
# (2) dock score .npy file with valid scores (clipped to -100 and absolute valued) for the given target                                  
# (3) image.npy file with the corresponding images retrieved from the -i image input file                                                 
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', type=str, help='raw input file')
    parser.add_argument('-i', type=str, help='image input file')
    args = parser.parse_args()
    return args

def validate_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(smiles)
    return smiles if mol is not None else None

# find the invalid indices                                                                                                               
def drop_indices(df):
    discard = []
    smiles=[]
    with multiprocessing.Pool() as p:
        gg = p.imap(validate_smiles,list(df.iloc[:,0]))
        i=0
        for g in tqdm(gg, desc='validate smiles'):
            if g is None :
                discard.append(i)
            else:
                 smiles.append(g)
            i+=1
    print("Discarding values at inds", discard)
    return discard

if __name__ == '__main__':
    args = get_args()
    df_r = pd.read_csv(args.r, sep='\t')
    images = np.load(args.i)
    smiles = pd.DataFrame(df_r.canonical_smile)

    # drop invalid smiles rows 
    d = drop_indices(smiles)
    df_r = df_r.drop(d)
    for target_name in ['NSUN6']: #df_r.columns[2:]:
        print(target_name)
        target_df = df_r[['canonical_smile', target_name]]
        print(target_df.shape)

        # find na locations, drop dock scores
        bool_arr = target_df[target_name].isna()
        target_df = target_df.dropna()
        print(target_df.shape)
        smiles = target_df.iloc[:,0]
        dock = np.clip(target_df.iloc[:,1],-100,0).abs()

        #find corresponding images
        print(bool_arr.shape)
        idx = np.argwhere(bool_arr.values==False)
        print(len(idx))
        images_target = images[idx.flatten()]
        assert(images_target.shape[0] == smiles.shape[0])
        assert(smiles.shape[0] == dock.shape[0])

        # now save the files
        save_path = "data_v5/DIR.ml." + target_name + ".images"
        os.mkdir(save_path)
        np.save(save_path + "/ml."+ target_name+ ".reg", dock.to_numpy())
        smiles.to_csv(save_path + "/ml."+target_name+".smi", index=False)
        np.save(save_path + "/ml."+target_name+".images", images_target)
