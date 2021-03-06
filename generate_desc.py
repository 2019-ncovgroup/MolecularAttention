import multiprocessing
from mordred import descriptors, Calculator
import argparse
from functools import partial
import numpy as np
from features.generateFeatures import smile_to_mordred
import pandas as pd
from rdkit import Chem

from train import validate_smiles
from tqdm import tqdm
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True)
    parser.add_argument('-o', type=str, required=True)
    parser.add_argument('-n', type=int, required=False, default=8)

    return parser.parse_args()


if __name__=='__main__':
    args = get_args()
    df = pd.read_csv(args.i, header=None)
    smiles = []
    with multiprocessing.Pool(args.n) as p:
        gg = filter(lambda x: x is not None, p.imap(validate_smiles, list(df.iloc[:, 0])))
        for g in tqdm(gg, desc='validate smiles'):
            smiles.append(g)
    del df

    calc = Calculator(descriptors, ignore_3D=True)
    mols = map(Chem.MolFromSmiles, smiles)
    df = calc.pandas(mols, nproc=args.n)
    df = np.array(df, dtype=np.float16)
    pd.DataFrame(df).to_hdf(args.o +".hdf", 'data')
    df = np.nan_to_num(df, posinf=0, neginf=0, nan=0)

    try:
        np.save(args.o, df)
    except:
        exit()
    # descs = np.stack(descs).astype(np.float16)
    # np.save(args.o, descs)