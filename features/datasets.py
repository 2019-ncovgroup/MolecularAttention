import mordred
import numpy as np
import torch
from rdkit import Chem
from torch.utils.data import Dataset
from features.generateFeatures import smiles_to_image, smile_to_mordred
from torchvision import transforms
import pickle
from rdkit import Chem
from rdkit import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

def logps(mol):
    try:
        return list(mordred.Calculator(mordred.SLogP.SLogP)(mol).values())[0]
    except:
        return None


def molecular_weight(mol):
    try:
        return list(mordred.Calculator(mordred.Weight.Weight)(mol).values())[0]
    except:
        return None


def rotate_bond_count(mol):
    try:
        return list(mordred.Calculator(mordred.RotatableBond.RotatableBondsCount)(mol).values())[0]
    except:
        return None


def acid_count(mol):
    try:
        return list(mordred.Calculator(mordred.AcidBase.AcidicGroupCount)(mol).values())[0]
    except:
        return None


def hacceptor_count(mol):
    try:
        return list(mordred.Calculator(mordred.HydrogenBond.HBondAcceptor)(mol).values())[0]
    except:
        return None


def hdonor_count(mol):
    try:
        return list(mordred.Calculator(mordred.HydrogenBond.HBondDonor)(mol).values())[0]
    except:
        return None

def sa_scorer(mol):
    try:
        return sascorer.calculateScore(mol)
    except:
        return None

def custom_prop_blank(mol):
    try:
        return np.nan
    except:
        return None

funcs = {
    'hdonor': hdonor_count,
    'hacceptor': hacceptor_count,
    'acid': acid_count,
    'weight': molecular_weight,
    'logp': logps,
    'rotatable_bonds': rotate_bond_count,
    'all': smile_to_mordred,
    'image': smiles_to_image,
    'sa' : sa_scorer,
    'custom' : custom_prop_blank,
}


def get_properety_function(name):
    return funcs[name]


class MolecularHolder:
    def __init__(self, smiles, values):
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(self.smiles)
        assert (self.mol is not None)
        self.image = None
        self.data = {}
        self.data.update(values)

    def get_image(self):
        if self.image is None:
            self.image = smiles_to_image(self.mol)
            self.data['image'] = self.image
        return self.image

    def get_property(self, name):
        if name in self.data:
            return self.data[name]
        elif name in funcs:
            self.data[name] = funcs[name](self.mol)
            return self.data[name]
        else:
            return None


class ImageDatasetPreLoaded(Dataset):
    def __init__(self, smiles, descs, imputer_pickle=None, property_func=logps, cache=True, values=1, rot=0, images=None, bw=False, mask=None):
        self.smiles = smiles
        self.images = images
        self.descs = descs
        self.property_func = property_func
        self.imputer = None
        self.scaler = None
        if imputer_pickle is not None:
            with open(imputer_pickle, 'rb') as f:
                dd = pickle.load(f)
                self.imputer, self.scaler = dd['imputer'], dd['scaler']
        self.cache = cache
        self.values = values
        self.data_cache = {}
        if mask is not None:
            self.mask = mask
            self.use_mask = True
        else:
            self.use_mask = False
        if bw:
            self.transform = transforms.Compose([transforms.RandomRotation(degrees=(0, rot)), transforms.Grayscale(num_output_channels=3), transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([transforms.RandomRotation(degrees=(0, rot)), transforms.ToTensor()])
        print("Using rotation ", rot)


    def __getitem__(self, item):
        if self.images is not None:
            image = transforms.ToPILImage()(torch.from_numpy(self.images[item].astype(np.float32) / 255.0))
            image = self.transform(image)

            if self.descs is None:
                return image
            
            if self.imputer is not None:
                vec = self.scaler.transform(self.imputer.transform(self.descs[item].reshape(1,-1))).flatten()
            else:
                vec = self.descs[item].flatten()
            vec = torch.from_numpy(np.nan_to_num(vec, nan=0, posinf=0, neginf=0)).float()
            if self.use_mask:
                return image, vec, self.mask[item]
            else:
                return image, vec

        if self.cache and self.smiles[item] in self.data_cache:
            if self.use_mask:
                assert(False)
            image = self.data_cache[self.smiles[item]]
            image = self.transform(image)

            if self.imputer is not None:
                vec = self.scaler.transform(self.imputer.transform(self.descs[item].reshape(1,-1))).flatten()
            else:
                vec = self.descs[item].flatten()
            vec = torch.from_numpy(np.nan_to_num(vec, nan=0, posinf=0, neginf=0)).float()
            return image, vec

        else:
            if self.use_mask:
                assert(False)
            mol = Chem.MolFromSmiles(self.smiles[item])
            image = smiles_to_image(mol)
            if self.imputer is not None:
                vec = self.scaler.transform(self.imputer.transform(self.descs[item].reshape(1,-1))).flatten()
            else:
                vec = self.descs[item].flatten()
            vec = torch.from_numpy(np.nan_to_num(vec, nan=0, posinf=0, neginf=0)).float()
            if self.cache:
                self.data_cache[self.smiles[item]] = image
            image = self.transform(image)
            return image, vec

    def __len__(self):
        return len(self.smiles)


class ImageDataset(Dataset):
    def __init__(self, smiles, property_func=logps, cache=True, values=1):
        self.smiles = smiles
        self.property_func = property_func
        self.cache = cache
        self.values = values
        self.data_cache = {}

    def __getitem__(self, item):
        if self.cache and self.smiles[item] in self.data_cache:
            return self.data_cache[self.smiles[item]]
        else:
            mol = Chem.MolFromSmiles(self.smiles[item])
            image = smiles_to_image(mol)
            property = self.property_func(mol)

            # TODO align property
            if self.values == 1:
                if property is None:
                    property = -1.0
                property = torch.FloatTensor([property]).view((1))
            else:
                property = torch.from_numpy(np.nan_to_num(property, nan=0, posinf=0, neginf=0)).float()

            if self.cache:
                self.data_cache[self.smiles[item]] = (image, property)
            return image, property

    def __len__(self):
        return len(self.smiles)
