# imports
import os
import csv
import sys
from rdkit import Chem
import torch
import torch.nn	as nn
torch.manual_seed(8) # for reproduce
CUDA_VISIBLE_DEVICES = 0
import numpy as	np
import sys
sys.setrecursionlimit(50000)
#torch.backends.cudnn.benchmark = True
#torch.set_default_tensor_type('torch.FloatTensor')
# from tensorboardX	import SummaryWriter
#torch.nn.Module.dump_patches = True
import copy
import pandas as pd
#then import my	own	modules
from AttentiveFP import	Fingerprint, Fingerprint_viz, save_smiles_dicts, get_smiles_array
#import setting
from sarpy.SARpytools import *



input_file = sys.argv[1]
output_file = sys.argv[2]

# current file directory
root = os.path.dirname(os.path.abspath(__file__))
# checkpoints directory
checkpoints_dir = os.path.abspath(os.path.join(root, "..", "..", "checkpoints"))
model_pretrained=os.path.join(checkpoints_dir, "model_ppb_3922_Tue_Dec_22_22-23-22_2020_54.pt")

filename = input_file.replace('.csv','')

batch_size = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# my model
def my_model(smiles_list):
        
    df = pd.DataFrame({'can_smiles': smiles_list})

    smiles_tasks_df = df.copy()
    smilesList = smiles_tasks_df.can_smiles.values
        
    atom_num_dist = []
    remained_smiles = []
    canonical_smiles_list = []
    for smiles in smilesList:
        try:
            mol = Chem.MolFromSmiles(smiles)
            atom_num_dist.append(len(mol.GetAtoms()))
            remained_smiles.append(smiles)
            canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
        except:
            print(smiles)
            pass
    print("number of successfully processed smiles: ", len(remained_smiles))
    smiles_tasks_df = smiles_tasks_df[smiles_tasks_df["can_smiles"].isin(remained_smiles)]
    # print(smiles_tasks_df)
    smiles_tasks_df['can_smiles'] =canonical_smiles_list
    assert canonical_smiles_list[0]==Chem.MolToSmiles(Chem.MolFromSmiles(smiles_tasks_df['can_smiles'][0]), isomericSmiles=True)


    #Calcule the molecula feature
    feature_dicts = save_smiles_dicts(smilesList,filename)
    remained_df = smiles_tasks_df[smiles_tasks_df["can_smiles"].isin(feature_dicts['smiles_to_atom_mask'].keys())]
    uncovered_df = smiles_tasks_df.drop(remained_df.index)
    print(str(len(uncovered_df.can_smiles))+' compounds cannot be featured')
    remained_df = remained_df.reset_index(drop=True)


    #Load the model
    batch_size = 64
    p_dropout= 0.1
    fingerprint_dim = 200

    weight_decay = 5 # also known as l2_regularization_lambda
    learning_rate = 2.5
    output_units_num = 1 # for regression model
    radius = 2
    T = 2

    x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array([canonical_smiles_list[0]],feature_dicts)
    num_atom_features = x_atom.shape[-1]
    num_bond_features = x_bonds.shape[-1]
    loss_function = nn.MSELoss()
    model = Fingerprint(radius, T, num_atom_features, num_bond_features,
                fingerprint_dim, output_units_num, p_dropout)
    
    model.to(device) ####change to CPU
    #model.cuda()

    best_model = torch.load(model_pretrained, map_location=torch.device('cpu')) ###change adding map_location
    #best_model = torch.load(model_pretrained)

    best_model_dict = best_model.state_dict()
    best_model_wts = copy.deepcopy(best_model_dict)

    model.load_state_dict(best_model_wts)
    (best_model.align[0].weight == model.align[0].weight).all()

    model_for_viz = Fingerprint_viz(radius, T, num_atom_features, num_bond_features,
                fingerprint_dim, output_units_num, p_dropout)
    #model_for_viz.cuda()
    model_for_viz.to(device) ####change to cpu

    model_for_viz.load_state_dict(best_model_wts)
    (best_model.align[0].weight == model_for_viz.align[0].weight).all()

    #Predict values
    remain_pred_list = eval(model, remained_df,feature_dicts)
    remained_df['Predicted_values'] = remain_pred_list

    remained_df.to_csv('temp.csv', index=False)


def eval(model, dataset,feature_dicts):
    model.eval()
#    eval_MAE_list = []
#    eval_MSE_list = []
#    y_val_list = []
    y_pred_list = []
    valList = np.arange(0,dataset.shape[0])
    batch_list = []
    for i in range(0, dataset.shape[0], batch_size):
        batch = valList[i:i+batch_size]
        batch_list.append(batch)
    for counter, eval_batch in enumerate(batch_list):
        batch_df = dataset.loc[eval_batch,:]
        smiles_list = batch_df.can_smiles.values
#         print(batch_df)
#        y_val = batch_df[tasks[0]].values

        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list,feature_dicts)
        
        atoms_prediction, mol_prediction = model(torch.Tensor(x_atom).to(device),torch.Tensor(x_bonds).to(device),torch.LongTensor(x_atom_index).to(device),torch.LongTensor(x_bond_index).to(device),torch.Tensor(x_mask).to(device))
#        MAE = F.l1_loss(mol_prediction, torch.Tensor(y_val).view(-1,1), reduction='none')
#        MSE = F.mse_loss(mol_prediction, torch.Tensor(y_val).view(-1,1), reduction='none')
#         print(x_mask[:2],atoms_prediction.shape, mol_prediction,MSE)
#        y_val_list.extend(y_val)
        y_pred_list.extend(np.array(mol_prediction.data.squeeze().cpu().numpy()))

#        eval_MAE_list.extend(MAE.data.squeeze().cpu().numpy())
#        eval_MSE_list.extend(MSE.data.squeeze().cpu().numpy())
#    r2 = r2_score(y_val_list,y_pred_list)
#    pr2 = scipy.stats.pearsonr(y_val_list,y_pred_list)[0]
    return y_pred_list

def dataset(compounds):
    dataset = []
    for compound in compounds:
        compound = readstring('smi',compound)
        structure = Structure(compound)
        dataset.append(structure)
    return dataset


# parse arguments

# read SMILES from .csv file, assuming one column with header
with open(input_file, "r") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    smiles_list = [r[0] for r in reader]

# run model
my_model(smiles_list)
