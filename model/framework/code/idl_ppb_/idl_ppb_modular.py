import sys
import warnings
import numpy as	np
from rdkit import Chem
from rdkit.Chem	import QED
from rdkit import Chem
try:
    from rdkit import Chem
except:
    warnings.warn('Cannot import rdkit or cairosvg, so report is not supported')
    Chem = None
    setting.report_substructures = False

import torch
sys.setrecursionlimit(50000)
torch.manual_seed(8) # for reproduce
torch.backends.cudnn.benchmark = True
torch.set_default_tensor_type('torch.FloatTensor')
torch.nn.Module.dump_patches = True

#then import my	own	modules
from AttentiveFP import get_smiles_dicts, get_smiles_array
from sarpy.SARpytools import *

radius = 2
T = 2

def smiles2canonical(smilesList):

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
    return canonical_smiles_list

def collectSubs(structures, grinder):
    if not structures:
        return []
    substructures = []
    for structure in structures:
        substructures.extend(grinder.getFragments(structure))
    #print(fragments(substructures)) #substructures generate by each iteration
#    print('%s\tsubstructures found...' % len(substructures))
    return substructures + collectSubs(substructures, grinder)

def fragments(substructures):
    fragments = []
    for i in range(len(substructures)):
        fragments.append(substructures[i].smiles)
    return fragments

def eval(model, dataset,feature_dicts):
    batch_size = 64
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
        
        atoms_prediction, mol_prediction = model(torch.Tensor(x_atom),torch.Tensor(x_bonds),torch.LongTensor(x_atom_index),torch.LongTensor(x_bond_index),torch.Tensor(x_mask))
#        MAE = F.l1_loss(mol_prediction, torch.Tensor(y_val).view(-1,1), reduction='none')
#        MSE = F.mse_loss(mol_prediction, torch.Tensor(y_val).view(-1,1), reduction='none')
#         print(x_mask[:2],atoms_prediction.shape, mol_prediction,MSE)
#        y_val_list.extend(y_val)
        y_pred_list.extend(np.array(mol_prediction.data.squeeze(axis=1).cpu().numpy()))
        
        '''if np.shape(mol_prediction.data)[0] > 1:
            y_pred_list.extend(np.array(mol_prediction.data.squeeze().cpu().numpy()))
        else:
            y_pred_list.extend(np.array(mol_prediction.data.cpu().numpy())[0])'''

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

def eval_for_viz(model, viz_list,batch_size):
    model.eval()
    test_MAE_list = []
    test_MSE_list = []
    mol_prediction_list = []
    atom_feature_list = []
    atom_weight_list = []
    mol_feature_list = []
    mol_feature_unbounded_list = []
    batch_list = []
    for i in range(0, len(viz_list), batch_size):
        batch = viz_list[i:i+batch_size]
        batch_list.append(batch)
    for counter, batch in enumerate(batch_list):
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(viz_list,get_smiles_dicts(batch))
        atom_feature_viz, atom_attention_weight_viz, mol_feature_viz, mol_feature_unbounded_viz, mol_attention_weight_viz, mol_prediction = model(
            torch.Tensor(x_atom), torch.Tensor(x_bonds),
            torch.LongTensor(x_atom_index),
            torch.LongTensor(x_bond_index),
            torch.Tensor(x_mask))
        #print(mol_attention_weight_viz[1].cpu().detach().numpy())
        mol_prediction_list.append(mol_prediction.cpu().detach().squeeze().numpy())
        atom_feature_list.append(np.stack([atom_feature_viz[L].cpu().detach().numpy() for L in range(radius+1)]))
        atom_weight_list.append(np.stack([mol_attention_weight_viz[t].cpu().detach().numpy() for t in range(T)]))
        mol_feature_list.append(np.stack([mol_feature_viz[t].cpu().detach().numpy() for t in range(T)]))
        mol_feature_unbounded_list.append(np.stack([mol_feature_unbounded_viz[t].cpu().detach().numpy() for t in range(T)]))

    mol_prediction_array = np.concatenate(mol_prediction_list,axis=0)
    atom_feature_array = np.concatenate(atom_feature_list,axis=1)
    #print(atom_weight_list)
    atom_weight_array = np.concatenate(atom_weight_list,axis=1)
    #print(atom_weight_array)
    mol_feature_array = np.concatenate(mol_feature_list,axis=1)
    mol_feature_unbounded_array = np.concatenate(mol_feature_unbounded_list,axis=1)
#     print(mol_prediction_array.shape, atom_feature_array.shape, atom_weight_array.shape, mol_feature_array.shape)
    return mol_prediction_array, atom_feature_array, atom_weight_array, mol_feature_array, mol_feature_unbounded_array

def get_smi_aw(remained_df,model_for_viz):
    batch_size = len(remained_df.can_smiles.values)
    train_df_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(remained_df.can_smiles.values[m]),isomericSmiles=True) for m in range(0,batch_size)]
    mol_prediction_array, atom_feature_array, atom_weight_array, mol_feature_array, mol_feature_unbounded_array =  eval_for_viz(model_for_viz, train_df_smiles,batch_size)
    x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(train_df_smiles,get_smiles_dicts(train_df_smiles))
    smi_aw = {}
    for i in range(batch_size):
        smiles = train_df_smiles[i]
        index_atom = smiles_to_rdkit_list[smiles]
        weight_recoder = np.stack([atom_weight_array[:,i][0][m] for m in np.argsort(index_atom)]) #1 is wrong
        atom_weight = [weight_recoder[m][0] for m in range(len(weight_recoder))]
        smi_aw[smiles] = atom_weight  #atom_weight
        #smi_ac[smiles] = np.sum(atom_weight)
    return smi_aw

def collectSubs(structures, grinder):
    if not structures:
        return []
    substructures = []
    for structure in structures:
        substructures.extend(grinder.getFragments(structure))
    #print(fragments(substructures)) #substructures generate by each iteration
#    print('%s\tsubstructures found...' % len(substructures))
    return substructures + collectSubs(substructures, grinder)

def fragments(substructures):
    fragments = []
    for i in range(len(substructures)):
        fragments.append(substructures[i].smiles)
    return fragments