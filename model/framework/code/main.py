import csv
import copy
import os
import sys
import pandas as pd
from rdkit import Chem
from scipy import stats
import torch
import torch.nn	as nn
torch.manual_seed(8) # for reproduce
CUDA_VISIBLE_DEVICES = 0
sys.setrecursionlimit(50000)
torch.backends.cudnn.benchmark = True
torch.set_default_tensor_type('torch.FloatTensor')
torch.nn.Module.dump_patches = True

#then import my	own	modules
from AttentiveFP import	Fingerprint, Fingerprint_viz, save_smiles_dicts, get_smiles_array
from sarpy.SARpytools import *
from idl_ppb_.idl_ppb_modular import *

#parse arguments
input_file = sys.argv[1]
output_file = sys.argv[2]

# current file directory
root = os.path.dirname(os.path.abspath(__file__))
# checkpoints directory
checkpoints_dir = os.path.abspath(os.path.join(root, "..", "..", "checkpoints"))
model_pretrained=os.path.join(checkpoints_dir, "model_ppb_3922_Tue_Dec_22_22-23-22_2020_54.pt")

filename = input_file.replace('.csv','')

batch_size = 64
radius = 2
T = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# my model
def my_model(smiles_list):
    df = pd.DataFrame({'can_smiles': smiles_list})
    smiles_tasks_df = df.copy()
    smilesList = smiles_tasks_df.can_smiles.values
    remained_smiles=smiles_list

    #smiles into canonical form
    canonical_smiles_list= smiles2canonical(smilesList)
        
    smiles_tasks_df = smiles_tasks_df[smiles_tasks_df["can_smiles"].isin(remained_smiles)]
    smiles_tasks_df['can_smiles'] =canonical_smiles_list
    assert canonical_smiles_list[0]==Chem.MolToSmiles(Chem.MolFromSmiles(smiles_tasks_df['can_smiles'][0]), isomericSmiles=True)
    #Calcule the molecula feature
    feature_dicts = save_smiles_dicts(smilesList,filename)
    remained_df = smiles_tasks_df[smiles_tasks_df["can_smiles"].isin(feature_dicts['smiles_to_atom_mask'].keys())]
    uncovered_df = smiles_tasks_df.drop(remained_df.index)
    remained_df = remained_df.reset_index(drop=True)
    #Load the model
    p_dropout= 0.1
    fingerprint_dim = 200
    weight_decay = 5 # also known as l2_regularization_lambda
    learning_rate = 2.5
    output_units_num = 1 # for regression model
    x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array([canonical_smiles_list[0]],feature_dicts)
    num_atom_features = x_atom.shape[-1]
    num_bond_features = x_bonds.shape[-1]
    loss_function = nn.MSELoss()
    model = Fingerprint(radius, T, num_atom_features, num_bond_features,
                fingerprint_dim, output_units_num, p_dropout)
    
    model.to(device) ####change to CPU
    #model.cuda()
    best_model = torch.load(model_pretrained, map_location=torch.device('cpu')) ###change adding map_location
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

    #making sure it returns all values.For the compounds that it is not possible to calculate the features the output is null
    array_ppbs= remained_df['Predicted_values'].values
    array_missindex= uncovered_df.index.values
    for indice in array_missindex:
        if indice < len(array_ppbs):
            array_ppbs = np.insert(array_ppbs, indice, None)
    return array_ppbs

# read SMILES from .csv file, assuming one column with header
with open(input_file, "r") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    smiles_list = [r[0] for r in reader]

# run model
outputs=my_model(smiles_list)

# wirte PPB values output in a .csv file
with open(output_file, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["ppb_fraction"])  # header
    for o in outputs:
        writer.writerow([o])


if os.path.exists(input_file.split(".")[0]+".pickle"):
    os.remove(input_file.split(".")[0]+".pickle")

for file in ["Dropout.patch","Fingerprint.patch", "GRUCell.patch", "Linear.patch", "ModuleList.patch"]:
    if os.path.exists(os.path.join(root, "..", file)):
        os.remove(os.path.join(root, "..", file))