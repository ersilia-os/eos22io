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
    print(str(len(uncovered_df.can_smiles))+' compounds cannot be featured')
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
    
    #remained_df.to_csv('temp.csv', index=False)

    #Get atom attention weights
    # Notably: for more than 500 compounds, be cautious!
    smi_aw = get_smi_aw(remained_df,model_for_viz)
    len(smi_aw)

    #Identify Privileged Substructure for each molecule. 
    df_ppb_psubs= pd.DataFrame()
    psubs_list=[]
    ppb_fractation_list=[]
    for key,value in smi_aw.items():
        print(key)
        grinder = Grinder(3,18)
        fragment_list = fragments(collectSubs(dataset([key]),grinder))
        psubs = []
        for fragment in fragment_list:
            patt = Chem.MolFromSmarts(fragment)
            smiles = Chem.MolFromSmiles(key)
            atom_numbers = smiles.GetSubstructMatches(patt)
            for j in range(len(atom_numbers)):
                fragment_atoms = [smi_aw[key][i] for i in atom_numbers[j]]
                rest_atoms = [smi_aw[key][i] for i in range(0,len(smi_aw[key]),1) if i not in atom_numbers[j]]
                if len(rest_atoms) < 3:
                    pass
                else:
                    try:
                        p_value = stats.mannwhitneyu(fragment_atoms,rest_atoms,alternative = 'greater')[1]
                        if p_value < 0.05:
                            psubs.append(fragment)
                    except ValueError:
                        pass
        psubs_del = []
        for i in range(len(psubs)-1):
            for j in range(i+1,len(psubs)):
                patt1 = Chem.MolFromSmarts(psubs[i])
                patt2 = Chem.MolFromSmarts(psubs[j])
                smi1 = Chem.MolFromSmiles(psubs[i])
                smi2 = Chem.MolFromSmiles(psubs[j])
                frag1 = smi2.HasSubstructMatch(patt1)
                frag2 = smi1.HasSubstructMatch(patt2)
                if frag1 == True:
                    if frag2 == False:
                        psubs_del.append(psubs[i])
                if frag1 == False:
                    if frag2 == True:
                        psubs_del.append(psubs[j])
        psub = [p for p in psubs if p not in psubs_del]
        
        #create a psubestructra and ppb fraction of the substructures
        psubs_list.append(psub)
        ppb_fractation_list.append(str(remained_df[remained_df.can_smiles == key].Predicted_values.values[0]))

    df_ppb_psubs['Priviledged Substructures']= psubs_list
    df_ppb_psubs['PPB fraction']= ppb_fractation_list

    df_ppb_psubs.to_csv(output_file, index= False)

# read SMILES from .csv file, assuming one column with header
with open(input_file, "r") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    smiles_list = [r[0] for r in reader]

# run model
my_model(smiles_list)
