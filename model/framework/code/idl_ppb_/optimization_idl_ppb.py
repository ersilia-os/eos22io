import os
import csv
import json
import torch
import torch.autograd as autograd
import torch.nn	as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data	as Data
torch.manual_seed(8) # for reproduce

import time
import numpy as	np
import gc
import sys
sys.setrecursionlimit(50000)
import pickle
torch.backends.cudnn.benchmark = True
torch.set_default_tensor_type('torch.cuda.FloatTensor')
# from tensorboardX	import SummaryWriter
torch.nn.Module.dump_patches = True
import copy
import pandas as pd
#then import my	own	modules
from AttentiveFP import	Fingerprint, Fingerprint_viz, save_smiles_dicts, get_smiles_dicts, get_smiles_array, moltosvg_highlight

from rdkit import Chem
# from rdkit.Chem import AllChem
from rdkit.Chem	import QED
from rdkit.Chem	import rdMolDescriptors, MolSurf
from rdkit.Chem.Draw import	SimilarityMaps
from rdkit import Chem
from rdkit.Chem	import AllChem
from rdkit.Chem	import rdDepictor
from rdkit.Chem.Draw import	rdMolDraw2D
from numpy.polynomial.polynomial import	polyfit
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.pyplot as	plt
from matplotlib	import gridspec
import matplotlib.cm as	cm
import matplotlib
#import seaborn as sns; sns.set_style("darkgrid")
from IPython.display import	SVG, display
#import	sascorer
import itertools
from sklearn.metrics import	r2_score
import scipy

from sarpy import SARpy
import operator,pybel
import os
#import setting
from sarpy.SARpytools import *
import warnings
try:
    from rdkit.Chem import AllChem
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem import rdDepictor
    import cairosvg
except:
    warnings.warn('Cannot import rdkit or cairosvg, so report is not supported')
    Chem = None
    setting.report_substructures = False

from scipy import stats
import operator
from cairosvg import svg2png
from idl_ppb_.idl_ppb_modular import fragments, collectSubs, dataset


def eliminate_redundancy(frag_list,train_df_sort):
    Fragments = []
    for fragment in frag_list:
        elim_red_frag = {}
        elim_red_frag['Fragment'] = fragment
        patt = Chem.MolFromSmarts(fragment)
        Get_smiles_index = []
        for key in train_df_sort.cano_smiles:
            smiles = Chem.MolFromSmiles(key)
            frag = smiles.HasSubstructMatch(patt)
            if frag == True:
                Get_smiles_index.append(train_df_sort[train_df_sort.cano_smiles == key].index.tolist()[0])
            else:
                pass
        elim_red_frag['smiles_index'] = list(set(Get_smiles_index))
        Fragments.append(elim_red_frag)
#    print(Fragments)
    Fragments_del_list = []
    for i in range(len(Fragments)-1):
        for j in range(i+1,len(Fragments)):
            if Fragments[i]['smiles_index'] == Fragments[j]['smiles_index']:
                patt1 = Chem.MolFromSmarts(Fragments[i]['Fragment'])
                patt2 = Chem.MolFromSmarts(Fragments[j]['Fragment'])
                smiles1 = Chem.MolFromSmiles(Fragments[i]['Fragment'])
                smiles2 = Chem.MolFromSmiles(Fragments[j]['Fragment'])
                frag1 = smiles2.HasSubstructMatch(patt1)
                frag2 = smiles1.HasSubstructMatch(patt2)
                if frag1 == True:
                    if frag2 == False:
                        Fragments_del_list.append(Fragments[i]['Fragment'])
                if frag2 == True:
                    if frag1 == False:
                        Fragments_del_list.append(Fragments[j]['Fragment'])
                if frag1 == False:
                    if frag2 == False:
                        pass
    Final_Fragments_list = [f for f in frag_list if f not in Fragments_del_list]
    return Final_Fragments_list

def get_virtual_index(patt):
    i = 0
    virtual_index = []
    for atom in patt.GetAtoms():
        if atom.GetSymbol() != '*':
            virtual_index.append(i)
        i += 1
    return virtual_index


def get_R_virtual_index(patt):
    i = 0
    virtual_index = []
    for atom in patt.GetAtoms():
        if atom.GetSymbol() == '*':
            virtual_index.append(i)
        i += 1
    return virtual_index

def Find_second_level_substructures(SAs_list,train_df_sort):
    result = []
    for subs in SAs_list:
        subs_smiles_list = []
        non_toxicity_SAs = []
        non_toxicity_SAs_values = {}
        match_compounds_numb = 0
        patt = Chem.MolFromSmarts(subs)
        for key in train_df_sort.cano_smiles:
            smiles = Chem.MolFromSmiles(key)
            frag = smiles.HasSubstructMatch(patt)
            if frag == True:
                match_compounds_numb +=1
                subs_smiles_list.append(key)
        print(str(subs) + ' matches ' + str(match_compounds_numb)+' compounds')
        grinder = Grinder(3,18)
        subs_fragment = fragments(collectSubs(dataset(subs_smiles_list),grinder))
        print('Totally find ' + str(len(subs_fragment)) + ' fragments')
        for sub in subs_fragment:
            sub_patt = Chem.MolFromSmarts(sub)
            compounds_contain_activity = []
            compounds_no_contain_activity = []
            for smi in subs_smiles_list:
                smile = Chem.MolFromSmiles(smi)
                frags = smile.HasSubstructMatch(sub_patt)
                if frags == True:
                    activity = train_df_sort[train_df_sort.cano_smiles == smi].endpoint.values[0]
                    compounds_contain_activity.append(activity)
                else:
                    inactivity = train_df_sort[train_df_sort.cano_smiles == smi].endpoint.values[0]
                    compounds_no_contain_activity.append(inactivity)
#           print('compounds contain '  + str(sub)+ ' is ' + str(len(compounds_contain_activity)))
            if len(compounds_contain_activity) < 5:
                pass
            elif len(compounds_no_contain_activity) < 5:
                pass
            elif len(compounds_contain_activity)/len(compounds_no_contain_activity) > 4:
                pass
            elif len(compounds_no_contain_activity)/len(compounds_contain_activity) > 4:
                pass
            else:
#               P_value = stats.mannwhitneyu(compounds_contain_activity,compounds_no_contain_activity,alternative = 'less')[1]  #'less' non_toxic
                P_value = stats.mannwhitneyu(compounds_contain_activity,compounds_no_contain_activity)[1]
                if P_value < 0.01:
                    non_toxicity_SAs.append(sub)
                    va = np.mean(compounds_contain_activity) - np.mean(compounds_no_contain_activity)
                    non_toxicity_SAs_values[sub] = va
#                    draw_violin(compounds_contain_activity,compounds_no_contain_activity)
                    #print('None_toxicity_SAs: '    + str(sub) + ' number of compounds contain SAs & not : '+ str(len(compounds_contain_activity)) + ':' + str(len(compounds_no_contain_activity)))
                else:
                    pass
        # Inside Molecule analysis
        subs_smiles_list_df = pd.DataFrame(data = subs_smiles_list,columns = ['cano_smiles'])
        non_toxicity_SAs1 = eliminate_redundancy(non_toxicity_SAs,subs_smiles_list_df)
        print('For ' + str(subs)+' totally find '+ str(len(non_toxicity_SAs1))+ ' second-level substructures!')
        for subss in non_toxicity_SAs1:
            SAs_Non_SAs = {}
            SAs_Non_SAs['SA'] = subs
            SAs_Non_SAs['Non_SAs'] = subss
            SAs_Non_SAs['score'] = non_toxicity_SAs_values[subss]
            non_toxic_patt = Chem.MolFromSmarts(subss)
            no_SA_R_index = get_R_virtual_index(non_toxic_patt)
            stat1 = [[m] for m in range(len(no_SA_R_index))]
            stat2 = []
            RES = 0
            CES = 0
            NTS = 0
            ZES = 0
            for smiss in subs_smiles_list:
                ssmile = Chem.MolFromSmiles(smiss)
                ffrag = ssmile.HasSubstructMatch(non_toxic_patt)
                if ffrag == True:
                    atom_numbers_SAs = ssmile.GetSubstructMatches(patt)
                    atom_numbers_non_SAs = ssmile.GetSubstructMatches(non_toxic_patt)
                    SA_index = get_virtual_index(patt)
                    no_SA_index = get_virtual_index(non_toxic_patt)
                    for j in range(len(atom_numbers_non_SAs)):
                        no_SA_distance_index = [atom_numbers_non_SAs[j][n] for n in no_SA_index]
                        atom_index = [atom_numbers_non_SAs[j][m] for m in no_SA_R_index]
                        for n in range(len(atom_index)):
                            atom = ssmile.GetAtomWithIdx(atom_index[n]).GetSymbol()
                            stat1[n].append(atom)
                        for i in range(len(atom_numbers_SAs)):
                            SA_distance_index = [atom_numbers_SAs[i][m] for m in SA_index]
                            distance = Chem.GetDistanceMatrix(ssmile)
                            subs_distance = np.array(distance[SA_distance_index,:][:,no_SA_distance_index])
                            if subs_distance.min() == 0:
                                RES += 1
                            elif subs_distance.min() == 1:
                                CES += 1 
                            elif subs_distance.min() <= 3:
                                ZES += 1
                            else:
                                NTS += 1
            SAs_Non_SAs['RES'] = RES
            SAs_Non_SAs['CES'] = CES
            SAs_Non_SAs['NTS'] = NTS
            SAs_Non_SAs['ZES'] = ZES
            for x in range(len(stat1)):
                stat3 = {}
                for y in stat1[x]:
                    stat3[y] = stat1[x].count(y)
                stat2.append(stat3)
            SAs_Non_SAs['R_group'] = stat2
            result.append(SAs_Non_SAs)
    return result


target = 'ppb_3922'
df = pd.read_csv(target+'.csv')
molset = [Chem.MolFromSmiles(smi) for smi in df.cano_smiles]
train_df = df[df.set_split == 'train'].drop('set_split',axis=1).reset_index(drop= True)
smi_list = [smi for smi in train_df.cano_smiles]
train_df_sort = train_df.sort_values(by = 'endpoint',ascending = False).reset_index(drop=True)


#SA = ['CN1CC(*)C1']
SA=['']
r = Find_second_level_substructures(SA,train_df_sort)
#data={'SA': 'CN1CC(*)C1', 'Non_SAs': '*c3ccccc3', 'score': 0.14786313364055315, 'RES': 0, 'CES': 23, 'NTS': 131, 'ZES': 41, 'R_group': [{0: 1, 'C': 80, 'O': 33, 'S': 5, 'N': 17, 'F': 10, 'Cl': 13, 'I': 2}]}
data_frame= pd.DataFrame()
for dict_r in r:
    data_frame=data_frame.append(dict_r, ignore_index=True)

print(data_frame) 

data_frame.to_csv("results.csv",index= False)


'''with open('Results.smi','w') as f:
    writer = csv.writer(f)
    writer.writerow(["Sub-2level"]) # header
    for o in r:
        writer.writerow([o])'''
        
'''f.write('SA_Fragment\tNAS\tScore\tRES\tCES\tZES\tNTS\n')
    for	i in range(len(r)):
        f.writerow
        f.write(str(r[i]['SA'])+'\t'+str(r[i]['Non_SAs'])+'\t'+str(r[i]['score'])+'\t'+str(r[i]['RES'])+'\t'+str(r[i]['CES'])+'\t'+str(r[i]['ZES'])+'\t'+str(r[i]['NTS']))+'\n'
    f.close()'''