import pickle
# from deepchem.feat import SmilesTokenizer
from transformers import BertTokenizer, AutoTokenizer,BertModel,RobertaModel,RobertaTokenizer
import argparse
import os
import os.path
import numpy as np
import pandas as pd
import torch

def get_smiles_embedding(smiles):
    # vocab_path = './vocab.txt'  # path to vocab.txt
    # tokenizer_ = SmilesTokenizer(vocab_path)
    tokenizer_ = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
    model_ = RobertaModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
    model_.eval()
    with torch.no_grad():
        outputs_ = model_(**tokenizer_(smiles, return_tensors='pt'))
    return outputs_.last_hidden_state[0][1:outputs_.last_hidden_state.shape[1]-1]

def generate_feature(args):


    data_path = args.root_data_path
    dataset = args.dataset
    output_data_path = data_path + '/' + dataset + '/compound/'

    if not os.path.exists(output_data_path):
        # 如果文件夹不存在，则创建
        os.makedirs(output_data_path)
        print(f"{output_data_path} created")
    else:
        print(f"{output_data_path} exists")

    dict_list = {}
    count = 0

    opts = ['train', 'test']
    for o in opts:
        if dataset == 'davis':
            raw_data = pd.read_csv(f'{data_path}/{dataset}.csv')
        else:
            raw_data = pd.read_csv(f'{data_path}/{dataset}_{o}.csv')

        smiles_values = raw_data['compound_iso_smiles'].values
        for s in smiles_values:
          if s in dict_list.keys():
              continue
          smiles_embedding = get_smiles_embedding(s)
          dict_list[s] = smiles_embedding
          count += 1

        with open(f'{output_data_path}/mol_dict.pkl', 'wb') as file:
            pickle.dump(dict_list, file)

    print(count)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_data_path', type=str, default='../data', help='Raw Data Path')
    parser.add_argument('--dataset', type=str, default='davis', help='Datasets')
    return parser.parse_args()

if __name__ == '__main__':
    params = parse_args()
    print(params)
    generate_feature(params)




