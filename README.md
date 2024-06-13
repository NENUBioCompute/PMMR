# PMMR
PMMR is a tool for drug-target binding prediction.

All benchmark datasets can be found at ./data/. Data preprocessing can be referred to ./preprocessing/.

## Requirements:
python 3.7<br>
cudatoolkit 10.1.243<br>
cudnn 7.6.0<br>
pytorch 1.4.0<br>
numpy 1.16.4<br>
scikit-learn 0.21.2<br>
pandas 0.24.2<br>
scipy 1.3.0<br>
numba 0.44.1<br>
rdkit 2023.09.2<br>
esm 2.0.0<br>
transformers 4.35.2<br>

## Data preprocessing
### Regression
Before running, please generate pre-trained features of proteins and drugs. If you want to perform affinity prediction tasks, taking the PDBbind data set as an example, you need to switch to the preprocessing/protein_pretrain folder and execute the following command to generate protein features:
```
python protein_dta.py --dataset pdb
```
Then switch to the preprocessing folder and execute the following command:
```
python compound_pretrain.py --dataset pdb
```
### Classification
And if you want to perform classification task, please switch to the preprocessing/protein_pretrain folder and execute the following command to generate protein features:
```
python protein_dti.py --dataset bindingdb
```
Then switch to the preprocessing folder and execute the following command:
```
python compound_pretrain.py --dataset bindingdb
```
## Training & Evaluation
To train and test the affinity prediction task based on the dataset we provide, please switch to the main folder and execute the command:
```
python main.py --objective regression --dataset pdb --batch_size 128 --max_epochs 200 --learning_rate 0.001
```
To train and test classification tasks, please execute the following code:
```
python main.py --objective classification --dataset bindingdb --batch_size 64 --max_epochs 500 --learning_rate 0.001
```
