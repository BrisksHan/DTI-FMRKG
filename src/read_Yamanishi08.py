from utils import obtain_RDFs
from utils import read_text_file
import pandas as pd
import numpy as np

def read_Yamanishi08_dataset():
    file_name1 = 'dataset/yamanishi08/kg_data/kegg_kg.txt'
    file_name2 = 'dataset/yamanishi08/kg_data/yamanishi_uniprot_kg.txt'

    kegg_kg_content = read_text_file(file_name1)
    yamanishi_kg_content = read_text_file(file_name2)

    RDFs = kegg_kg_content + yamanishi_kg_content
    RDFs = list(set(RDFs))

    RDFs = obtain_RDFs(RDFs)
    return RDFs

def read_warm_1_1_split_Train(split_id = 1):#1~10
    file = f"dataset/yamanishi08/data_folds/warm_start_1_1/train_fold_{split_id}.csv"
    raw_info = read_text_file(file)
    del raw_info[0]
    DTI_Positive = []
    DTI_Negative = []
    for item in raw_info:
        cur_RDF = item.split(',')
        cur_RDF[-1] = int(float(cur_RDF[-1]))
        if cur_RDF[-1] == 1:
            DTI_Positive.append(cur_RDF[0:3])
        elif cur_RDF[-1] == 0:
            DTI_Negative.append(cur_RDF[0:3])
        else:
            raise Exception('read_warm_1_1_split_Train error')
    return DTI_Positive, DTI_Negative

def read_warm_1_1_split_Test(split_id = 1):
    file = f"dataset/yamanishi08/data_folds/warm_start_1_1/test_fold_{split_id}.csv"
    raw_info = read_text_file(file)
    del raw_info[0]
    DTI_Positive = []
    DTI_Negative = []
    for item in raw_info:
        cur_RDF = item.split(',')
        cur_RDF[-1] = int(float(cur_RDF[-1]))
        if cur_RDF[-1] == 1:
            DTI_Positive.append(cur_RDF[0:3])
        elif cur_RDF[-1] == 0:
            DTI_Negative.append(cur_RDF[0:3])
        else:
            raise Exception('read_warm_1_1_split_Test error')
    return DTI_Positive, DTI_Negative

def read_warm_1_10_split_Train(split_id = 1):
    file = f"dataset/yamanishi08/data_folds/warm_start_1_10/train_fold_{split_id}.csv"
    raw_info = read_text_file(file)
    del raw_info[0]
    DTI_Positive = []
    DTI_Negative = []
    for item in raw_info:
        cur_RDF = item.split(',')
        cur_RDF[-1] = int(float(cur_RDF[-1]))
        if cur_RDF[-1] == 1:
            DTI_Positive.append(cur_RDF[0:3])
        else:
            DTI_Negative.append(cur_RDF[0:3])
    return DTI_Positive, DTI_Negative

def read_warm_1_10_split_Test(split_id = 1):
    file = f"dataset/yamanishi08/data_folds/warm_start_1_10/test_fold_{split_id}.csv"
    raw_info = read_text_file(file)
    del raw_info[0]
    DTI_Positive = []
    DTI_Negative = []
    for item in raw_info:
        cur_RDF = item.split(',')
        cur_RDF[-2] = int(float(cur_RDF[-2]))
        if cur_RDF[-2] == 1:
            DTI_Positive.append(cur_RDF[0:3])
        elif cur_RDF[-2] == 0:
            DTI_Negative.append(cur_RDF[0:3])
        else:
            raise Exception('read_warm_1_10_split_Test label error')
    return DTI_Positive, DTI_Negative


def get_training_Yamanishi08_warm_start_1_1(split_id = 1):
    DTI_Positive, DTI_Negative = read_warm_1_1_split_Train(split_id = split_id)
    other_triples = read_Yamanishi08_dataset()
    Training_triples = DTI_Positive+other_triples
    return Training_triples, DTI_Positive, DTI_Negative

def get_training_Yamanishi08_warm_start_1_10(split_id = 1):
    DTI_Positive, DTI_Negative = read_warm_1_10_split_Train(split_id = split_id)
    other_triples = read_Yamanishi08_dataset()
    Training_triples = DTI_Positive+other_triples
    return Training_triples, DTI_Positive, DTI_Negative

def read_drug_cold_1_10_split_Train(split_id = 1):
    file = f"dataset/yamanishi08/data_folds/drug_coldstart/train_fold_{split_id}.csv"
    raw_info = read_text_file(file)
    del raw_info[0]
    DTI_Positive = []
    DTI_Negative = []
    for item in raw_info:
        cur_RDF = item.split(',')
        cur_RDF[-1] = int(float(cur_RDF[-1]))
        if cur_RDF[-1] == 1:
            DTI_Positive.append(cur_RDF[0:3])
        else:
            DTI_Negative.append(cur_RDF[0:3])
    return DTI_Positive, DTI_Negative

def read_drug_cold_1_10_split_Test(split_id = 1):
    file = f"dataset/yamanishi08/data_folds/drug_coldstart/test_fold_{split_id}.csv"
    raw_info = read_text_file(file)
    del raw_info[0]
    DTI_Positive = []
    DTI_Negative = []
    for item in raw_info:
        cur_RDF = item.split(',')
        cur_RDF[-2] = int(float(cur_RDF[-2]))
        if cur_RDF[-2] == 1:
            DTI_Positive.append(cur_RDF[0:3])
        elif cur_RDF[-2] == 0:
            DTI_Negative.append(cur_RDF[0:3])
        else:
            raise Exception('read_drug_coldstart_Test label error')
    return DTI_Positive, DTI_Negative


def get_training_Yamanishi08_drug_cold_start(split_id = 1):
    DTI_Positive, DTI_Negative = read_drug_cold_1_10_split_Train(split_id = split_id)
    other_triples = read_Yamanishi08_dataset()
    Training_triples = DTI_Positive+other_triples
    return Training_triples, DTI_Positive, DTI_Negative

def read_target_cold_1_10_split_Train(split_id = 1):
    file = f"dataset/yamanishi08/data_folds/protein_coldstart/train_fold_{split_id}.csv"
    raw_info = read_text_file(file)
    del raw_info[0]
    DTI_Positive = []
    DTI_Negative = []
    for item in raw_info:
        cur_RDF = item.split(',')
        cur_RDF[-1] = int(float(cur_RDF[-1]))
        if cur_RDF[-1] == 1:
            DTI_Positive.append(cur_RDF[0:3])
        else:
            DTI_Negative.append(cur_RDF[0:3])
    return DTI_Positive, DTI_Negative

def read_target_cold_1_10_split_Test(split_id = 1):
    file = f"dataset/yamanishi08/data_folds/protein_coldstart/test_fold_{split_id}.csv"
    raw_info = read_text_file(file)
    del raw_info[0]
    DTI_Positive = []
    DTI_Negative = []
    for item in raw_info:
        cur_RDF = item.split(',')
        cur_RDF[-2] = int(float(cur_RDF[-2]))
        if cur_RDF[-2] == 1:
            DTI_Positive.append(cur_RDF[0:3])
        elif cur_RDF[-2] == 0:
            DTI_Negative.append(cur_RDF[0:3])
        else:
            raise Exception('read_drug_coldstart_Test label error')
    return DTI_Positive, DTI_Negative

def get_training_Yamanishi08_target_cold_start(split_id = 1):
    DTI_Positive, DTI_Negative = read_target_cold_1_10_split_Train(split_id = split_id)
    other_triples = read_Yamanishi08_dataset()
    Training_triples = DTI_Positive+other_triples
    return Training_triples, DTI_Positive, DTI_Negative

def get_yamanishi08_drug_protain_IDs():
    drug_id = pd.read_csv('./dataset/yamanishi08/791drug_struc.csv')['drug_id']
    protain_id = pd.read_csv('./dataset/yamanishi08/989proseq.csv')['pro_ids']
    return drug_id, protain_id

def get_yamanishi08_protain_drug_features():
    drug_features = np.loadtxt('dataset/yamanishi08/morganfp.txt',delimiter=',')
    protain_features = np.loadtxt('dataset/yamanishi08/pro_ctd.txt',delimiter=',')
    return drug_features, protain_features

def get_yamanishi08_drug_smiles():
    drug_id = pd.read_csv('./dataset/yamanishi08/791drug_struc.csv')['drug_id']
    drug_smiles = pd.read_csv('./dataset/yamanishi08/791drug_struc.csv')['smiles']
    return drug_id, drug_smiles

def get_yamanishi08_target_squences():
    target_id = pd.read_csv('./dataset/yamanishi08/989proseq.csv')['pro_ids']
    target_sequences = pd.read_csv('./dataset/yamanishi08/989proseq.csv')['seq']
    return target_id, target_sequences
