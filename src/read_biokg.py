from utils import read_text_file
import pandas as pd


def read_biokg_kg():
    file_name = 'dataset/BioKG/kg.csv'

    kg_content = read_text_file(file_name)

    del kg_content[0]
    RDFs = []
    for item in kg_content:
        #print(item)
        cur_RDF = item.split(',')
        if cur_RDF[3][-1] == '\n':
            new_RDF = [cur_RDF[1], cur_RDF[2], cur_RDF[3][0:-1]]#other wise \n will appear
        else:
            new_RDF = [cur_RDF[1], cur_RDF[2], cur_RDF[3]]
        RDFs.append(new_RDF)
    return RDFs

def read_warm_1_10_split_Test(split_id = 1):
    file = f"dataset/BioKG/data_folds/warm_start_1_10/test_fold_{split_id}.csv"
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
            raise Exception('read_warm_1_10_split_Test label error')
    return DTI_Positive, DTI_Negative

def read_warm_1_10_split_Train(split_id = 1):
    file = f"dataset/BioKG/data_folds/warm_start_1_10/train_fold_{split_id}.csv"
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

def read_biokg_drug_cold_start_train(split_id = 1):
    file = f"dataset/BioKG/data_folds/drug_coldstart/train_fold_{split_id}.csv"
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

def read_biokg_drug_cold_start_test(split_id = 1):
    file = f"dataset/BioKG/data_folds/drug_coldstart/test_fold_{split_id}.csv"
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


def read_biokg_target_cold_start_train(split_id = 1):
    file = f"dataset/BioKG/data_folds/protein_coldstart/train_fold_{split_id}.csv"
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

def read_biokg_target_cold_start_test(split_id = 1):
    file = f"dataset/BioKG/data_folds/protein_coldstart/test_fold_{split_id}.csv"
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

def read_biokg_IDs_features():
    fp_df = pd.read_csv('dataset/BioKG/fp_df.csv', header = 0)
    drug_ids = fp_df['comp_id']
    del fp_df['comp_id']
    drug_features = fp_df.values
    #print(fp_df)
    prodes_df = pd.read_csv('dataset/BioKG/prodes_df.csv', header = 0)
    protain_ids = prodes_df['pro_ids']
    del prodes_df['pro_ids']
    protain_feaures = prodes_df.values
    #print(prodes_df)
    return drug_ids, drug_features, protain_ids, protain_feaures

def get_biokg_warm_start_1_10(split_id):
    DTI_Positive, DTI_Negative = read_warm_1_10_split_Train(split_id = split_id)
    other_triples = read_biokg_kg()
    Training_triples = DTI_Positive + other_triples
    return Training_triples, DTI_Positive, DTI_Negative

def get_biokg_drug_cold_start(split_id):
    DTI_Positive, DTI_Negative = read_biokg_drug_cold_start_train(split_id = split_id)
    other_triples = read_biokg_kg()
    Training_triples = DTI_Positive + other_triples
    return Training_triples, DTI_Positive, DTI_Negative

def get_biokg_target_cold_start(split_id):
    DTI_Positive, DTI_Negative = read_biokg_target_cold_start_train(split_id = split_id)
    other_triples = read_biokg_kg()
    Training_triples = DTI_Positive + other_triples
    return Training_triples, DTI_Positive, DTI_Negative

def get_biokg_drug_smiles():
    drug_id = pd.read_csv('./dataset/BioKG/comp_struc.csv')['head']
    drug_smiles = pd.read_csv('./dataset/BioKG/comp_struc.csv')['smiles']
    return drug_id, drug_smiles

def get_biokg_target_sequences():
    target_id = pd.read_csv('./dataset/BioKG/pro_seq.csv')['pro_id']
    target_sequences = pd.read_csv('./dataset/BioKG/pro_seq.csv')['Sequence']
    return target_id, target_sequences