import argparse
import read_Yamanishi08
import read_biokg
import utils
import eval
import prediction_model
import structural_encoding
import KG_embedding

#--------------------------------------------main function------------------------------------------------------------

def get_dataset(dataset_name = 'yamanishi08_1_10', split = 1):
    print('the selcted dataset: ',dataset_name)
    if dataset_name == 'yamanishi08_1_1':
        Training_Triples, Training_DTI_Positive, Training_DTI_Negative = read_Yamanishi08.get_training_Yamanishi08_warm_start_1_1(split)
        Test_DTI_Positive, Test_DTI_Negative = read_Yamanishi08.read_warm_1_1_split_Test(split)
    elif dataset_name == 'yamanishi08_1_10':
        Training_Triples, Training_DTI_Positive, Training_DTI_Negative = read_Yamanishi08.get_training_Yamanishi08_warm_start_1_10(split)
        Test_DTI_Positive, Test_DTI_Negative = read_Yamanishi08.read_warm_1_10_split_Test(split)
    elif dataset_name == 'yamanishi08_drug_cold':
        Training_Triples, Training_DTI_Positive, Training_DTI_Negative = read_Yamanishi08.get_training_Yamanishi08_drug_cold_start(split)
        Test_DTI_Positive, Test_DTI_Negative = read_Yamanishi08.read_drug_cold_1_10_split_Test(split)
    elif dataset_name == 'yamanishi08_target_cold':
        Training_Triples, Training_DTI_Positive, Training_DTI_Negative = read_Yamanishi08.get_training_Yamanishi08_target_cold_start(split)
        Test_DTI_Positive, Test_DTI_Negative = read_Yamanishi08.read_target_cold_1_10_split_Test(split)
    elif dataset_name == 'BioKG_1_10':
        Training_Triples, Training_DTI_Positive, Training_DTI_Negative = read_biokg.get_biokg_warm_start_1_10(split)
        Test_DTI_Positive, Test_DTI_Negative = read_biokg.read_warm_1_10_split_Test(split)
    elif dataset_name == 'BioKG_drug_cold':
        Training_Triples, Training_DTI_Positive, Training_DTI_Negative = read_biokg.get_biokg_drug_cold_start(split)
        Test_DTI_Positive, Test_DTI_Negative = read_biokg.read_biokg_drug_cold_start_test(split)
    elif dataset_name == 'BioKG_target_cold':
        Training_Triples, Training_DTI_Positive, Training_DTI_Negative = read_biokg.get_biokg_target_cold_start(split)
        Test_DTI_Positive, Test_DTI_Negative = read_biokg.read_biokg_target_cold_start_test(split)

        
    else:
        raise Exception('dataset not exisits, please input yamanishi08_1_1 or yamanishi08_1_10 or yamanishi08_drug_cold or yamanishi08_target_cold or the corresponding setting in BioKG')
    return Training_Triples, Training_DTI_Positive, Training_DTI_Negative, Test_DTI_Positive, Test_DTI_Negative


def construct_sentence(option):
    if str(option) == 1:
        pass
    elif str(option) == 2:
        pass
    else:
        raise Exception('please input 1 or 2 for sentence construction for mlm')

def eval_performance(labels, scores, save_result = False, path = ''):
    AUPR_result = eval.eval_AUPR(labels, scores)
    AUROC_result = eval.eval_AUROC(labels, scores)
    [f'AUPR:{AUPR_result}',f'AUROC:{AUROC_result}']
    if save_result and path:
        utils.save_txt(AUPR_result, path)
        print('save result successful')
    else:
        print('save result failed')

def check_bool_options(option):
    if option not in ['True', 'true', 'False', 'false']:
        raise Exception('Please input True or False for use_finetuned_bert, use_previous_corpus, use_previous_prediction and scale_embedding')
    elif option == 'True' or option == 'true':
        return True
    else:
        return False

def parse_args():
    parser = argparse.ArgumentParser(description='A script that demonstrates command-line argument parsing.')

    # Define command-line arguments
    parser.add_argument('--use_previous_kg', type = str, default = 'True',help = 'use the pretrained kg embedding')

    parser.add_argument('--kg_dim', type=int, default= 400, help='the embedding dimension of kg entity')

    parser.add_argument('--dataset', type=str, default = 'yamanishi08_1_1', help='select task name, such as yamanishi08_1_1, yamanishi08_1_10, yamanishi08_drug_cold, yamanishi08_target_cold, BioKG_1_10, BioKG_drug_cold, BioKG_target_cold')

    parser.add_argument('--split', type=int, default = 1, help= 'the fold of the experiment')

    parser.add_argument('--use_previous_prediction', type = str, default = 'False', help= 'use the previous prediction model')

    parser.add_argument('--batch_size', type = int, default = 100, help ='the batch size for traububf deeo kearbubf')

    parser.add_argument('--kg_training', type = str, default= 'cuda:0')

    parser.add_argument('--gpu_inference', type = str, default = 'cuda:0')

    parser.add_argument('--gpu_training', type = str, default = 'cuda:0')

    parser.add_argument('--kg_epoch', type =int, default = 100)

    parser.add_argument('--kg_method', type = str, default = 'TransD')

    args = parser.parse_args()

    args.use_previous_kg = check_bool_options(args.use_previous_kg)
    args.use_previous_prediction = check_bool_options(args.use_previous_prediction)
    #args.scale_embedding = check_bool_options(args.scale_embedding)    

    return args

def main(args):
    
    print('start running experiment on DTI')
    args_dict = vars(args)

    # Iterate through the key-value pairs
    for key, value in args_dict.items():
        print(f"{key}: {value}")
        # get the datasets

    if args.split > 10 or args.split < 1:
        raise Exception('split must be between 1~10')

    #--------------------------------------------create path for the current task---------------------------------------

    Training_Triples, Training_DTI_Positive, Training_DTI_Negative, Test_DTI_Positive, Test_DTI_Negative = get_dataset(args.dataset, args.split)
    
    Training_labels = [1]*len(Training_DTI_Positive) + [0]*len(Training_DTI_Negative)

    Test_labels = [1]*len(Test_DTI_Positive) + [0]*len(Test_DTI_Negative)

    print(f'current dataset: {args.dataset}')

    print(f'training_triples num:{len(Training_Triples)}, training DTI positive:{len(Training_DTI_Positive)}, training DTI negative:{len(Training_DTI_Negative)}')

    print(f'test DTI positive:{len(Test_DTI_Positive)}, test DTI negative:{len(Test_DTI_Negative)}')

    prediction_path = 'trained_model/'+ args.dataset + f'_{args.split}_' + f'_add_mix_CNN_prediction_model_{args.kg_method}_s2.pth'

    kg_embedding_path = 'trained_model/' + args.dataset + f'_{args.split}_' + f'data_kg_embedding_d{args.kg_dim}_e{args.kg_epoch}_{args.kg_method}.pkl'

    result_save_path = 'results/' + args.dataset + f'_{args.split}_' + f'_result_add_CNNs_d{args.kg_dim}_{args.kg_method}_{args.kg_epoch}_dual.txt'

    print(prediction_path)
    print(result_save_path)
    print(kg_embedding_path)

    #--------------------------------------------protain_features PCA and drug feature------------------------------------------------------

    if args.dataset == 'yamanishi08_1_10' or args.dataset == 'yamanishi08_1_1' or args.dataset == 'yamanishi08_drug_cold' or args.dataset == 'yamanishi08_target_cold':
        drug_IDs, drug_smiles = read_Yamanishi08.get_yamanishi08_drug_smiles()
        target_IDs, target_seqeunces = read_Yamanishi08.get_yamanishi08_target_squences()
    elif args.dataset == 'BioKG_1_10' or args.dataset == 'BioKG_drug_cold' or args.dataset == 'BioKG_target_cold' :
        drug_IDs, drug_smiles = read_biokg.get_biokg_drug_smiles()
        target_IDs, target_seqeunces = read_biokg.get_biokg_target_sequences()
    else:
        raise Exception('dataset not exisits, please input yamanishi08_1_1 or yamanishi08_1_10 or yamanishi08_drug_cold or yamanishi08_target_cold or the corresping task in  BioKG')


    drug_IDs = drug_IDs.tolist()
    target_IDs = target_IDs.tolist()

    if args.use_previous_kg:
        try:
            print('use the previous kg embedding')
            KG_embedding_dict = utils.load_from_pickle(kg_embedding_path)
            entity_embedding_np, entity_embedding_dict, relation_embedding_np, relation_embedding_dict = KG_embedding_dict
        except:
            print('the previous kg embedding is not exist, start training a new one')
            entity_embedding_np, entity_embedding_dict, relation_embedding_np, relation_embedding_dict = KG_embedding.get_KG_embedding_dict(Training_Triples, model = args.kg_method, epochs = args.kg_epoch, save_path = kg_embedding_path, embedding_dim = args.kg_dim, device = args.kg_training)
    else:
        print('start training a new kg embedding')
        entity_embedding_np, entity_embedding_dict, relation_embedding_np, relation_embedding_dict = KG_embedding.get_KG_embedding_dict(Training_Triples, model = args.kg_method,  epochs = args.kg_epoch, save_path = kg_embedding_path, embedding_dim = args.kg_dim, device = args.kg_training)

    train_drug_smiles_str = []
    train_drug_embedding = []
    train_drug_smiles = []
    train_target_embedding = []
    train_target_sequences = []


    #print('target_sequence[0]:',target_seqeunces[0])

    for index, item in enumerate(Training_DTI_Positive + Training_DTI_Negative):#DTI hr rt
        cur_drug = item[0]
        cur_target = item[2]
        train_drug_embedding.append(entity_embedding_np[entity_embedding_dict[cur_drug]])
        train_target_embedding.append(entity_embedding_np[entity_embedding_dict[cur_target]])
        cur_drug_index = drug_IDs.index(cur_drug)
        cur_target_index = target_IDs.index(cur_target)
        train_drug_smiles_str.append(drug_smiles[cur_drug_index])
        train_target_sequences.append(structural_encoding.sequence_encoding(target_seqeunces[cur_target_index]))
        train_drug_smiles.append(structural_encoding.smile_encoding(drug_smiles[cur_drug_index]))
    
    print('pos+neg:',len(Training_DTI_Positive + Training_DTI_Negative))
    print('training_embedding:', len(train_drug_embedding[0]),' ',len(train_target_embedding[0]))
    print('training label:',len(Training_labels))

    Train_att = []

    for i in range(len(train_target_sequences)):
        cur = [train_drug_smiles_str[i], train_drug_smiles[i], train_drug_embedding[i], train_target_sequences[i], train_target_embedding[i], Training_labels[i]]
        #print(train_drug_smiles_str[i][0:100])
        Train_att.append(cur)

    #utils.save_to_pickle(Train_att, 'data_temp/train_v11.pkl')

    test_drug_smiles_str = []
    test_drug_embedding = []
    test_drug_smiles = []
    test_target_embedding = []
    test_target_sequences = []


    for index, item in enumerate(Test_DTI_Positive + Test_DTI_Negative):
        cur_drug = item[0]
        cur_target = item[2]

        test_drug_embedding.append(entity_embedding_np[entity_embedding_dict[cur_drug]])
        test_target_embedding.append(entity_embedding_np[entity_embedding_dict[cur_target]])
        cur_drug_index = drug_IDs.index(cur_drug)
        cur_target_index = target_IDs.index(cur_target)
        test_drug_smiles_str.append(drug_smiles[cur_drug_index])
        test_target_sequences.append(structural_encoding.sequence_encoding(target_seqeunces[cur_target_index]))
        test_drug_smiles.append(structural_encoding.smile_encoding(drug_smiles[cur_drug_index]))

    Test_att = []

    for i in range(len(test_target_sequences)):
        cur = [test_drug_smiles_str[i], test_drug_smiles[i], test_drug_embedding[i], test_target_sequences[i], test_target_embedding[i]]
        Test_att.append(cur)

    #--------------------------------------------train_prediction model----------------------------------------------------

    
    if not args.use_previous_prediction:
        print('start training a prediction model')
        lr = 0.001 # prev
        #lr = 0.0001 worked also well   0.92
        #lr = 0.00001 worked well
        #
        prediction_model.Train_Prediction_Model(Train_att, kg_dim = args.kg_dim , save_path = prediction_path, device_ids = args.gpu_training, batch_size = 384, lr = 0.0001)
    else:
        print('use the previous prediction model')#1540

    #--------------------------------------------train_prediction model-----------------------------------------------------

    print('predict test samples')

    Test_prediction = prediction_model.Attention_prediction(Test_att, kg_dim = args.kg_dim, load_path = prediction_path, device = args.gpu_training)

    #--------------------------------------------evaluate model-------------------------------------------------------------
    
    print('start evaluate results')

    results = eval.all_eval(Test_labels, Test_prediction)

    print('eval result')

    utils.save_txt(results, result_save_path)
    #---------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main(parse_args())