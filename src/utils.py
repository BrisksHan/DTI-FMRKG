def save_to_pickle(data, file_path):
    import pickle
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
        print("Data saved successfully.")
    except Exception as e:
        raise Exception(f"An error occurred while saving data: {str(e)}")

def load_from_pickle(file_path):
    import pickle
    loaded_data = None
    try:
        with open(file_path, 'rb') as file:
            loaded_data = pickle.load(file)
        print("Data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred while loading data: {str(e)}")
    return loaded_data

def read_text_file(file_name):
    try:
        with open(file_name, 'r') as file:
            file_contents = file.readlines()
            return file_contents
    except FileNotFoundError:
        raise Exception(f'can not read {file_name}')
    
def obtain_RDFs(file_contents):
    RDFs = []

    for item in file_contents:
        new_items = item.split('\t')
        RDFs.append([new_items[0],new_items[1],new_items[2][0:-1]])

    return RDFs

def save_txt(contents, file_name):
    with open(file_name, 'w') as file:
        for a_line in contents:
            file.write(a_line + '\n')
    print('save succesuffly')
    
def save_csv(contents, file_name):
    import csv
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(contents)
    print('save succesuffly')
    
def transpose_lists(original_list):
    transposed_list = list(zip(*original_list))
    return [list(item) for item in transposed_list]

if __name__ == "__main__":
    pass
    #found_file = read_text_file('dataset/Yamanishi08/kegg_kg.txt')
    
    #print(found_file)
    #read_Yamanishi08_dataset()
    #T1_P, T1_N = read_warm_1_1_split_Train(5)
    #print(len(T1_P),'   ',len(T1_N))