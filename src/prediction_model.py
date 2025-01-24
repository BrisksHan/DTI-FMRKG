import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from torch.nn.parallel import DataParallel

from torch_geometric.data import Data, Dataset, InMemoryDataset

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import networkx as nx

from torch_geometric.nn import GATConv, GINConv, LEConv, FAConv, global_max_pool, GeneralConv, GENConv, SAGEConv
import torch.nn.functional as F
#v1 is for two seperate attention mechanism
#v2 is for a unified attention mechanism

#---------------------------------------------------model------------------------------------------------------------------

class mutil_head_attention(nn.Module):
    def __init__(self,head = 8,conv = 32):
        super(mutil_head_attention,self).__init__()
        self.conv = conv
        self.head = head
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.drug_a = nn.Linear(self.conv * 3, self.conv * 3 * head)
        self.protein_a = nn.Linear(self.conv * 3, self.conv * 3 * head)
        self.scale = nn.Parameter(torch.sqrt(torch.FloatTensor([self.conv * 3])), requires_grad = False)

    def forward(self, drug, protein):
        batch_size, drug_c, drug_l = drug.shape
        batch_size, protein_c, protein_l = protein.shape
        drug_att = self.relu(self.drug_a(drug.permute(0, 2, 1))).view(batch_size,self.head,drug_l,drug_c)
        protein_att = self.relu(self.protein_a(protein.permute(0, 2, 1))).view(batch_size,self.head,protein_l,protein_c)
        interaction_map = torch.mean(self.tanh(torch.matmul(drug_att, protein_att.permute(0, 1, 3, 2)) / self.scale),1)
        Compound_atte = self.tanh(torch.sum(interaction_map, 2)).unsqueeze(1)
        Protein_atte = self.tanh(torch.sum(interaction_map, 1)).unsqueeze(1)
        drug = drug * Compound_atte
        protein = protein * Protein_atte
        return drug, protein

class CNNs_Attention(nn.Module):
    def __init__(self, protein_MAX_LENGH = 1200, protein_kernel = [4,8,12],
            drug_MAX_LENGH = 100, drug_kernel = [4,6,8], kg_dim = 400,
            conv = 64, char_dim = 128,head_num = 8,dropout_rate = 0.1, embed_dim = 192, device = 'cuda:0'):
        super(CNNs_Attention, self).__init__()
        self.protein_kernel = protein_kernel
        self.drug_kernel = drug_kernel
        self.device = device
        self.dim = char_dim
        #self.drug_padding_size = 
        self.conv = conv
        self.dropout_rate = dropout_rate
        self.head_num = head_num
        self.drug_MAX_LENGH = drug_MAX_LENGH
        self.protein_MAX_LENGH = protein_MAX_LENGH
        self.kg_dim = kg_dim

        self.protein_embed = nn.Embedding(26, self.dim,padding_idx=0)
        self.drug_embed = nn.Embedding(65, self.dim,padding_idx=0)

        self.Drug_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels= self.conv,  kernel_size=self.drug_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels= self.conv*2,  kernel_size=self.drug_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv*2, out_channels= self.conv*3,  kernel_size=self.drug_kernel[2]),
            nn.ReLU(),
        )

        #self.Drug_max_pool_0 = nn.MaxPool1d(self.drug_MAX_LENGH-self.drug_kernel[0]-self.drug_kernel[1]-self.drug_kernel[2]+3)
        self.Drug_max_pool_0_cnn = nn.MaxPool1d(85)
        self.Drug_max_pool_1_cnn = nn.AvgPool1d(192)

        self.Drug_max_pool_0_gnn = nn.MaxPool1d(100)
        self.Drug_max_pool_1_gnn = nn.AvgPool1d(192)

        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv, kernel_size=self.protein_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2, kernel_size=self.protein_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 3, kernel_size=self.protein_kernel[2]),
            nn.ReLU(),
        )
        self.Protein_max_pool_0_cnn = nn.MaxPool1d(self.protein_MAX_LENGH - self.protein_kernel[0] - self.protein_kernel[1] - self.protein_kernel[2] + 3)
        self.Protein_max_pool_1_cnn = nn.AvgPool1d(192)

        self.Protein_max_pool_0_gnn = nn.MaxPool1d(self.protein_MAX_LENGH - self.protein_kernel[0] - self.protein_kernel[1] - self.protein_kernel[2] + 3)
        self.Protein_max_pool_1_gnn = nn.AvgPool1d(192)


        self.attention_cnn = mutil_head_attention(head = self.head_num, conv=self.conv)
        self.attention_gnn = mutil_head_attention(head = self.head_num, conv=self.conv)
        #self.attention_kg = multi_head_attention_kg(head = self.head_num)

        #self.x_encoder_1 = GINConv(110, 96, heads=head_num, dropout=dropout_rate)
        #self.x_encoder_2 = GINConv(96 * head_num, 128, heads=head_num, dropout=dropout_rate)
        #self.x_encoder_3 = GINConv(128 * head_num, 128, heads=head_num, dropout=dropout_rate)
        #self.x_encoder_4 = GINConv(128 * head_num, embed_dim, dropout=dropout_rate)

        #self.x_encoder_1 = GATConv(110, 192)
        #self.x_encoder_2 = GATConv(192, embed_dim)
        #self.x_encoder_3 = GATConv(192, 192)
        #self.x_encoder_4 = GATConv(192, embed_dim)

        #self.x_encoder_5 = nn.Linear(embed_dim, embed_dim)
        #self.x_encoder_6 = nn.Linear(embed_dim, embed_dim)

        #self.x_encoder_1 = LEConv(110, 192)
        #self.x_encoder_2 = LEConv(192, embed_dim)
        #GATConv, GINConv, LEConv, FAConv
        #self.x_encoder_1 = GATConv(110, 192)#GENConv and GAT was OK
        #self.x_encoder_2 = GATConv(192, embed_dim)#GENConv and GAT was OK

        self.x_encoder_1 = SAGEConv(110, embed_dim)
        self.x_encoder_2 = SAGEConv(embed_dim, embed_dim)
        self.x_encoder_3 = SAGEConv(embed_dim, embed_dim)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.dropout_kg = nn.Dropout(self.dropout_rate)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU()

        self.fc_drug_conv_0 = nn.Linear(self.conv * 3, self.conv * 3 )
        self.fc_target_conv_0 = nn.Linear(self.conv * 3 , self.conv * 3 )

        self.fc_drug_conv_1 = nn.Linear(185, self.conv * 3 )
        self.fc_target_conv_1 = nn.Linear(1179, self.conv * 3 )

        #self.drug_info_ff = nn.Linear(self.conv * 6, self.conv * 3)

        #self.fc_drug = nn.Linear(self.kg_dim, self.conv * 3 )
        #self.fc_target = nn.Linear(self.kg_dim, self.conv * 3 )

        self.drug_info_ff = nn.Linear(569, 569)
        self.target_info_ff = nn.Linear(2742, 569)

        self.fc1 = nn.Linear(1938, 1536)#192, 1024     1632  1568
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(1536, 1536)#1024, 1024
        self.dropout2 = nn.Dropout(self.dropout_rate)
        self.fc3 = nn.Linear(1536, 1024)#1024, 512
        self.dropout3 = nn.Dropout(self.dropout_rate)

        self.out = nn.Linear(1024, 1)#512, 1
        torch.nn.init.constant_(self.out.bias, 5)

    def process_mol_data(self, graph: torch.Tensor, batch: torch.Tensor, target_size: int) -> torch.Tensor:
        """Process molecular features tensor to have exactly target_size nodes through padding or truncation for each graph in batch."""
        unique_batch = torch.unique(batch)
        batch_size = len(unique_batch)
        num_features = graph.size(-1)
        
        # Create output tensor for all graphs in batch
        output = torch.zeros(batch_size, target_size, num_features, device=graph.device)
        
        # Process each graph in batch
        for i, b in enumerate(unique_batch):
            mask = (batch == b)
            graph_i = graph[mask]
            num_nodes = graph_i.size(0)
            
            # Copy data up to target size for this graph
            n = min(num_nodes, target_size)
            output[i, :n] = graph_i[:n]
            
        return output

    def forward(self, drug_graph, drug_smile, drug_kg_embedding, target_sequence, target_kg_embedding):
        
        graph, edge_index, batch = drug_graph.x.to(self.device), drug_graph.edge_index.to(self.device), drug_graph.batch.to(self.device)
        #print('the graph:',graph)
        #print('edge index:',edge_index)
        graph = F.relu_(self.x_encoder_1(graph, edge_index))
        graph = F.relu_(self.x_encoder_2(graph, edge_index))
        graph = F.relu_(self.x_encoder_3(graph, edge_index))
        #graph = F.relu_(self.x_encoder_4(graph, edge_index))

        graph_feature = self.process_mol_data(graph, batch, 100)
        graph_feature = graph_feature.permute(0, 2, 1)

        #graph_feature = graph_feature.permute(0, 2, 1)

        #print(graph_process.shape)
        #raise Exception('stop here')
        #print(graph.shape) #384
        #print(batch.shape)
        #graph = global_max_pool(graph, batch)
        #graph = F.relu_(self.x_encoder_5(graph))
        #graph = F.dropout(graph, p=.5, training=self.training)
        #graph = F.relu_(self.x_encoder_6(graph))

        #print('the graph:',graph.shape) #256 192

        drugembed = self.drug_embed(drug_smile)
        proteinembed = self.protein_embed(target_sequence)
        drugembed = drugembed.permute(0, 2, 1)
        proteinembed = proteinembed.permute(0, 2, 1)

        drugConv = self.Drug_CNNs(drugembed)
        proteinConv = self.Protein_CNNs(proteinembed)

        #drugConv = drugConv.permute(0, 2, 1)
        #proteinConv = proteinConv.permute(0, 2, 1)
        
        #print(graph_feature.shape)
        #print(drugConv.shape)
        
        #graph_info = torch.cat([graph_feature, drugConv], dim=1)
        #graph_info = graph_info.permute(0, 2, 1)
        #print(graph_info.shape)
        #raise Exception('stop here')

        #print('the drugConv:',drugConv.shape)
        #print('the protein:',proteinConv.shape)
        #print('the graph:',graph_feature.shape)

        drugConv_0, proteinConv_0 = self.attention_cnn(drugConv,proteinConv)
        drugGNN_0, proteinGNN_0 = self.attention_gnn(graph_feature,proteinConv)

        #drugConv_0 ,proteinConv_0 = self.attention(graph_info,proteinConv)
        #print(drugConv_0.shape)
        drugConv_1 = drugConv_0.permute(0, 2, 1)
        drugGNN_1 = drugGNN_0.permute(0, 2, 1)

        proteinConv_1 = proteinConv_0.permute(0, 2, 1)
        proteinGNN_1 = proteinGNN_0.permute(0, 2, 1)

        #print(drugConv_0.shape)
        drugConv_0 = self.Drug_max_pool_0_cnn(drugConv_0).squeeze(2)
        proteinConv_0 = self.Protein_max_pool_0_cnn(proteinConv_0).squeeze(2)
        drugGNN_0 = self.Drug_max_pool_0_gnn(drugGNN_0).squeeze(2)
        proteinGNN_0 = self.Protein_max_pool_0_gnn(proteinGNN_0).squeeze(2)

        drugConv_1 = self.Drug_max_pool_1_cnn(drugConv_1).squeeze(2)
        proteinConv_1 = self.Protein_max_pool_1_cnn(proteinConv_1).squeeze(2)
        drugGNN_1 = self.Drug_max_pool_1_gnn(drugGNN_1).squeeze(2)
        proteinGNN_1 = self.Protein_max_pool_1_gnn(proteinGNN_1).squeeze(2)

        #drugConv_1 = drugConv_1 + graph

        #drug_kg_embedding = self.fc_drug(drug_kg_embedding)
        #drug_kg_embedding = self.dropout_kg(drug_kg_embedding)
        #target_kg_embedding = self.fc_target(target_kg_embedding)
        #target_kg_embedding = self.dropout_kg(target_kg_embedding)

        #drugConv_0 = self.fc_drug_conv_0(drugConv_0)
        #drugConv_0 = self.dropout_kg(drugConv_0)
        #proteinConv_0 = self.fc_target_conv_0(proteinConv_0)
        #proteinConv_0 = self.dropout_kg(proteinConv_0)
        #print(drugConv_1.shape)
        #print(drugConv_1.shape)

        #drugConv_1 = self.fc_drug_conv_1(drugConv_1)
        #drugConv_1 = self.dropout_kg(drugConv_1)
        #proteinConv_1 = self.fc_target_conv_1(proteinConv_1)
        #proteinConv_1 = self.dropout_kg(proteinConv_1)

        #drug_info = torch.cat([drugConv_1, graph], dim=1)

        #drug_info = self.drug_info_ff(drug_info)

        #drug_info = drugConv + drug_kg_embedding
        #target_info = proteinConv + target_kg_embedding
        '''
        print(drugConv_0.shape)
        print(drugConv_1.shape)
        print(drug_kg_embedding.shape)
        print(proteinConv_0.shape)
        print(proteinConv_1.shape)
        print(target_kg_embedding.shape)
        '''

        #print('d0:',drugConv_0.shape)
        #print('d1:',drugConv_1.shape)
        #print('p0:',proteinConv_0.shape)
        #print('p1:',proteinConv_1.shape)

        drug_pool_info = torch.cat([drugConv_0, drugConv_1, drugGNN_0, drugGNN_1], dim=1)
        target_pool_info = torch.cat([proteinConv_0, proteinConv_1, proteinGNN_0, proteinGNN_1], dim=1)

        drug_pool_info = self.drug_info_ff(drug_pool_info)
        target_pool_info = self.target_info_ff(target_pool_info)

        #all_info = torch.cat([drugConv_0, drugConv_1, drug_kg_embedding, proteinConv_0, proteinConv_1, target_kg_embedding], dim=1)
        #all_info = torch.cat([drugConv_1, drug_kg_embedding, proteinConv_1, target_kg_embedding], dim=1)
        #all_info = torch.cat([drugConv_0, drugConv_1, drug_kg_embedding, proteinConv_0, proteinConv_1, target_kg_embedding], dim=1)
        all_info = torch.cat([drug_pool_info, drug_kg_embedding, target_pool_info, target_kg_embedding], dim=1)

        fully1 = self.leaky_relu(self.fc1(all_info))
        fully1 = self.dropout1(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout2(fully2)
        fully3 = self.leaky_relu(self.fc3(fully2))
        fully3 = self.dropout3(fully3)
        predict = self.out(fully3)
        #print(predict.shape)
        return predict
    
#-------------------------------------------------dataset------------------------------------------------------------




def atom_features(atom):
    HYB_list = [Chem.rdchem.HybridizationType.S, Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2,
                Chem.rdchem.HybridizationType.UNSPECIFIED, Chem.rdchem.HybridizationType.OTHER]
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Sm', 'Tc', 'Gd', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetExplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding(atom.GetFormalCharge(), [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5]) +
                    one_of_k_encoding(atom.GetHybridization(), HYB_list) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))



def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature)

    features = np.array(features)

    edges = []
    edge_type = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edge_type.append(bond.GetBondTypeAsDouble())
    g = nx.Graph(edges).to_directed()
    edge_index = []

    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    if not edge_index:
        edge_index = [[0, 0]] #add self loop
        edge_index = np.array(edge_index).transpose(1, 0)
    else:
        edge_index = np.array(edge_index).transpose(1, 0)

    #if not edge_index:
    #    #print('an edge is empty:', )
    #    raise Exception('an edge is empty:', list(mol.GetBonds()))

    return c_size, features, edge_index, edge_type



class TrainDataset(InMemoryDataset):
    def __init__(self, data):
        super(TrainDataset, self).__init__()
        self.data = data
        self.drug_smiles = [item[0] for item in data]
        
        # Process molecular graphs
        self.graphs = []
        for smiles in self.drug_smiles:
            #print(smiles)
            c_size, features, edge_index, edge_type = smile_to_graph(smiles)
            g = Data(
                x=torch.FloatTensor(features),
                edge_index=torch.LongTensor(edge_index),
                edge_attr=torch.FloatTensor(edge_type) if len(edge_type) > 0 else torch.FloatTensor([]),
                num_nodes=c_size
            )
            self.graphs.append(g)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get molecular graph
        #mol_graph = self.graphs[idx]
        
        smiles, smiles_feat, kg_embed1, seq_feat, kg_embed2, label = self.data[idx]
        
        return (
                self.graphs[idx],
                torch.LongTensor(smiles_feat),
                torch.FloatTensor(kg_embed1), 
                torch.LongTensor(seq_feat),
                torch.FloatTensor(kg_embed2),
                torch.FloatTensor([label])
            )



#----------------------------------------Test Custom Dataset-------------------------------------------------

class TestDataset(InMemoryDataset):
    def __init__(self, data):
        super(TestDataset, self).__init__()
        self.data = data
        self.drug_smiles = [item[0] for item in data]
        
        # Process molecular graphs
        self.graphs = []
        for smiles in self.drug_smiles:
            #print('the smile:',smiles)
            c_size, features, edge_index, edge_type = smile_to_graph(smiles)
            g = Data(
                x=torch.FloatTensor(features),
                edge_index=torch.LongTensor(edge_index),
                edge_attr=torch.FloatTensor(edge_type) if len(edge_type) > 0 else torch.FloatTensor([]),
                num_nodes=c_size
            )
            self.graphs.append(g)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get molecular graph
        #mol_graph = self.graphs[idx]
        
        smiles, smiles_feat, kg_embed1, seq_feat, kg_embed2 = self.data[idx]
        
        return (
                self.graphs[idx],
                torch.LongTensor(smiles_feat),
                torch.FloatTensor(kg_embed1), 
                torch.LongTensor(seq_feat),
                torch.FloatTensor(kg_embed2),
            )

#---------------------------------------Training Custom Dataset-----------------------------------------------

class Custom_Train_Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input1, input2, input3, input4, label = self.data[idx]
        input1 = torch.tensor(input1)
        input2 = torch.tensor(input2, dtype=torch.float)
        input3 = torch.tensor(input3)
        input4 = torch.tensor(input4, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.float)
        return input1, input2, input3, input4, label

#--------------------------------------Training prediction Model-----------------------------------------------

def calculate_average_loss(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for input0, input1, input2, input3, input4, labels in data_loader:
            input0 = input0.to(device)
            input1 = input1.to(device)
            input2 = input2.to(device)
            input3 = input3.to(device)
            input4 = input4.to(device)
            labels = labels.to(device)
            outputs = model(input0, input1, input2, input3, input4)
            #outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item() * len(input1)
            total_samples += len(input1)

    return total_loss / total_samples

def Train_Prediction_Model(input_data, kg_dim = 96, num_epochs = 100, batch_size = 512, device_ids = 'cuda:0', save_path = 'prediction_model/bilinear_model.pth', log_filename = 'log/training_log.txt', early_stop = True, early_stop_threshold=0.001, lr = 0.001):
    gpu_ids = device_ids.split(',')
    final_device_ids = []
    for item in gpu_ids:
        final_device_ids.append(item)
    custom_dataset = TrainDataset(input_data)
    train_loader = DataLoader(custom_dataset, batch_size = batch_size, shuffle=True)

    model = CNNs_Attention(kg_dim = kg_dim, device = final_device_ids[0])

    """weight initialize"""
    weight_p, bias_p = [], []
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for name, p in model.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]
    """load trained model"""
    
    if len(final_device_ids) > 1:
        model = DataParallel(model, device_ids=final_device_ids)
        model = model.to(final_device_ids)
    else:
        model = model.to(final_device_ids[0]) 
    
    criterion = nn.BCEWithLogitsLoss()
    #criterion = nn.CrossEntropyLoss()   maybe try CrossEntropyLoss
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    clip_value = 1.0
    #best_loss = float('inf')
    model.train()

    if len(final_device_ids) == 1:#use a single gpu
        print("training with the following gpu:",final_device_ids)
        for epoch in tqdm(range(num_epochs)):
            running_loss = 0.0
            total_batches = len(train_loader)
            #with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
            for i, (smiles_graph, drug_smile, drug_kg_embedding, target_sequence, target_kg_embedding, labels) in enumerate(train_loader):
                
                #drug_smile, drug_embedding, target_sequence, target_embedding
                smiles_graph = smiles_graph.to(final_device_ids[0])
                drug_smile = drug_smile.cuda(final_device_ids[0])
                drug_kg_embedding = drug_kg_embedding.cuda(final_device_ids[0])
                target_sequence = target_sequence.cuda(final_device_ids[0])
                target_kg_embedding = target_kg_embedding.cuda(final_device_ids[0])
                labels = labels.cuda(final_device_ids[0])

                optimizer.zero_grad()

                # Forward pass
                outputs = model(smiles_graph, drug_smile, drug_kg_embedding, target_sequence, target_kg_embedding)
                loss = criterion(outputs, labels)#outputs.squeeze()
               
                # Backward pass and optimization
                loss.backward()
                
                # Clip gradients to prevent exploding gradients
                nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                
                optimizer.step()

                running_loss += loss.item()
                #if (i + 1) % 1 == 0:
                #    print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
                    #logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            
            # Calculate average loss for the epoch
            average_loss = running_loss / total_batches
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss}')
            #logging.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss}')

            # Check for early stopping
            if average_loss < early_stop_threshold and early_stop == True:
                print(f'Loss is below the early stopping threshold. Training stopped.')
                break
    
    print('Training finished!')
    torch.save(model.state_dict(), save_path)
    print('save model succesuffly')

def Attention_prediction(Test_data, kg_dim = 100, load_path = 'prediction_model/bilinear_MLP.pth', batch_size = 128, sigmoid_transform = True , device = 'cuda:0'):
    print('start loading data')

    model = CNNs_Attention(kg_dim = kg_dim, device = device)

    state_dict = torch.load(load_path, map_location=device)  # Load the state dictionary

    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)

    model = model.to(device)

    model.eval()

    print('prediction load succesfully')

    Test_dataset = TestDataset(Test_data)

    Test_loader = DataLoader(Test_dataset, batch_size = batch_size, shuffle = False)

    all_outputs = []

    with torch.no_grad():

        for i, (smile_graph, drug_smile, drug_kg_embedding, target_sequence, target_kg_embedding) in enumerate(tqdm(Test_loader, desc='Inference')):
            #print(smile_graph)
            smile_graph = smile_graph.to(device)
            drug_smile = drug_smile.cuda(device)
            drug_kg_embedding = drug_kg_embedding.cuda(device)
            target_sequence = target_sequence.cuda(device)
            target_kg_embedding = target_kg_embedding.cuda(device)
            logits = model(smile_graph, drug_smile, drug_kg_embedding, target_sequence, target_kg_embedding)
            if sigmoid_transform:
                logits = torch.sigmoid(logits)
            outputs_list = logits.tolist()

            all_outputs.extend(outputs_list)
    results = []

    for item in all_outputs:

        results.append(item[0])
        
    return results
    