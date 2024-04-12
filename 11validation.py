# coding: utf-8
import os
# Disable HIP and set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use NVIDIA GPU 0
os.environ["ROCBLAS_LAYER"] = "0"  # Disable ROCm/HIP
os.environ["ROCFFT_LAYER"] = "0"  # Disable ROCm/HIP
import torch
import my_Utils
# Set GPU memory fraction (adjust the fraction as needed)
#torch.cuda.set_per_process_memory_fraction(0.5)
torch.backends.cuda.max_split_size_mb = 256  # Adjust this value as needed
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time
import random
import math
torch.manual_seed(100)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
time_start=time.time()
# GO_IDs = []

CFG = {
    'cfg00': [16, 'M', 16, 'M'],
    'cfg01': [16, 'M', 32, 'M'],
    'cfg02': [32, 'M'],
    'cfg03': [64, 'M'],
    'cfg04': [16, 'M', 16, 'M',32, 'M'],
    'cfg05': [128, 'M', 256, 'M',256, 'M'],
    'cfg06': [64, 'M', 32, 'M',32, 'M'],
    'cfg07': [128, 'M', 64, 'M2'],
    'cfg08': [512, 'M', 128, 'M2',32, 'M2'],
}
OUT_nodes = {
    'BP': 215,
    'MF': 131,
    'CC': 50,
}

Thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
              0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2,
              0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3,
              0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4,
              0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5,
              0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6,
              0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7,
              0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8,
              0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9,
              0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]

file_path = '/home/bvs/neelam/input_ourmodel/input/protVec_dict.npy'
ProtVec = np.load(file_path, allow_pickle=True).item()
Seqfile_name = 'seqSet.csv'
Domainfile_name = 'NewdomainSetF.csv'
GOfile_name = 'BactProteinGO.csv'

class Dataload(Dataset):
    def __init__(self, benchmark_list, seqfile_name, domainfile_name, GOfile_name, func='BP', transform=None):
        self.benchmark_list = benchmark_list
        self.sequeces = {}
        self.max_seq_len = 1500  # 
        self.doamins = {}
        self.max_domains_len = 357  # 
        self.ppiVecs = {}
        self.GO_annotiations = {}
        #self.num_output_features = 215 
        # self.max_GOnums_len = 0     #

        with open(seqfile_name, 'r') as f:  #seqfile_name = 'seqSet.csv'
            for line in f:
                items = line.strip().split(',')
                prot, seq = items[0], items[1]
                self.sequeces[prot] = seq
        self.protDict = ProtVec

        with open(domainfile_name, 'r') as f:   #domainfile_name = 'domainSet.csv'
            for line in f:
                items = line.strip().split(',')
                prot, domains = items[0], items[1:]
                domains = [int(x) for x in domains]
                self.doamins[prot] = domains
        ppi_file = 'selected_208964_protein_score.csv'
        print(ppi_file)
        with open(ppi_file, 'r') as f:
            num = 1
            for line in f:
                if num == 1:
                    num = 2
                    continue
                items = line.strip().split(',')
                prot, vector =items[0], items[1:]
                self.ppiVecs[prot] = vector

        with open(GOfile_name, 'r') as f:       #GOfile_name = 'humanProteinGO.csv'
            num = 1
            for line in f:
                if num == 1:
                    num = 2
                    continue
                items = line.strip().split(',')
                if func == 'BP':
                    prot, GO_annotiation = items[0], items[1:216]
                elif func == 'MF':
                    prot, GO_annotiation = items[0], items[216:347]
                elif func == 'CC':
                    prot, GO_annotiation = items[0], items[347:]
                # prot, GO_annotiation = items[0], items[1:]
                self.GO_annotiations[prot] = GO_annotiation
    def __getitem__(self, idx):
        iprID = self.benchmark_list[idx]
        seq = self.sequeces[iprID]
        if len(seq) > self.max_seq_len:
            seq = seq[0:self.max_seq_len]
        seqMatrix = my_Utils.mer_k(seq, self.protDict, 3)
        seqMatrix = np.array(seqMatrix, dtype=float)
        if (seqMatrix.shape[0]) < self.max_seq_len:
            seqMatrix = np.pad(seqMatrix, ((0, self.max_seq_len - (seqMatrix.shape[0])), (0, 0)),
                               'constant', constant_values=0)
        seqMatrix = seqMatrix.T
        seqMatrix = torch.from_numpy(seqMatrix).type(torch.FloatTensor).cuda()
        #domain
        domain_s = self.doamins[iprID]
        if len(domain_s) >= self.max_domains_len:
            domain_s = np.array(domain_s[0:self.max_domains_len], dtype=int)
        # if len(domain_s) < self.max_domains_len:
        else:
            domain_s = np.array(domain_s, dtype=int)
            domain_s = np.pad(domain_s, ((0, self.max_domains_len-len(domain_s))), 'constant', constant_values=0)
        domainSentence = torch.from_numpy(domain_s).type(torch.LongTensor).cuda()

        # PPI
        if iprID not in self.ppiVecs:
            ppiVect = np.zeros((5563), dtype=float).tolist()
        else:
            ppiVect = self.ppiVecs[iprID]
            ppiVect = [float(x) for x in ppiVect]
        ppiVect = torch.Tensor(ppiVect).cuda()
        ppiVect = ppiVect.type(torch.FloatTensor)

        #GO
        GO_annotiations = self.GO_annotiations[iprID]
        GO_annotiations = [int(x) for x in GO_annotiations]
        GO_annotiations = torch.Tensor(GO_annotiations).cuda()
        # GO_annotiations = GO_annotiations.type(torch.LongTensor)
        GO_annotiations = GO_annotiations.type(torch.FloatTensor)
        #target = torch.randint(0, 2, (GO_annotiations.size(0),))  # Random binary labels for each sample
        

        return seqMatrix, domainSentence, ppiVect, GO_annotiations

    def __len__(self):
        return len(self.benchmark_list)     #


class weight_Dataload(Dataset):
    def __init__(self, benchmark_list, seqdict, domaindict, ppidict, GOfile_name, func = 'BP', transform=None):
        self.benchmark = benchmark_list
        self.weghtdict = {}
        self.GO_annotiations = {}

        for i in range(len(benchmark_list)):
            prot = benchmark_list[i]
            temp = [seqdict[prot], domaindict[prot], ppidict[prot]]
            temp = np.array(temp)
            self.weghtdict[benchmark_list[i]] = temp.flatten().tolist()
            assert len(seqdict[prot]) == len(domaindict[prot]) == len(ppidict[prot]) == OUT_nodes[func]

        with open(GOfile_name, 'r') as f:      
            num = 1
            for line in f:
                if num == 1:
                    num = 2
                    # items = line.strip().split(',')
                    # GO_IDs = items
                    continue
                items = line.strip().split(',')
                if func == 'BP':
                    prot, GO_annotiation = items[0], items[1:216]
                elif func == 'MF':
                    prot, GO_annotiation = items[0], items[216:327]
                elif func == 'CC':
                    prot, GO_annotiation = items[0], items[347:]
                # prot, GO_annotiation = items[0], items[1:]
                self.GO_annotiations[prot] = GO_annotiation
    def __getitem__(self, idx):
        prot = self.benchmark[idx]
        #weight_classifier
        weight_features = self.weghtdict[prot]
        weight_features = [float(x) for x in weight_features]
        weight_features = torch.Tensor(weight_features).cuda()
        weight_features = weight_features.type(torch.FloatTensor)
        GO_annotiations = self.GO_annotiations[prot]
        GO_annotiations = [int(x) for x in GO_annotiations]
        GO_annotiations = torch.Tensor(GO_annotiations).cuda()
        # GO_annotiations = GO_annotiations.type(torch.LongTensor)
        GO_annotiations = GO_annotiations.type(torch.FloatTensor)

        return weight_features, GO_annotiations

    def __len__(self):
        return len(self.benchmark)


class Seq_Module(nn.Module):
    def __init__(self, func):
        super(Seq_Module, self).__init__()
        self.seq_CNN = self.SeqConv1d(CFG['cfg05']).cuda()
        self.seq_FClayer = nn.Linear(in_features=48128, out_features=1024, bias=True)
        #self.outlayer = nn.Linear(in_features=215, out_features=1720, bias=True)

        #self.seq_FClayer = nn.Linear(3008, 1024).cuda()
        self.seq_outlayer = nn.Linear(1024, OUT_nodes[func]).cuda()
 
    def forward(self, seqMatrix):
        # seqMatrix = self.seq_emblayer(seqSentence)
        seq_out = self.seq_CNN(seqMatrix)
        seq_out = seq_out.view(seq_out.size(0), -1)  
        # print(seq_out)
        seq_out = F.dropout(self.seq_FClayer(seq_out), p=0.3, training=self.training)
        seq_out = F.relu(seq_out)
        #seq_out = self.seq_outlayer(seq_out)
        seq_out = torch.sigmoid(self.seq_outlayer(seq_out))
        #seq_out = F.sigmoid(self.seq_outlayer(seq_out))
        #seq_out = F.sigmoid(seq_out)
        return seq_out

    # sequence的1D_CNN
    def SeqConv1d(self, cfg):
        layers = []
        in_channels = 100
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool1d(kernel_size=2)]
            elif x == 'M2':
                layers += [nn.MaxPool1d(kernel_size=2, stride=1)]
            else:
                layers += [nn.Conv1d(in_channels, x, kernel_size=16, stride=1, padding=8),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)


class Domain_Module(nn.Module):
    def __init__(self, func):
        super(Domain_Module, self).__init__()
        self.dom_emblayer = nn.Embedding(2984, 128, padding_idx=0).cuda()
        self.dom_CNN = self.DomainConv1d(CFG['cfg07']).cuda()
        self.dom_FClayer = nn.Linear(1088, 512).cuda()
        self.dom_outlayer = nn.Linear(512, OUT_nodes[func]).cuda()

    def forward(self, domainSentence): #seq 4981*100  ,domain
        domain_matrix = self.dom_emblayer(domainSentence)
        domain_out = self.dom_CNN(domain_matrix)
        domain_out = domain_out.view(domain_out.size(0), -1)  # 
        # print(domain_out)
        domain_out = F.dropout(self.dom_FClayer(domain_out), p=0.3, training=self.training)
        domain_out = F.relu(domain_out)
        domain_out = self.dom_outlayer(domain_out)
        domain_out = torch.sigmoid(domain_out)
       # domain_out = F.sigmoid(domain_out)
        return domain_out
    # domain1D_CNN
    def DomainConv1d(self, cfg):
        layers = []
        # in_channels = 128
        in_channels = 357
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
            elif x == 'M2':
                layers += [nn.MaxPool1d(kernel_size=2, stride=1)]
            else:
                layers += [nn.Conv1d(in_channels, x, kernel_size=2, stride=2, padding=2),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)
class PPI_Module(nn.Module):
    def __init__(self, func):
        super(PPI_Module, self).__init__()
        self.ppi_inputlayer = nn.Linear(5563, 4096).cuda()
        self.ppi_hiddenlayer = nn.Linear(4096, 1024).cuda()
        self.ppi_outlayer = nn.Linear(1024, OUT_nodes[func]).cuda()
        #self.ppi_outlayer = nn.Linear(1024, OUT_nodes[func]).cuda()
    def forward(self, ppiVec):
        ppi_out = F.dropout(self.ppi_inputlayer(ppiVec), p=0.00005, training=self.training)
        ppi_out = F.dropout(self.ppi_hiddenlayer(ppi_out), p=0.3, training=self.training)
        ppi_out = self.ppi_outlayer(ppi_out)
        ppi_out = torch.sigmoid(ppi_out)
        #ppi_out = F.sigmoid(ppi_out)
        return ppi_out

class Weight_classifier(nn.Module):
    def __init__(self, func):
        super(Weight_classifier, self).__init__()
        # self.weight_layer = nn.Linear(OUT_nodes[func]*3, OUT_nodes[func])
        self.weight_layer = MaskedLinear(OUT_nodes[func]*3, OUT_nodes[func], '/home/bvs/neelam/input_ourmodel/input/BP_maskmatrix645.csv'.format(func)).cuda()
        self.outlayer= nn.Linear(OUT_nodes[func], 215)

    def forward(self, weight_features):
        weight_out = self.weight_layer(weight_features)
        # weight_out = F.sigmoid(weight_out)
        weight_out = F.relu(weight_out)
        weight_out = F.sigmoid(self.outlayer(weight_out))
        #weight_out = weight_out.unsqueeze(0).expand(8, -1)
        return weight_out


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, relation_file, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)

        mask = self.readRelationFromFile(relation_file)
        self.register_buffer('mask', mask)
        self.iter = 0

    def forward(self, input):
        masked_weight = self.weight * self.mask
        return F.linear(input, masked_weight, self.bias)

    def readRelationFromFile(self, relation_file):
        mask = []
        with open(relation_file, 'r') as f:
            for line in f:
                l = [int(x) for x in line.strip().split(',')]
                for item in l:
                    assert item == 1 or item == 0  # relation 只能为0或者1
                mask.append(l)
        return Variable(torch.Tensor(mask))



def benchmark_set_split(term_arg='MF'):
    benchmark_file = '/_{}benchmarkSetcomm_2.csv'.format(
        term_arg)
    print(benchmark_file)
    trainset, testset = [], []
    all_data = []
    with open(benchmark_file, 'r') as f:
        for line in f:
            item = line.strip()
            all_data.append(item)
    idx_list = np.arange(len(all_data)).tolist()
    
    nums = {
        'BP': 215, #10700,
        'MF': 131, #10500,
        'CC':  50, #10000,
        'test': 10
    }
    random_index = []
    with open('{}_random_index.csv'.format(term_arg), 'r') as f:
        for line in f:
            item = line.strip().split(',')
            random_index.append(int(item[0]))
    for i in range(len(all_data)):
        if i in random_index:
            trainset.append(all_data[i])
        else:
            testset.append(all_data[i])
    assert len(trainset) + len(testset) == len(all_data)
    return trainset, testset

def calculate_performance(actual, pred_prob, threshold=0.7, average='micro'):
    pred_lable = []
    for l in range(len(pred_prob)):
        eachline = (np.array(pred_prob[l]) > threshold).astype(int)
        eachline = eachline.tolist()
        pred_lable.append(eachline)
    f_score = f1_score(np.array(actual), np.array(pred_lable), average=average)
    recall = recall_score(np.array(actual), np.array(pred_lable), average=average)
    precision = precision_score(np.array(actual), np.array(pred_lable), average=average)
    return f_score, recall,  precision

def cacul_aupr(lables, pred):
    precision, recall, _thresholds = metrics.precision_recall_curve(lables, pred)
    aupr = metrics.auc(recall, precision)
    return aupr

def Seq_train(learningrate, batchsize, train_benchmark, test_benchmark, epochtime, func='BP'):
    print('{}  seqmodel start'.format(func))
    seq_model = Seq_Module(func).cuda()
    batch_size = 8  # You can reduce the batch size to a smaller value
    #batch_size = batchsize
    learning_rate = learningrate
    epoch_times = epochtime
    print(seq_model)
    print('batch_size_{},learning_rate_{},epoch_times_{}'.format(batch_size, learning_rate, epoch_times))
    #loss_function = nn.BCEWithLogitsLoss()
    loss_function = nn.BCEWithLogitsLoss(reduction='mean')  # Add reduction argument
    #loss_function = nn.BCELoss()
    optimizer = optim.Adam(seq_model.parameters(), lr=learning_rate, weight_decay = 0.00001)

    train_dataset = Dataload(train_benchmark, Seqfile_name, Domainfile_name, GOfile_name, func=func)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = Dataload(test_benchmark, Seqfile_name, Domainfile_name, GOfile_name, func=func)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    seq_model.train()
    best_fscore = 0
    for epoch in range(epoch_times):
        _loss = 0
        batch_num = 0
        torch.cuda.empty_cache()  # Add this line to release GPU memory
        for batch_idx, (seqMatrix, domainStence, ppiVect, GO_annotiations) in enumerate(train_data_loader):
           # print(type(seqMatrix), type(domainStence), type(ppiVect), type(GO_annotiations))
            seqMatrix = Variable(seqMatrix).cuda()
            GO_annotiations = torch.squeeze(GO_annotiations)
            GO_annotiations = Variable(GO_annotiations.unsqueeze(1)).cuda()
            GO_annotiations = Variable(GO_annotiations).cuda()
            GO_annotiations = torch.squeeze(GO_annotiations, dim=0)  # Remove the first dimension
            #GO_annotiations = GO_annotiations.unsqueeze(0)  # Add a batch dimension
            #GO_annotiations = GO_annotiations.unsqueeze(-1)
            out = seq_model(seqMatrix)
            #print("Output Shape:", out.shape)
            #print("Target Shape:", GO_annotiations.shape)
            # Reshape the model's output to match the shape of GO_annotiations
            out = out.view(GO_annotiations.shape)
            optimizer.zero_grad()
            #GO_annotiations = GO_annotiations.view(batch_size, -1)
            #print(f"Output shape: {out.shape}, Target shape: {GO_annotiations.shape}")
            loss = loss_function(out, GO_annotiations)
            batch_num += 1
            loss.backward()
            optimizer.step()
            _loss += loss.item()

            # Before training loop
            torch.cuda.empty_cache()
        epoch_loss = "{}".format(_loss / batch_num)
        t_loss = 0
        test_batch_num = 0
        pred = []
        actual = []
        for idx, (seqMatrix, domainStence, ppiVect, GO_annotiations) in enumerate(test_data_loader):
            seqMatrix = Variable(seqMatrix).cuda()
            #GO_annotiations = Variable(GO_annotiations).cuda()
            GO_annotiations = Variable(GO_annotiations.unsqueeze(1)).cuda()
            GO_annotiations = torch.squeeze(GO_annotiations, dim=0)  # Remove the first dimension
            #GO_annotiations = GO_annotiations.unsqueeze(0)  # Add a batch dimension
            out = seq_model(seqMatrix)
            test_batch_num = test_batch_num + 1
            pred.append(out.data[0].cpu().tolist())
            actual.append(GO_annotiations.data[0].cpu().tolist())
            one_loss = loss_function(out, GO_annotiations)
            t_loss += one_loss.item()
           # t_loss += one_loss.data[0]
        test_loss = "{}".format(t_loss / test_batch_num)
        fpr, tpr, th = roc_curve(np.array(actual).flatten(), np.array(pred).flatten(), pos_label=1)
        auc_score = auc(fpr, tpr)
        score_dict = {}
        each_best_fcore = 0

        each_best_scores = []
        for i in range(len(Thresholds)):
            f_score, recall, precision = calculate_performance(
                actual, pred, threshold=Thresholds[i], average='micro')
            if f_score >= each_best_fcore:
                each_best_fcore = f_score
                each_best_scores = [Thresholds[i], f_score, recall, precision, auc_score]
            scores = [f_score, recall, precision, auc_score]
            score_dict[Thresholds[i]] = scores
        if each_best_fcore >= best_fscore:
            best_fscore = each_best_fcore
            best_scores = each_best_scores
            best_score_dict = score_dict
            torch.save(seq_model, 'savedpkl/Seq1DVal_{}_{}_{}_{}.pkl'.format(func, batch_size, learning_rate, epoch_times))
        t, f_score, recall = each_best_scores[0], each_best_scores[1], each_best_scores[2]
        precision, auc_score = each_best_scores[3], each_best_scores[4]
        print('epoch{},loss{},testloss:{},t{},f_score{}, auc{}, recall{}, precision{}'.format(
            epoch, epoch_loss, test_loss, t, f_score, auc_score, recall, precision))
    bestthreshold, f_max, recall_max = best_scores[0], best_scores[1], best_scores[2]
    prec_max, bestauc_score = best_scores[3], best_scores[4]
    print('lr:{},batch:{},epoch{},f_max:{}\nauc{},recall_max{},prec_max{},threshold:{}'.format(
        learning_rate, batch_size, epoch_times,
        f_max, bestauc_score, recall_max, prec_max, bestthreshold))
    test_Seqmodel = torch.load('savedpkl/Seq1DVal_{}_{}_{}_{}.pkl'.format(func, batch_size, learning_rate, epoch_times)).cuda()
    t_loss = 0
    seq_test_outs = {}
    # seq_test_outs = []
    pred = []
    actual = []
    score_dict = {}
    batch_num = 0
    for batch_idx, (seqMatrix, domainStence, ppiVect, GO_annotiations) in enumerate(test_data_loader):
        seqMatrix = Variable(seqMatrix).cuda()
        GO_annotiations = Variable(GO_annotiations.unsqueeze(1)).cuda()
        #GO_annotiations = Variable(GO_annotiations).cuda()
        GO_annotiations = torch.squeeze(GO_annotiations, dim=0)  # Remove the first dimension
        #GO_annotiations = GO_annotiations.unsqueeze(0)  # Add a batch dimension
        out = test_Seqmodel(seqMatrix)
        batch_num += 1
        seq_test_outs[test_benchmark[batch_idx]] = out.data[0].cpu().tolist()
        pred.append(out.data[0].cpu().tolist())
        actual.append(GO_annotiations.data[0].cpu().tolist())
        #print(f"Output shape: {out.shape}, Target shape: {GO_annotiations.shape}")
        loss = loss_function(out, GO_annotiations)
        t_loss += one_loss.item()
        #t_loss += loss.data[0]
    test_loss = "{}".format(t_loss / batch_num)
    fpr, tpr, th = roc_curve(np.array(actual).flatten(), np.array(pred).flatten(), pos_label=1)
    auc_score = auc(fpr, tpr)
    each_best_fcore = 0
    for i in range(len(Thresholds)):
        f_score, recall, precision = calculate_performance(
            actual, pred, threshold=Thresholds[i], average='micro')
        if f_score > each_best_fcore:
            each_best_fcore = f_score
            each_best_scores = [Thresholds[i], f_score, recall, precision, auc_score]
        scores = [f_score, recall, precision, auc_score]
        score_dict[Thresholds[i]] = scores
    bestthreshold, f_max, recall_max = each_best_scores[0], each_best_scores[1], each_best_scores[2]
    prec_max, bestauc_score = each_best_scores[3], each_best_scores[4]

    print('test_loss:{},lr:{},batch:{},epoch{},f_max:{}\nauc_score{},recall_max{},prec_max{},threshold:{}'.format(
        test_loss, learning_rate, batch_size, epoch_times,
        f_max, auc_score ,recall_max, prec_max, bestthreshold))

    output_dir = 'out/weight_out/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open('out/weight_out/Seqout{}_lr{}_bat{}_epo{}.csv'.format(
            func, learning_rate, batch_size, epoch_times), 'w') as f:
        f.write('lr:{},batchsize:{},epochtimes:{}\n'.format(learning_rate, batch_size, epoch_times))
        f.write('f_max:{},recall_max{},prec_max{},auc_score:{}\n'.format(
            f_max,recall_max, prec_max, auc_score))
        f.write('threshold,f_score,recall,precision, roc_auc,auc\n')
        for i in range(len(Thresholds)):
            f.write('{},'.format(str(Thresholds[i])))
            f.write('{}\n'.format(','.join(str(x) for x in score_dict[Thresholds[i]])))
        for key, var in seq_test_outs.items():
            f.write('{},'.format(str(key)))
            f.write('{}\n'.format(','.join(str(x) for x in var)))
    train_out_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
    seq_train_outs = {}
    for batch_idx, (seqMatrix, domainsMatrix, ppiVect, GO_annotiations) in enumerate(train_out_loader):
        seqMatrix = Variable(seqMatrix).cuda()
        # GO_annotiations = Variable(GO_annotiations).cuda()
        out = test_Seqmodel(seqMatrix)
        seq_train_outs[train_benchmark[batch_idx]] = out.data[0].cpu().tolist()
    return seq_train_outs, seq_test_outs,bestthreshold        #Seqweight_classifier

def Domain_train(learningrate, batchsize, train_benchmark, test_benchmark, epochtime, func='BP'):
    print('{}  domainmodel start'.format(func))
    domain_model = Domain_Module(func).cuda()
    batch_size = 8 # You can reduce the batch size to a smaller value
    #batch_size = batchsize
    learning_rate = learningrate
    epoch_times = epochtime
    print(domain_model)
    print('batch_size_{},learning_rate_{},epoch_times_{}'.format(batch_size, learning_rate, epoch_times))
    loss_function = nn.BCEWithLogitsLoss(reduction='mean')
    #loss_function = nn.BCELoss()
    optimizer = optim.Adam(domain_model.parameters(), lr=learning_rate, weight_decay=0.00001)

    train_dataset = Dataload(train_benchmark, Seqfile_name, Domainfile_name, GOfile_name, func=func)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = Dataload(test_benchmark, Seqfile_name, Domainfile_name, GOfile_name, func=func)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    domain_model.train()
    best_fscore = 0
    for epoch in range(epoch_times):
        _loss = 0
        batch_num = 0
        torch.cuda.empty_cache()  # Add this line to release GPU memory
        for batch_idx, (seqMatrix, domainStence, ppiVect, GO_annotiations) in enumerate(train_data_loader):
            #print(type(seqMatrix), type(domainStence), type(ppiVect), type(GO_annotiations))
            domainStence = Variable(domainStence).cuda()
            GO_annotiations = torch.squeeze(GO_annotiations)
            GO_annotiations = Variable(GO_annotiations).cuda()
            out = domain_model(domainStence)
            optimizer.zero_grad()
            #GO_annotiations = GO_annotiations.view(batch_size, -1)
            #print(f"Output shape: {out.shape}, Target shape: {GO_annotiations.shape}")
            loss = loss_function(out, GO_annotiations)
            batch_num += 1
            loss.backward()
            optimizer.step()
            _loss += loss.item()
           # _loss += loss.data[0]
        epoch_loss = "{}".format(_loss / batch_num)
        t_loss = 0
        test_batch_num = 0
        pred = []
        actual = []
        for idx, (seqMatrix, domainStence, ppiVect, GO_annotiations) in enumerate(test_data_loader):
            domainStence = Variable(domainStence).cuda()
            GO_annotiations = Variable(GO_annotiations).cuda()
            out = domain_model(domainStence)
            test_batch_num = test_batch_num + 1
            pred.append(out.data[0].cpu().tolist())
            actual.append(GO_annotiations.data[0].cpu().tolist())
            one_loss = loss_function(out, GO_annotiations)
            t_loss += one_loss.item()  # Use .item() to get the scalar value
            #t_loss += one_loss.data[0]
        test_loss = "{}".format(t_loss / test_batch_num)
        fpr, tpr, th = roc_curve(np.array(actual).flatten(), np.array(pred).flatten(), pos_label=1)
        auc_score = auc(fpr, tpr)
        score_dict = {}
        each_best_fcore = 0
        each_best_scores = []
        for i in range(len(Thresholds)):
            f_score, recall, precision = calculate_performance(
                actual, pred, threshold=Thresholds[i], average='micro')
            if f_score >= each_best_fcore:
                each_best_fcore = f_score
                each_best_scores = [Thresholds[i], f_score, recall, precision, auc_score]
            scores = [f_score, recall, precision, auc_score]
            score_dict[Thresholds[i]] = scores
        if each_best_fcore >= best_fscore:
            best_fscore = each_best_fcore
            best_scores = each_best_scores
            best_score_dict = score_dict
            torch.save(domain_model,
                       'savedpkl/Doamin1DVal_{}_{}_{}_{}.pkl'.format(func, batch_size, learning_rate, epoch_times))
        t, f_score, recall = each_best_scores[0], each_best_scores[1], each_best_scores[2]
        precision, auc_score = each_best_scores[3], each_best_scores[4]
        print('epoch{},loss{},testloss:{},t{},f_score{}, auc{}, recall{}, precision{}'.format(
            epoch, epoch_loss, test_loss, t, f_score, auc_score, recall, precision))
    bestthreshold, f_max, recall_max = best_scores[0], best_scores[1], best_scores[2]
    prec_max, bestauc_score = best_scores[3], best_scores[4]
    print('lr:{},batch:{},epoch{},f_max:{}\nauc{},recall_max{},prec_max{},threshold:{}'.format(
        learning_rate, batch_size, epoch_times,
        f_max, bestauc_score, recall_max, prec_max, bestthreshold))
    test_Domainmodel = torch.load(
        'savedpkl/Doamin1DVal_{}_{}_{}_{}.pkl'.format(func, batch_size, learning_rate, epoch_times)).cuda()
    t_loss = 0
    doamin_test_outs = {}
    pred = []
    actual = []
    score_dict = {}
    batch_num = 0
    for batch_idx, (seqMatrix, domainStence, ppiVect, GO_annotiations) in enumerate(test_data_loader):
        domainStence = Variable(domainStence).cuda()
        GO_annotiations = Variable(GO_annotiations).cuda()
        out = test_Domainmodel(domainStence)
        batch_num += 1
        doamin_test_outs[test_benchmark[batch_idx]] = out.data[0].cpu().tolist()
        pred.append(out.data[0].cpu().tolist())
        actual.append(GO_annotiations.data[0].cpu().tolist())
        #print(f"Output shape: {out.shape}, Target shape: {GO_annotiations.shape}")
        loss = loss_function(out, GO_annotiations)
        t_loss += loss.item()
        #t_loss += loss.data[0]
    test_loss = "{}".format(t_loss / batch_num)
    fpr, tpr, th = roc_curve(np.array(actual).flatten(), np.array(pred).flatten(), pos_label=1)
    auc_score = auc(fpr, tpr)
    each_best_fcore = 0
    for i in range(len(Thresholds)):
        f_score, recall, precision = calculate_performance(
            actual, pred, threshold=Thresholds[i], average='micro')
        if f_score > each_best_fcore:
            each_best_fcore = f_score
            each_best_scores = [Thresholds[i], f_score, recall, precision, auc_score]
        scores = [f_score, recall, precision, auc_score]
        score_dict[Thresholds[i]] = scores
    bestthreshold, f_max, recall_max = each_best_scores[0], each_best_scores[1], each_best_scores[2]
    prec_max, bestauc_score = each_best_scores[3], each_best_scores[4]

    print('test_loss:{},lr:{},batch:{},epoch{},f_max:{}\nauc_score{},recall_max{},prec_max{},threshold:{}'.format(
        test_loss, learning_rate, batch_size, epoch_times,
        f_max, auc_score, recall_max, prec_max, bestthreshold))

    with open('out/weight_out/Domainout{}_lr{}_bat{}_epo{}.csv'.format(
            func, learning_rate, batch_size, epoch_times), 'w') as f:
        f.write('lr:{},batchsize:{},epochtimes:{}\n'.format(learning_rate, batch_size, epoch_times))
        f.write('f_max:{},recall_max{},prec_max{},auc_score:{}\n'.format(
            f_max, recall_max, prec_max, auc_score))
        f.write('threshold,f_score,recall,precision, roc_auc,auc\n')
        for i in range(len(Thresholds)):
            f.write('{},'.format(str(Thresholds[i])))
            f.write('{}\n'.format(','.join(str(x) for x in score_dict[Thresholds[i]])))
        for key, var in doamin_test_outs.items():
            f.write('{},'.format(str(key)))
            f.write('{}\n'.format(','.join(str(x) for x in var)))
    train_out_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
    domain_train_outs = {}
    for batch_idx, (seqMatrix, domainStence, ppiVect, GO_annotiations) in enumerate(train_out_loader):
        domainStence = Variable(domainStence).cuda()
        # GO_annotiations = Variable(GO_annotiations).cuda()
        out = test_Domainmodel(domainStence)
        domain_train_outs[train_benchmark[batch_idx]] = out.data[0].cpu().tolist()
    return domain_train_outs, doamin_test_outs, bestthreshold  # Domainweight_classifier

def PPI_train(learningrate, batchsize, train_benchmark, test_benchmark, epochtime, func='BP'):
    print('{}  PPImodel start'.format(func))
    ppi_model = PPI_Module(func).cuda()
    batch_size = 8 # You can reduce the batch size to a smaller value
    #batch_size = batchsize
    learning_rate = learningrate
    epoch_times = epochtime
    print(ppi_model)
    #print('batch_size_{},learning_rate_{},epoch_times_{}'.format(batch_size, learning_rate, epoch_times))
    loss_function = nn.BCEWithLogitsLoss(reduction='mean')
    #loss_function = nn.BCELoss()
    optimizer = optim.Adam(ppi_model.parameters(), lr=learning_rate, weight_decay=0.00001)

    train_dataset = Dataload(train_benchmark, Seqfile_name, Domainfile_name, GOfile_name, func=func)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = Dataload(test_benchmark, Seqfile_name, Domainfile_name, GOfile_name, func=func)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    ppi_model.train()
    best_fscore = 0
    for epoch in range(epoch_times):
        _loss = 0
        batch_num = 0
        torch.cuda.empty_cache()  # Add this line to release GPU memory
        for batch_idx, (seqMatrix, domainStence, ppiVect, GO_annotiations) in enumerate(train_data_loader):
            ppiVect = Variable(ppiVect).cuda()
            GO_annotiations = torch.squeeze(GO_annotiations)
            GO_annotiations = Variable(GO_annotiations.unsqueeze(1)).cuda()
            GO_annotiations = torch.squeeze(GO_annotiations, dim=0)
            #GO_annotiations = Variable(GO_annotiations).cuda()
            out = ppi_model(ppiVect)
            # Reshape the model's output to match the shape of GO_annotiations
            out = out.view(GO_annotiations.shape)
            optimizer.zero_grad()
            #GO_annotiations = GO_annotiations.view(batch_size, -1)
            #print(f"Output shape: {out.shape}, Target shape: {GO_annotiations.shape}")
            loss = loss_function(out, GO_annotiations)
            batch_num += 1
            loss.backward()
            optimizer.step()
            _loss += loss.item()
            #_loss += loss.data[0]
        epoch_loss = "{}".format(_loss / batch_num)
        t_loss = 0
        test_batch_num = 0
        pred = []
        actual = []
        for idx, (seqMatrix, domainStence, ppiVect, GO_annotiations) in enumerate(test_data_loader):
            ppiVect = Variable(ppiVect).cuda()
            GO_annotiations = Variable(GO_annotiations.unsqueeze(1)).cuda()
            GO_annotiations = torch.squeeze(GO_annotiations, dim=0)
            #GO_annotiations = Variable(GO_annotiations).cuda()
            out = ppi_model(ppiVect)
            test_batch_num = test_batch_num + 1
            pred.append(out.data[0].cpu().tolist())
            actual.append(GO_annotiations.data[0].cpu().tolist())
            one_loss = loss_function(out, GO_annotiations)
            t_loss += one_loss.item()  # Use .item() to get the scalar value
            #t_loss += one_loss.data[0]
        test_loss = "{}".format(t_loss / test_batch_num)
        fpr, tpr, th = roc_curve(np.array(actual).flatten(), np.array(pred).flatten(), pos_label=1)
        auc_score = auc(fpr, tpr)
        score_dict = {}
        each_best_fcore = 0
        each_best_scores = []
        for i in range(len(Thresholds)):
            f_score, recall, precision = calculate_performance(
                actual, pred, threshold=Thresholds[i], average='micro')
            if f_score >= each_best_fcore:
                each_best_fcore = f_score
                each_best_scores = [Thresholds[i], f_score, recall, precision, auc_score]
            scores = [f_score, recall, precision, auc_score]
            score_dict[Thresholds[i]] = scores
        if each_best_fcore >= best_fscore:
            best_fscore = each_best_fcore
            best_scores = each_best_scores
            best_score_dict = score_dict
            torch.save(ppi_model,
                       'savedpkl/PPIVal_{}_{}_{}_{}.pkl'.format(func, batch_size, learning_rate, epoch_times))
        t, f_score, recall = each_best_scores[0], each_best_scores[1], each_best_scores[2]
        precision, auc_score = each_best_scores[3], each_best_scores[4]
        print('epoch{},loss{},testloss:{},t{},f_score{}, auc{}, recall{}, precision{}'.format(
            epoch, epoch_loss, test_loss, t, f_score, auc_score, recall, precision))
    bestthreshold, f_max, recall_max = best_scores[0], best_scores[1], best_scores[2]
    prec_max, bestauc_score = best_scores[3], best_scores[4]
    print('lr:{},batch:{},epoch{},f_max:{}\nauc{},recall_max{},prec_max{},threshold:{}'.format(
        learning_rate, batch_size, epoch_times,
        f_max, bestauc_score, recall_max, prec_max, bestthreshold))
    test_PPImodel = torch.load(
        'savedpkl/PPIVal_{}_{}_{}_{}.pkl'.format(func, batch_size, learning_rate, epoch_times)).cuda()
    t_loss = 0
    ppi_test_outs = {}
    pred = []
    actual = []
    score_dict = {}
    batch_num = 0
    for batch_idx, (seqMatrix, domainStence, ppiVect, GO_annotiations) in enumerate(test_data_loader):
        ppiVect = Variable(ppiVect).cuda()
        GO_annotiations = Variable(GO_annotiations.unsqueeze(1)).cuda()
        GO_annotiations = torch.squeeze(GO_annotiations, dim=0)
        #GO_annotiations = Variable(GO_annotiations).cuda()
        out = test_PPImodel(ppiVect)
        batch_num += 1
        ppi_test_outs[test_benchmark[batch_idx]] = out.data[0].cpu().tolist()
        pred.append(out.data[0].cpu().tolist())
        actual.append(GO_annotiations.data[0].cpu().tolist())
        #print(f"Output shape: {out.shape}, Target shape: {GO_annotiations.shape}")
        loss = loss_function(out, GO_annotiations)
        t_loss += loss.item()
        #t_loss += loss.data[0]
    test_loss = "{}".format(t_loss / batch_num)
    fpr, tpr, th = roc_curve(np.array(actual).flatten(), np.array(pred).flatten(), pos_label=1)
    auc_score = auc(fpr, tpr)
    each_best_fcore = 0
    for i in range(len(Thresholds)):
        f_score, recall, precision = calculate_performance(
            actual, pred, threshold=Thresholds[i], average='micro')
        if f_score > each_best_fcore:
            each_best_fcore = f_score
            each_best_scores = [Thresholds[i], f_score, recall, precision, auc_score]
        scores = [f_score, recall, precision, auc_score]
        score_dict[Thresholds[i]] = scores
    bestthreshold, f_max, recall_max = each_best_scores[0], each_best_scores[1], each_best_scores[2]
    prec_max, bestauc_score = each_best_scores[3], each_best_scores[4]

    print('test_loss:{},lr:{},batch:{},epoch{},f_max:{}\nauc_score{},recall_max{},prec_max{},threshold:{}'.format(
        test_loss, learning_rate, batch_size, epoch_times,
        f_max, auc_score, recall_max, prec_max, bestthreshold))

    with open('out/weight_out/PPIout{}_lr{}_bat{}_epo{}.csv'.format(
            func, learning_rate, batch_size, epoch_times), 'w') as f:
        f.write('lr:{},batchsize:{},epochtimes:{}\n'.format(learning_rate, batch_size, epoch_times))
        f.write('f_max:{},recall_max{},prec_max{},auc_score:{}\n'.format(
            f_max, recall_max, prec_max, auc_score))
        f.write('threshold,f_score,recall,precision, roc_auc,auc\n')
        for i in range(len(Thresholds)):
            f.write('{},'.format(str(Thresholds[i])))
            f.write('{}\n'.format(','.join(str(x) for x in score_dict[Thresholds[i]])))
        for key, var in ppi_test_outs.items():
            f.write('{},'.format(str(key)))
            f.write('{}\n'.format(','.join(str(x) for x in var)))

    # 
    train_out_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
    ppi_train_outs = {}
    for batch_idx, (seqMatrix, domainStence, ppiVect, GO_annotiations) in enumerate(train_out_loader):
        ppiVect = Variable(ppiVect).cuda()
        # GO_annotiations = Variable(GO_annotiations).cuda()
        out = test_PPImodel(ppiVect)
        ppi_train_outs[train_benchmark[batch_idx]] = out.data[0].cpu().tolist()
    return ppi_train_outs, ppi_test_outs, bestthreshold  # weight_classifier
def compute_multilabel_confusion_matrix(true_labels, pred_labels):
    num_labels = len(true_labels[0])
    confusion_matrices = []
    for i in range(num_labels):
        cm = confusion_matrix([label[i] for label in true_labels],
                              [label[i] for label in pred_labels],
                              labels=[0, 1])
        confusion_matrices.append(cm)
    return confusion_matrices
# Plot AUC-ROC curve
def plot_auc_roc_curve(true_labels, pred_probs):
    num_labels = len(true_labels[0])
    for i in range(num_labels):
        fpr, tpr, _ = roc_curve([label[i] for label in true_labels], [prob[i] for prob in pred_probs])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()

def Main(train_benchmark, test_benchmark, func='BP'):
    if func == 'BP':
        seq_train_out, seq_test_out, seq_t = Seq_train(0.00001, 8, train_benchmark, test_benchmark, 10, func)  # 15
        domain_train_out, domain_test_out, domain_t = Domain_train(0.00001, 8, train_benchmark, test_benchmark, 10,
                                                                       func)  # 40
    else:
        seq_train_out, seq_test_out, seq_t = Seq_train(0.00001, 8, train_benchmark, test_benchmark, 10, func)    #15
        domain_train_out, domain_test_out, domain_t = Domain_train(0.00001, 8, train_benchmark, test_benchmark, 10, func)  #40
    ppi_train_out, ppi_test_out, ppi_t = PPI_train(0.00001, 8, train_benchmark, test_benchmark, 10, func)   #40

    print('{}  Weight_model start'.format(func))
    learning_rate = 0.00001
    batch_size = 8
    epoch_times = 10
    weight_model = Weight_classifier(func).cuda()
    print(weight_model)
    print('batch_size_{},learning_rate_{},epoch_times_{}'.format(batch_size, learning_rate, epoch_times))
    loss_function = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = optim.Adam(weight_model.parameters(), lr=learning_rate, weight_decay=0.00001)

    train_dataset = weight_Dataload(train_benchmark, seq_train_out, domain_train_out, ppi_train_out, GOfile_name, func=func)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = weight_Dataload(test_benchmark, seq_test_out, domain_test_out, ppi_test_out, GOfile_name, func=func)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    weight_model.train()
    best_fscore = 0
    train_losses = []
    val_losses = []
    for epoch in range(epoch_times):
        _loss = 0
        batch_num = 0
        for batch_idx, (weight_features, GO_annotiations) in enumerate(train_data_loader):
            weight_features = Variable(weight_features).cuda()
            GO_annotiations = GO_annotiations.cuda() 
            out = weight_model(weight_features)
            optimizer.zero_grad()
            loss = loss_function(out, GO_annotiations)
            batch_num += 1
            loss.backward()
            optimizer.step()
            _loss += loss.item()
        epoch_train_loss = _loss / batch_num
        train_losses.append(epoch_train_loss)
        
        t_loss = 0
        test_batch_num = 0
        pred = []
        actual = []
        
        for idx, (weight_features, GO_annotiations) in enumerate(test_data_loader):
            weight_features = Variable(weight_features).cuda()
            GO_annotiations = Variable(GO_annotiations).cuda()
            out = weight_model(weight_features)
            test_batch_num += 1
            pred.append(out.data[0].cpu().tolist())
            actual.append(GO_annotiations.data[0].cpu().tolist())
            one_loss = loss_function(out, GO_annotiations)
            t_loss += one_loss.item()
        epoch_val_loss = t_loss / test_batch_num
        val_losses.append(epoch_val_loss)    
        test_loss = t_loss / test_batch_num

        fpr, tpr, th = roc_curve(np.array(actual).flatten(), np.array(pred).flatten(), pos_label=1)
        auc_score = auc(fpr, tpr)
        score_dict = {}
        each_best_fcore = 0
        each_best_scores = []
        for i in range(len(Thresholds)):
            f_score, recall, precision = calculate_performance(
                actual, pred, threshold=Thresholds[i], average='micro')
            if f_score >= each_best_fcore:
                each_best_fcore = f_score
                each_best_scores = [Thresholds[i], f_score, recall, precision, auc_score]
            scores = [f_score, recall, precision, auc_score]
            score_dict[Thresholds[i]] = scores
        if each_best_fcore >= best_fscore:
            best_fscore = each_best_fcore
            best_scores = each_best_scores
            best_score_dict = score_dict
            torch.save(weight_model,
                       'savedpkl/WeightVal_{}_{}_{}_{}.pkl'.format(func, batch_size, learning_rate, epoch_times))
        t, f_score, recall = each_best_scores[0], each_best_scores[1], each_best_scores[2]
        precision, auc_score = each_best_scores[3], each_best_scores[4]
        print('epoch{},loss{},testloss:{},t{},f_score{}, auc{}, recall{}, precision{}'.format(
            epoch, epoch_train_loss, test_loss, t, f_score, auc_score, recall, precision))
    bestthreshold, f_max, recall_max = best_scores[0], best_scores[1], best_scores[2]
    prec_max, bestauc_score = best_scores[3], best_scores[4]
    print('lr:{},batch:{},epoch{},f_max:{}\nauc{},recall_max{},prec_max{},threshold:{}'.format(
        learning_rate, batch_size, epoch_times,
        f_max, bestauc_score, recall_max, prec_max, bestthreshold))
    test_weight_model = torch.load(
        'savedpkl/WeightVal_{}_{}_{}_{}.pkl'.format(func, batch_size, learning_rate, epoch_times)).cuda()
    t_loss = 0
    weight_test_outs = {}
    pred = []
    actual = []
    pred_labels = []
    true_labels = []
    score_dict = {}
    batch_num = 0
    for batch_idx, (weight_features, GO_annotiations) in enumerate(test_data_loader):
        weight_features = Variable(weight_features).cuda()
        GO_annotiations = Variable(GO_annotiations).cuda()
        out = test_weight_model(weight_features)
        batch_num += 1
        weight_test_outs[test_benchmark[batch_idx]] = out.data[0].cpu().tolist()
        pred_labels.extend(torch.round(torch.sigmoid(out)).cpu().tolist())
        true_labels.extend(GO_annotiations.cpu().tolist())
        pred.append(out.data[0].cpu().tolist())
        actual.append(GO_annotiations.data[0].cpu().tolist())
        loss = loss_function(out, GO_annotiations)
        t_loss += loss.item()
    epoch_val_loss = t_loss / test_batch_num

    test_loss = t_loss / batch_num
    fpr, tpr, th = roc_curve(np.array(actual).flatten(), np.array(pred).flatten(), pos_label=1)
    auc_score = auc(fpr, tpr)
    aupr = cacul_aupr(np.array(actual).flatten(), np.array(pred).flatten())
    each_best_fcore = 0
    for i in range(len(Thresholds)):
        f_score, recall, precision = calculate_performance(
            actual, pred, threshold=Thresholds[i], average='micro')
        if f_score > each_best_fcore:
            each_best_fcore = f_score
            each_best_scores = [Thresholds[i], f_score, recall, precision, auc_score, aupr]
        scores = [f_score, recall, precision, auc_score]
        score_dict[Thresholds[i]] = scores
    bestthreshold, f_max, recall_max = each_best_scores[0], each_best_scores[1], each_best_scores[2]
    prec_max, bestauc_score, aupr_score = each_best_scores[3], each_best_scores[4], each_best_scores[5]
    print(f'Epoch {epoch}, Train Loss: {epoch_train_loss}, Validation Loss: {epoch_val_loss}')
    print('test_loss:{},lr:{},batch:{},epoch{},f_max:{}\nauc_score{},recall_max{},prec_max{},threshold:{}'.format(
        test_loss, learning_rate, batch_size, epoch_times,
        f_max, auc_score, recall_max, prec_max, bestthreshold))
    conf_matrices = compute_multilabel_confusion_matrix(true_labels, pred_labels)
    # Plot confusion matrix
     # Plot confusion matrix for each label
    for i, conf_mat in enumerate(conf_matrices):
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap="Blues")
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title(f'Confusion Matrix for Label {i}')
        plt.savefig('confusion_matrix.png')
        plt.clf()
        plt.show()
    # Save confusion matrix plot
    #plt.savefig('confusion_matrix.png')
    # Plot AUC-ROC curve
    fpr, tpr, _ = roc_curve(np.array(true_labels).flatten(), np.array(pred).flatten())
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('ROC.png')
    plt.clf()
    plt.show()
    
    return each_best_scores

def read_benchmark(term_arg='BP'):
    benchmark_file = '{}_benchmarkSetcomm_2.csv'.format(term_arg)
    print(benchmark_file)
    all_data = []
    with open(benchmark_file, 'r') as f:
        for line in f:
            item = line.strip()
            all_data.append(item)
    return all_data
def validation(func='BP', k_fold=5):
    kf = KFold(n_splits=k_fold)
    benchmark = np.array(read_benchmark(func))
    scores = []
    for train_index, test_index in kf.split(benchmark):
        train_set = benchmark[train_index].tolist()
        test_set = benchmark[test_index].tolist()
        each_fold_scores = Main(train_set, test_set, func=func)
        scores.append(each_fold_scores)
    f_maxs, pre_maxs, rec_maxs, auc_s, aupr_s = [], [], [], [], []
    for i in range(len(scores)):
        f_maxs.append(scores[i][1])
        rec_maxs.append(scores[i][2])
        pre_maxs.append(scores[i][3])
        auc_s.append(scores[i][4])
        aupr_s.append(scores[i][5])
    f_mean = np.mean(np.array(f_maxs))
    rec_mean = np.mean(np.array(rec_maxs))
    pre_mean = np.mean(np.array(pre_maxs))
    auc_mean = np.mean(np.array(auc_s))
    aupr_mean = np.mean(np.array(aupr_s))
    print('{}:f_mean{},rec_mean{},pre_mean{},auc_mean{}, aupr_mean{}'.format(
        func, f_mean, rec_mean, pre_mean, auc_mean, aupr_mean))
if __name__ == '__main__':
    Terms = ['BP', 'MF', 'CC']
    validation(Terms[0], 5)
    time_end = time.time()
    print('time cost', time_end - time_start,'s')

