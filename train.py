import argparse

import math
import time
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from aidoc.Utils import VisitSequenceWithLabelDataset
from aidoc.Models import Doctor
from aidoc.Optim import ScheduledOptim
#-data processed_data/processed_data -save_model trained -save_mode best -proj_share_weight

Count = 0
# @profile
def visit_collate_fn(batch):
    batch_seq, batch_label = zip(*batch)
    max_len = 498

    batch_seq_data = np.array([inst + [0] * (max_len - len(inst)) for inst in batch_seq])
    batch_seq_position = np.array([ [pos_i+1 if w_i!=0 else 0 for pos_i, w_i in enumerate(inst)] for inst in batch_seq_data] )
    batch_seq_data_tensor = Variable(torch.LongTensor(batch_seq_data))
    batch_seq_position_tensor = Variable(torch.LongTensor(batch_seq_position))

    batch_labels = np.array( [[inst] for inst in batch_label])
    batch_labels_tensor = Variable(torch.LongTensor(batch_labels))
    length = np.array([len(inst) for inst in batch_seq])
    return (batch_seq_data_tensor, batch_seq_position_tensor), batch_labels_tensor, length



def get_performance(crit, pred, tgt, smoothing=False, num_class=None):
    # TODO: Add smoothing
    if smoothing:
        assert bool(num_class)
        eps = 0.1
        gold = tgt * (1 - eps) + (1 - tgt) * eps / num_class
        raise NotImplementedError
    tgt = tgt.squeeze(1)
    loss = crit(pred, tgt)
    return loss

def train_epoch(model, training_data, crit, optimizer, opt):
    model.train()

    total_loss = 0
    total_patient = 0

    labels = []
    outputs = []
    # for batch in tqdm(training_data, mininterval=2, desc=' -(Training) ', leave=False):
    # for batch in training_data:
    for bi, batch in enumerate(tqdm(training_data, desc="batches".format('training'), leave=False)):
        src, tgt, length = batch

        if opt.cuda:
            tgt = tgt.cuda()

        #========forward=======#
        optimizer.zero_grad()
        pred = model(src, tgt, length)

        #========backward======#
        loss = get_performance(crit, pred, tgt)
        loss.backward()

        #========update parameters=======#
        optimizer.step()
        optimizer.update_learning_rate()

        #========note loss==========#
        total_loss += loss.data[0]
        total_patient += tgt.size()[0]

        #========note acu=======#
        outputs.append(F.softmax(pred).data)
        labels.append(tgt.data)
        predict = torch.cat(outputs,0)
        labels_true = torch.cat(labels,0)
        if opt.cuda:
            predict = predict.cpu()
            labels_true = labels_true.cpu()


        acc = roc_auc_score(labels_true.numpy(), predict.numpy()[:,1], average='weighted')
        print('---interator{}---training loss:{loss:5.3f}---traing accurcy:{acc:5.3f}'.format(bi,loss=total_loss/total_patient, acc=acc))



    return total_loss/total_patient, acc

def eval_epoch(model, validation_data, crit, opt):
    model.eval()

    total_loss = 0
    total_patient = 0
    n_total_correct = 0
    labels = []
    outputs = []
    for bi, batch in enumerate(tqdm(validation_data, desc="batches".format('validing'), leave=False)):
        #===prepare data===#
        src, tgt, length = batch
        if opt.cuda:
            tgt = tgt.cuda()
        #===forward===#
        pred = model(src, tgt, length)

        loss = get_performance(crit, pred, tgt)

        #===note loss===#
        total_loss += loss.data[0]
        total_patient += tgt.size()[0]
        #===note acu===#

        outputs.append(F.softmax(pred).data)
        labels.append(tgt.data)
        predict = torch.cat(outputs, 0)
        labels_true = torch.cat(labels, 0)
        if opt.cuda:
            predict = predict.cpu()
            labels_true = labels_true.cpu()
        acc = roc_auc_score(labels_true.numpy(), predict.numpy()[:, 1], average='weighted')



    return total_loss/total_patient,acc


def train(model, training_data, validation_data, crit, optimizer, opt):
    log_train_file = None
    log_valid_file = None
    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_valid_file = opt.log + '.valid.log'
        print('[Info] Training performance will be written to file:{} and {}'.format(log_train_file,log_valid_file))
        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    valid_accus = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')
        start = time.time()

        train_loss, train_accu = train_epoch(model, training_data, crit, optimizer, opt)
        print('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, ' \
              'elapse: {elapse:3.3f} min'.format(
            ppl=math.exp(min(train_loss, 100)), accu=100 * train_accu,
            elapse=(time.time() - start) / 60))

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, crit, opt)
        print('  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, ' \
              'elapse: {elapse:3.3f} min'.format(
            ppl=math.exp(min(valid_loss, 100)), accu=100 * valid_accu,
            elapse=(time.time() - start) / 60))

        valid_accus += [valid_accu]
        model_state_dict = model.state_dict()

        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i
        }

        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.save_model + '.chkpt'
                if valid_accu >= max(valid_accus):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', required=True)

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=4)

    # parser.add_argument('-d_word_vec', type=int, default=512)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=1024)
    parser.add_argument('-d_k', type=int, default=512)
    parser.add_argument('-d_v', type=int, default=512)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model
    print(opt)

    #=======Loading Dataset=======#
    data = torch.load(opt.data)
    opt.max_dia_seq_len = data['settings'].max_dia_seq_len



    #=======Custruct train set======#
    train_set = VisitSequenceWithLabelDataset(data['train']['src'], data['train']['tgt'])
    val_set = VisitSequenceWithLabelDataset(data['valid']['src'],data['valid']['tgt'])
    #=======Preparing Dataloader =======#
    training_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size,collate_fn=visit_collate_fn, shuffle=True)
    validation_loader = DataLoader(dataset=val_set, batch_size=opt.batch_size,collate_fn=visit_collate_fn, shuffle=True)
    opt.src_dia_size = data['settings'].src_dia_size

    #=======Preparing Model=========#
    doctor = Doctor(
        opt.src_dia_size,
        opt.max_dia_seq_len,
        proj_share_weight=opt.proj_share_weight,
        embs_share_weight=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner_hid=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout,
        cudas = opt.cuda,
        batch_size = opt.batch_size,
        max_length = opt.max_dia_seq_len
    )

    # #=======Preparing Optimizer======#

    optimizer = ScheduledOptim(
        optim.Adam(
            doctor.get_trainable_parameters(),
            betas=(0.9, 0.98), eps=1e-09
        ), opt.d_model, opt.n_warmup_steps)

    criterion = nn.CrossEntropyLoss()
    if opt.cuda:
        doctor = doctor.cuda()
        criterion = criterion.cuda()
    train(doctor, training_loader, validation_loader, criterion, optimizer, opt)

if __name__ == '__main__':
    main()
