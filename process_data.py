#import pickle
import pickle
import argparse
import numpy as np
import torch

#-data_seq_dir .data/total.seqs -data_label_dir ./data/total.morts -data_save_dir ./processed_data/processed_data
def main():


    parser = argparse.ArgumentParser()
    parser.add_argument('-data_seq_dir', required=True)
    parser.add_argument('-data_label_dir', required=True)
    parser.add_argument('-data_save_dir', required=True)
    opt = parser.parse_args()



    print('===> convert total data to train val test dataset')
    with open('./data/total.seqs', 'rb') as f:
        total_seqs = pickle.load(f)

    with open('./data/total.morts', 'rb') as f:
        total_label = pickle.load(f)

    #
    # max_code = max(map(lambda p: max(map(lambda v: max(v), p)), total_seqs))
    # num_features = max_code + 1

    temp_seqs = total_seqs
    total_seqs = []
    for patient in temp_seqs:
        temp_patient = []
        for visits in patient:
            for dignosis in visits:
                temp_patient.append(dignosis+1)
        total_seqs.append(temp_patient)

    max_dig_len = 0
    for patient in total_seqs:
        if len(patient) > max_dig_len:
            max_dig_len = len(patient)
            print(max_dig_len)

    print(max_dig_len)

    types = []
    for patient in total_seqs:
        for dia in patient:
            if dia not in types:
                types.append(dia)


    src_dia_size = len(types)

    a = int(len(total_seqs)*0.75)
    b = int(len(total_seqs)*0.9)
    c = int(len(total_seqs))
    print(a,b,c)
    train_seqs = total_seqs[:a]
    train_labels = total_label[:a]

    val_seqs = total_seqs[a:b]
    val_labels = total_label[a:b]

    test_seqs = total_seqs[b:]
    test_labels = total_label[b:]

    #processed_dir = './processed_data/'


    opt.max_dia_seq_len = max_dig_len
    opt.src_dia_size = src_dia_size
    data = {
        'settings': opt,
        'train':{
            'src': train_seqs,
            'tgt': train_labels
        },
        'test': {
            'src': test_seqs,
            'tgt': test_labels
        },
        'valid':{
            'src': val_seqs,
            'tgt': val_labels
        }
    }

    print('[Info] Dumpling the processed data to pickle file', opt.data_save_dir)
    torch.save(data, opt.data_save_dir)
    print('[Info] Finish')
    # pickle.dump(train_seqs, open(processed_dir+'train.seqs', 'wb'), -1)
    # pickle.dump(train_labels, open(processed_dir+'train.labels', 'wb'), -1)
    #
    # pickle.dump(val_seqs, open(processed_dir+'valid.seqs', 'wb'), -1)
    # pickle.dump(val_labels, open(processed_dir+'valid.labels', 'wb'), -1)
    #
    # pickle.dump(test_seqs, open(processed_dir+'test.seqs', 'wb'), -1)
    # pickle.dump(test_labels, open(processed_dir+'test.labels', 'wb'), -1)


if __name__ == '__main__':
    main()