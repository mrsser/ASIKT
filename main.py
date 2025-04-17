# Code reused from https://github.com/arghosh/AKT
import os
import os.path
import glob
import argparse
import numpy as np
import torch
from data_pre import preprocess
from run import train, test
from ASIKT import ASIKT
from run import device
import time


def try_makedirs(path_):
    if not os.path.isdir(path_):
        try:
            os.makedirs(path_)
        except FileExistsError:
            pass


def get_file_name_identifier(params):
    file_name = [['_b', params.batch_size], ['_nb', params.n_block], ['_lr', params.lr],
                 ['_s', params.seed], ['_sl', params.seqlen], ['_do', params.dropout], ['_dm', params.d_model],
                 ['_ts', params.train_set]]
    return file_name


def load_model(params):
    model = ASIKT(n_kc=params.n_kc, n_q=params.n_q, d_model=params.d_model, n_blocks=params.n_block,
                    rasch=params.rasch, dropout=params.dropout, d_ff=params.d_ff).to(device)
    return model



def train_one_dataset(params, file_name,
                      train_q_data, train_kc_data, train_a_data,
                      train_data_q_dif_mask, train_data_qa_dif_mask,
                      valid_q_data, valid_kc_data, valid_a_data,
                      valid_data_q_dif_mask, valid_data_qa_dif_mask):

    model = load_model(params)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params.lr, betas=(0.9, 0.999), eps=1e-8)

    print("\n")

    all_train_loss = {}
    all_train_accuracy = {}
    all_train_auc = {}
    all_valid_loss = {}
    all_valid_accuracy = {}
    all_valid_auc = {}
    best_valid_auc = 0

    for idx in range(params.max_iter):
        # Train Model
        start_time = time.time()
        train_loss, train_accuracy, train_auc = train(
            model, params, optimizer,
            train_q_data, train_kc_data, train_a_data,
            train_data_q_dif_mask, train_data_qa_dif_mask,
            process='train'
        )
        end_time = time.time()
        epoch_time = end_time - start_time

        # Validation step
        valid_loss, valid_accuracy, valid_auc = test(
            model, params, optimizer,
            valid_q_data, valid_kc_data, valid_a_data,
            valid_data_q_dif_mask, valid_data_qa_dif_mask,
            process='valid'
        )

        print(f"Epoch {idx + 1}")
        print("valid_auc\t", valid_auc, "\ttrain_auc\t", train_auc)
        print("valid_accuracy\t", valid_accuracy,
              "\ttrain_accuracy\t", train_accuracy)
        print("valid_loss\t", valid_loss, "\ttrain_loss\t", train_loss)
        print(f"training time: {epoch_time:.2f} seconds")

        try_makedirs('model')
        try_makedirs(os.path.join('model', params.model))
        try_makedirs(os.path.join('model', params.model, params.save))

        all_valid_auc[idx + 1] = valid_auc
        all_train_auc[idx + 1] = train_auc
        all_valid_loss[idx + 1] = valid_loss
        all_train_loss[idx + 1] = train_loss
        all_valid_accuracy[idx + 1] = valid_accuracy
        all_train_accuracy[idx + 1] = train_accuracy

        # output the epoch with the best validation auc
        if valid_auc > best_valid_auc:
            path = os.path.join('model', params.model,
                                params.save, file_name) + '_*'
            for i in glob.glob(path):
                os.remove(i)
            best_valid_auc = valid_auc
            best_epoch = idx + 1
            torch.save({'epoch': idx,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_loss,
                        },
                       os.path.join('model', params.model, params.save,
                                    file_name) + '_' + str(idx + 1)
                       )
        if idx - best_epoch > 5:
            break

    try_makedirs('result')
    try_makedirs(os.path.join('result', params.model))
    try_makedirs(os.path.join('result', params.model, params.save))
    f_save_log = open(os.path.join(
        'result', params.model, params.save, file_name), 'w')
    f_save_log.write("valid_auc:\n" + str(all_valid_auc) + "\n\n")
    f_save_log.write("train_auc:\n" + str(all_train_auc) + "\n\n")
    f_save_log.write("valid_loss:\n" + str(all_valid_loss) + "\n\n")
    f_save_log.write("train_loss:\n" + str(all_train_loss) + "\n\n")
    f_save_log.write("valid_accuracy:\n" + str(all_valid_accuracy) + "\n\n")
    f_save_log.write("train_accuracy:\n" + str(all_train_accuracy) + "\n\n")
    f_save_log.close()
    return best_epoch


def test_one_dataset(params, file_name,
                     test_q_data, test_kc_data, test_a_data,
                     test_data_q_dif_mask, test_data_qa_dif_mask,
                     best_epoch):
    print("\n\nStart testing ......................\n Best epoch:", best_epoch)
    model = load_model(params)

    checkpoint = torch.load(os.path.join(
        'model', params.model, params.save, file_name) + '_' + str(best_epoch))
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_accuracy, test_auc = test(
        model, params, None,
        test_q_data, test_kc_data, test_a_data,
        test_data_q_dif_mask, test_data_qa_dif_mask,
        process='test'
    )
    print("\ntest_auc\t", test_auc)
    print("test_accuracy\t", test_accuracy)
    print("test_loss\t", test_loss)

    # Now Delete all the models
    path = os.path.join('model', params.model, params.save, file_name) + '_*'
    for i in glob.glob(path):
        os.remove(i)


if __name__ == '__main__':

    # Parse Arguments
    parser = argparse.ArgumentParser(description='Script to test KT')
    # Basic Parameters
    parser.add_argument('--max_iter', type=int, default=300,
                        help='number of iterations')
    parser.add_argument('--train_set', type=int, default=1)
    parser.add_argument('--seed', type=int, default=224, help='default seed')

    # Common parameters
    parser.add_argument('--optim', type=str, default='adam',
                        help='Default Optimizer')
    parser.add_argument('--batch_size', type=int,
                        default=48, help='the batch size')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--final_fc_dim', type=int, default=512,
                        help='hidden state dim for final fc layer')

    # Specific Parameter
    parser.add_argument('--d_model', type=int, default=128,
                        help='Transformer d_model shape')
    parser.add_argument('--d_ff', type=int, default=1024,
                        help='Transformer d_ff shape')
    parser.add_argument('--dropout', type=float,
                        default=0.2, help='Dropout rate')
    parser.add_argument('--n_block', type=int, default=2,
                        help='number of blocks')
    parser.add_argument('--n_head', type=int, default=8,
                        help='number of heads in multihead attention')

    # Datasets and Model
    parser.add_argument('--model', type=str, default='ASIKT',
                        help="model type")
    parser.add_argument('--dataset', type=str, default="assist2009")

    params = parser.parse_args()
    dataset = params.dataset

    if dataset in {"assist2009"}:
        params.seqlen = 200
        params.data_dir = 'data/' + dataset
        params.data_name = dataset
        params.n_kc = 107
        params.n_q = 9798
        params.rasch = 1


    if dataset in {"assist2017"}:
        params.seqlen = 200
        params.data_dir = 'data/' + dataset
        params.data_name = dataset
        params.n_kc = 97
        params.n_q = 2521
        params.rasch = 0


    if dataset in {"assist2012"}:
        params.seqlen = 200
        params.data_dir = 'data/' + dataset
        params.data_name = dataset
        params.n_kc = 254
        params.n_q = 37438
        params.rasch = 1


    if dataset in {"eedi"}:
        params.seqlen = 200
        params.data_dir = 'data/' + dataset
        params.data_name = dataset
        params.n_kc = 52
        params.n_q = 915
        params.rasch = 0

    params.save = params.data_name
    params.load = params.data_name

    # Setup
    dat = preprocess(seqlen=params.seqlen, separate_char=',')
    seedNum = params.seed
    np.random.seed(seedNum)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seedNum)
    np.random.seed(seedNum)
    file_name_identifier = get_file_name_identifier(params)

    ###Train- Test
    d = vars(params)
    for key in d:
        print('\t', key, '\t', d[key])
    file_name = ''
    for item_ in file_name_identifier:
        file_name = file_name + item_[0] + str(item_[1])

    train_data_path = params.data_dir + "/" + \
                      params.data_name + "_train" + str(params.train_set) + ".csv"
    valid_data_path = params.data_dir + "/" + \
                      params.data_name + "_valid" + str(params.train_set) + ".csv"

    train_data_q_dif_mask = np.load(params.data_dir + "/" + \
                                    params.data_name + "_train" + str(params.train_set) + "_q_mask.npy")
    train_data_qa_dif_mask = np.load(params.data_dir + "/" + \
                                     params.data_name + "_train" + str(params.train_set) + "_qa_mask.npy")

    valid_data_q_dif_mask = np.load(params.data_dir + "/" + \
                                    params.data_name + "_valid" + str(params.train_set) + "_q_mask.npy")
    valid_data_qa_dif_mask = np.load(params.data_dir + "/" + \
                                     params.data_name + "_valid" + str(params.train_set) + "_qa_mask.npy")

    train_kc_data, train_a_data, train_q_data = dat.load_data(train_data_path)
    valid_kc_data, valid_a_data, valid_q_data = dat.load_data(valid_data_path)

    test_data_path = params.data_dir + "/" + \
                     params.data_name + "_test" + str(params.train_set) + ".csv"
    test_data_q_dif_mask = np.load(params.data_dir + "/" + \
                                   params.data_name + "_test" + str(params.train_set) + "_q_mask.npy")
    test_data_qa_dif_mask = np.load(params.data_dir + "/" + \
                                    params.data_name + "_test" + str(params.train_set) + "_qa_mask.npy")
    test_kc_data, test_a_data, test_q_data = dat.load_data(test_data_path)

    print("\n")
    print("train_q_data.shape", train_q_data.shape)
    print("train_kc_data.shape", train_kc_data.shape)
    print("train_a_data.shape", train_a_data.shape)
    print("valid_q_data.shape", valid_q_data.shape)
    print("valid_kc_data.shape", valid_kc_data.shape)
    print("valid_a_data.shape", valid_a_data.shape)
    print("test_q_data.shape", test_q_data.shape)
    print("test_kc_data.shape", test_kc_data.shape)
    print("test_a_data.shape", test_a_data.shape)
    print("\n")

    # Train and get the best episode
    best_epoch = train_one_dataset(
        params, file_name,
        train_q_data, train_kc_data, train_a_data,
        train_data_q_dif_mask, train_data_qa_dif_mask,
        valid_q_data, valid_kc_data, valid_a_data,
        valid_data_q_dif_mask, valid_data_qa_dif_mask)

    test_one_dataset(params, file_name,
                     test_q_data, test_kc_data, test_a_data,
                     test_data_q_dif_mask, test_data_qa_dif_mask,
                     best_epoch)
