# Code reused from https://github.com/jennyzhang0215/DKVMN
import numpy as np
import torch
import math
from sklearn import metrics

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def binaryEntropy(target, pred, mod="avg"):
    loss = target * np.log(np.maximum(1e-10, pred)) + \
           (1.0 - target) * np.log(np.maximum(1e-10, 1.0 - pred))
    if mod == 'avg':
        return np.average(loss) * (-1.0)
    elif mod == 'sum':
        return - loss.sum()
    else:
        assert False


def compute_auc(all_target, all_pred):
    # fpr, tpr, thresholds = metrics.roc_curve(all_target, all_pred, pos_label=1.0)
    return metrics.roc_auc_score(all_target, all_pred)


def compute_accuracy(all_target, all_pred):
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <= 0.5] = 0.0
    return metrics.accuracy_score(all_target, all_pred)


def train(net, params, optimizer, q_data, kc_data, a_data, q_dif_mask, qa_dif_mask, process):
    net.train()
    N = int(math.ceil(len(q_data) / params.batch_size))
    q_data = q_data.T  # Shape: (200,3633)
    kc_data = kc_data.T  # Shape: (200,3633)
    a_data = a_data.T
    # Shuffle the data
    shuffled_ind = np.arange(q_data.shape[1])
    np.random.shuffle(shuffled_ind)
    q_data = q_data[:, shuffled_ind]
    kc_data = kc_data[:, shuffled_ind]
    a_data = a_data[:, shuffled_ind]
    q_dif_mask = q_dif_mask[shuffled_ind, :, :]  # Shape: (3633,200,200)
    qa_dif_mask = qa_dif_mask[shuffled_ind, :, :]  # Shape: (3633,200,200)

    pred_list = []
    target_list = []

    element_count = 0
    true_el = 0
    p = process
    for idx in range(N):
        optimizer.zero_grad()

        process = [p, idx+1-N]

        # shape (bs, sql)
        q_one_seq = q_data[:, idx * params.batch_size:(idx + 1) * params.batch_size]
        kc_one_seq = kc_data[:, idx * params.batch_size:(idx + 1) * params.batch_size]
        a_one_seq = a_data[:, idx * params.batch_size:(idx + 1) * params.batch_size]
        # Shape (bs, sql, sql)
        q_dif_one_seq = q_dif_mask[idx * params.batch_size:(idx + 1) * params.batch_size, :, :]
        qa_dif_one_seq = qa_dif_mask[idx * params.batch_size:(idx + 1) * params.batch_size, :, :]

        input_q = np.transpose(q_one_seq[:, :])  # Shape (bs, seqlen)
        input_kc = np.transpose(kc_one_seq[:, :])  # Shape (bs, seqlen)
        input_a = np.transpose(a_one_seq[:, :])  # Shape (bs, seqlen)
        target_1 = np.where(input_kc > 0, input_a, -1)

        # shift to get the future answer
        target_shifted = np.zeros_like(target_1)
        target_shifted[:, :-1] = target_1[:, 1:]
        target_shifted[:, -1] = -1

        el = np.sum(target_shifted >= -.9)
        element_count += el

        input_q = torch.from_numpy(input_q).long().to(device)
        input_kc = torch.from_numpy(input_kc).long().to(device)
        input_a = torch.from_numpy(input_a).long().to(device)
        target = torch.from_numpy(target_shifted).float().to(device)
        input_q_dif = torch.from_numpy(q_dif_one_seq).float().to(device)
        input_qa_dif = torch.from_numpy(qa_dif_one_seq).float().to(device)

        loss, pred, true_ct = net(input_q, input_kc, input_a, target,
                                  input_q_dif, input_qa_dif)
        pred = pred.detach().cpu().numpy()  # (seqlen * batch_size, 1)
        loss.backward()
        true_el += true_ct.cpu().numpy()

        optimizer.step()

        # correct: 1.0; wrong 0.0; padding -1.0
        target = target_shifted.reshape((-1,))

        nopadding_index = np.flatnonzero(target >= -0.9)
        nopadding_index = nopadding_index.tolist()
        pred_nopadding = pred[nopadding_index]
        target_nopadding = target[nopadding_index]

        pred_list.append(pred_nopadding)
        target_list.append(target_nopadding)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    loss = binaryEntropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)

    return loss, accuracy, auc


def test(net, params, optimizer, q_data, kc_data, a_data, q_dif_mask, qa_dif_mask, process):
    # dataArray: [ array([[],[],..])] Shape: (3633, 200)
    net.eval()
    N = int(math.ceil(float(len(q_data)) / float(params.batch_size)))
    q_data = q_data.T  # Shape: (200,3633)
    kc_data = kc_data.T  # Shape: (200,3633)
    a_data = a_data.T

    seq_num = q_data.shape[1]
    pred_list = []
    target_list = []

    count = 0
    true_el = 0
    element_count = 0
    p = process
    for idx in range(N):
        process = [p, idx+1-N]

        # Shape (bs, sql)
        q_one_seq = q_data[:, idx * params.batch_size:(idx + 1) * params.batch_size]
        kc_one_seq = kc_data[:, idx * params.batch_size:(idx + 1) * params.batch_size]
        a_one_seq = a_data[:, idx * params.batch_size:(idx + 1) * params.batch_size]


        # Shape (bs, sql, sql)
        q_dif_one_seq = q_dif_mask[idx * params.batch_size:(idx + 1) * params.batch_size, :, :]
        qa_dif_one_seq = qa_dif_mask[idx * params.batch_size:(idx + 1) * params.batch_size, :, :]

        # Shape (seqlen, batch_size)
        input_q = np.transpose(q_one_seq[:, :])
        input_kc = np.transpose(kc_one_seq[:, :])
        input_a = np.transpose(a_one_seq[:, :])
        target_1 = np.where(input_kc > 0, input_a, -1)

        # shift to get the future answer
        target_shifted = np.zeros_like(target_1)
        target_shifted[:, :-1] = target_1[:, 1:]
        target_shifted[:, -1] = -1

        el = np.sum(target_1 >= -.9)
        element_count += el


        input_q = torch.from_numpy(input_q).long().to(device)
        input_kc = torch.from_numpy(input_kc).long().to(device)
        input_a = torch.from_numpy(input_a).long().to(device)
        target = torch.from_numpy(target_shifted).float().to(device)
        input_q_dif = torch.from_numpy(q_dif_one_seq).float().to(device)
        input_qa_dif = torch.from_numpy(qa_dif_one_seq).float().to(device)

        with torch.no_grad():
            loss, pred, ct = net(input_q, input_kc, input_a, target,
                                 input_q_dif, input_qa_dif)

        pred = pred.cpu().numpy()  # (seqlen * batch_size, 1)
        true_el += ct.cpu().numpy()
        # target = target.cpu().numpy()
        if (idx + 1) * params.batch_size > seq_num:
            real_batch_size = seq_num - idx * params.batch_size
            count += real_batch_size
        else:
            count += params.batch_size

        # correct: 1.0; wrong 0.0; padding -1.0
        target = target_shifted.reshape((-1,))
        nopadding_index = np.flatnonzero(target >= -0.9)
        nopadding_index = nopadding_index.tolist()
        pred_nopadding = pred[nopadding_index]
        target_nopadding = target[nopadding_index]

        element_count += pred_nopadding.shape[0]
        # print avg_loss
        pred_list.append(pred_nopadding)
        target_list.append(target_nopadding)

    assert count == seq_num, "Seq not matching"

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)
    loss = binaryEntropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)

    return loss, accuracy, auc
