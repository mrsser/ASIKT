import numpy as np
from tqdm import tqdm
import math


class preprocess(object):
    def __init__(self, seqlen, separate_char):
        self.separate_char = separate_char
        self.seqlen = seqlen

    # data format
    # id, true_student_id
    # question1, question2, ...
    # kc1, kc2, ...
    # answer1, answer2, ...

    def load_data(self, path):
        f_data = open(path, 'r')
        q_data = []
        a_data = []
        p_data = []
        for lineID, line in enumerate(f_data):
            line = line.strip()
            # lineID starts from 0
            if lineID % 4 == 0:
                student_id = lineID // 4
            if lineID % 4 == 2:
                Q = line.split(self.separate_char)
                if len(Q[len(Q) - 1]) == 0:
                    Q = Q[:-1]
                # print(len(Q))
            if lineID % 4 == 1:
                P = line.split(self.separate_char)
                if len(P[len(P) - 1]) == 0:
                    P = P[:-1]

            elif lineID % 4 == 3:
                A = line.split(self.separate_char)
                if len(A[len(A) - 1]) == 0:
                    A = A[:-1]
                # print(len(A),A)

                # start split the data
                n_split = 1
                # print('len(Q):',len(Q))
                if len(Q) > self.seqlen:
                    n_split = math.floor(len(Q) / self.seqlen)
                    if len(Q) % self.seqlen:
                        n_split = n_split + 1
                # print('n_split:',n_split)
                for k in range(n_split):
                    question_sequence = []
                    problem_sequence = []
                    answer_sequence = []
                    if k == n_split - 1:
                        endINdex = len(A)
                    else:
                        endINdex = (k + 1) * self.seqlen
                    for i in range(k * self.seqlen, endINdex):
                        if len(Q[i]) > 0:
                            question_sequence.append(int(Q[i]))
                            problem_sequence.append(int(P[i]))
                            answer_sequence.append(int(float(A[i])))
                        else:
                            print(Q[i])
                    q_data.append(question_sequence)
                    a_data.append(answer_sequence)
                    p_data.append(problem_sequence)

        f_data.close()
        # data: [[],[],[],...] <-- set_max_seqlen is used
        # convert data into ndarrays for better speed during training
        q_dataArray = np.zeros((len(q_data), self.seqlen))
        for j in range(len(q_data)):
            dat = q_data[j]
            q_dataArray[j, :len(dat)] = dat

        a_dataArray = np.zeros((len(a_data), self.seqlen))
        for j in range(len(a_data)):
            dat = a_data[j]
            a_dataArray[j, :len(dat)] = dat

        p_dataArray = np.zeros((len(p_data), self.seqlen))
        for j in range(len(p_data)):
            dat = p_data[j]
            p_dataArray[j, :len(dat)] = dat
        return q_dataArray, a_dataArray, p_dataArray


def global_feature_kc(kc_data, a_data):
    right_dict = {}
    error_dict = {}
    acc_dict = {}
    freq_dict = {}
    Bacc_dict = {}
    for i, q_list in enumerate(kc_data):
        for j, q in enumerate(q_list):
            answer = a_data[i][j]
            acc_dict[str(q)] = 0
            if str(q) not in right_dict:
                right_dict[str(q)] = 0
            if str(q) not in error_dict:
                error_dict[str(q)] = 0
            if answer == 1:
                right_dict[str(q)] += 1
            else:
                error_dict[str(q)] += 1
    del acc_dict['0.0']
    answer_time = 0
    total_acc = 0
    for key in acc_dict.keys():
        answer_time += (right_dict[key] + error_dict[key])
        freq_dict[key] = right_dict[key] + error_dict[key]
        acc = right_dict[key] / (right_dict[key] + error_dict[key])
        acc_dict[key] = acc
        total_acc += acc
    average_answer_time = answer_time / len(acc_dict)
    average_acc = total_acc / len(acc_dict)
    for key in acc_dict.keys():
        acc = (freq_dict[key] * acc_dict[key] + average_answer_time * average_acc) / (
                freq_dict[key] + average_answer_time)
        Bacc_dict[key] = acc
    return acc_dict, Bacc_dict, freq_dict, average_answer_time, average_acc


def global_feature_q(q_data, a_data):
    right_dict = {}
    error_dict = {}
    acc_dict = {}
    freq_dict = {}
    Bacc_dict = {}
    for i, q_list in enumerate(q_data):
        for j, q in enumerate(q_list):
            answer = a_data[i][j]
            acc_dict[str(q)] = 0
            if str(q) not in right_dict:
                right_dict[str(q)] = 0
            if str(q) not in error_dict:
                error_dict[str(q)] = 0
            if answer == 1:
                right_dict[str(q)] += 1
            else:
                error_dict[str(q)] += 1
    del acc_dict['0.0']
    answer_time = 0
    total_acc = 0
    for key in acc_dict.keys():
        answer_time += (right_dict[key] + error_dict[key])
        freq_dict[key] = right_dict[key] + error_dict[key]
        acc = right_dict[key] / (right_dict[key] + error_dict[key])
        acc_dict[key] = acc
        total_acc += acc
    average_answer_time = answer_time / len(acc_dict)
    average_acc = total_acc / len(acc_dict)
    for key in acc_dict.keys():
        acc = (freq_dict[key] * acc_dict[key] + average_answer_time * average_acc) / (
                freq_dict[key] + average_answer_time)
        Bacc_dict[key] = acc
    return acc_dict, Bacc_dict, freq_dict, average_answer_time, average_acc


def KC_accuracy(kc_data, a_data, freq_dict, Bacc_dict):
    kc_acc_data = []
    for i, kc_list in enumerate(kc_data):
        acc_list = []
        right_dict = {}
        error_dict = {}
        for j, kc in enumerate(kc_list):
            if str(kc) != '0.0' and j == 0:
                acc_list.append(Bacc_dict[str(kc)])
            elif str(kc) != '0.0' and j != 0:
                answer = a_data[i][j-1]
                if str(kc) not in right_dict:
                    right_dict[str(kc)] = 0
                if str(kc) not in error_dict:
                    error_dict[str(kc)] = 0
                if answer == 1:
                    right_dict[str(kc)] += 1
                else:
                    error_dict[str(kc)] += 1
                kc_freq = right_dict[str(kc)] + error_dict[str(kc)]
                kc_acc = right_dict[str(kc)] / kc_freq
                acc = (kc_freq * kc_acc + freq_dict[str(kc)] * Bacc_dict[str(kc)]) / (kc_freq + freq_dict[str(kc)])
                acc_list.append(acc)
            else:
                acc_list.append(0)
        kc_acc_data.append(acc_list)
    return kc_acc_data


def Q_accuracy(q_data, a_data, freq_dict, Bacc_dict):
    q_acc_data = []
    for i, q_list in enumerate(q_data):
        acc_list = []
        right_dict = {}
        error_dict = {}
        for j, q in enumerate(q_list):
            if str(q) != '0.0' and j == 0:
                acc_list.append(Bacc_dict[str(q)])
            elif str(q) != '0.0' and j != 0:
                answer = a_data[i][j-1]
                if str(q) not in right_dict:
                    right_dict[str(q)] = 0
                if str(q) not in error_dict:
                    error_dict[str(q)] = 0
                if answer == 1:
                    right_dict[str(q)] += 1
                else:
                    error_dict[str(q)] += 1
                q_freq = right_dict[str(q)] + error_dict[str(q)]
                kc_acc = right_dict[str(q)] / q_freq
                acc = (q_freq * kc_acc + freq_dict[str(q)] * Bacc_dict[str(q)]) / (q_freq + freq_dict[str(q)])
                acc_list.append(acc)
            else:
                acc_list.append(0)
        q_acc_data.append(acc_list)
    return q_acc_data


def Q_mask(q_data, kc_data, a_data, q_acc_data, kc_acc_data, theta):
    print('creating mask')
    q_weights = []
    qa_weights = []
    for i, q_list in tqdm(enumerate(q_data), total=len(q_data)):
        q_local_weights = np.zeros((len(q_list), len(q_list)))
        qa_local_weights = np.zeros((len(q_list), len(q_list)))
        for j, q_t in enumerate(q_list):
            answer_t = a_data[i][j]
            kc_q = kc_data[i][j]
            if float(kc_q) > 0:
                w_t = (1 - q_acc_data[i][j])
            else:
                w_t = 0
            q_local_weights[j][j] = w_t * w_t * np.exp(1 - kc_acc_data[i][j])
            qa_local_weights[j][j] = w_t * w_t * np.exp(1 - kc_acc_data[i][j])
            for k, q_tao in enumerate(q_list[:j]):
                answer_tao = a_data[i][k]
                kc_tao = kc_data[i][k]
                w_tao = (1 - q_acc_data[i][k])
                if kc_q == kc_tao:
                    w_q_t_tao = w_t * w_tao * np.exp(1 - kc_acc_data[i][k])
                    w_qa_t_tao = w_q_t_tao
                else:
                    w_q_t_tao = w_t * w_tao
                    if answer_t == answer_tao:
                        w_qa_t_tao = w_t * w_tao
                    else:
                        w_qa_t_tao = w_t * w_tao * theta

                q_local_weights[j][k] = w_q_t_tao
                qa_local_weights[j][k] = w_qa_t_tao
        q_weights.append(q_local_weights)
        qa_weights.append(qa_local_weights)
    return np.array(q_weights), np.array(qa_weights)


if __name__ == '__main__':
    dataset = 'assist2009'
    seqlen = 200
    data_dir = 'data/' + dataset
    data_name = dataset
    theta = 0.5
    dat = preprocess(seqlen=seqlen, separate_char=',')

    data_path = data_dir + "/" + data_name + ".csv"
    total_kc_data, total_a_data, total_q_data = dat.load_data(data_path)
    kc_acc_dict, kc_Bacc_dict, kc_freq_dict, kc_average_answer_time, kc_average_acc = global_feature_kc(total_kc_data, total_a_data)
    q_acc_dict, q_Bacc_dict, q_freq_dict, q_average_answer_time, q_average_acc = global_feature_q(total_q_data, total_a_data)

    # for data_id in [1, 2, 3, 4, 5]:
    for data_id in [1]:
        for mission in ['train', 'valid', 'test']:
            print('processing data' + str(data_id) + ':' + mission)
            data_path = data_dir + "/" + data_name + "_" + mission + str(data_id) + ".csv"
            kc_data, a_data, q_data = dat.load_data(data_path)
            kc_acc_data = KC_accuracy(kc_data, a_data, kc_freq_dict, kc_Bacc_dict)
            q_acc_data = Q_accuracy(q_data, a_data, q_freq_dict, q_Bacc_dict)

            q_mask, qa_mask = Q_mask(q_data, kc_data, a_data, q_acc_data, kc_acc_data, theta)
            np.save(data_dir + "/" + data_name + "_" + mission + str(data_id) + "_q_mask.npy", q_mask)
            np.save(data_dir + "/" + data_name + "_" + mission + str(data_id) + "_qa_mask.npy", qa_mask)



