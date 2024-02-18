import random
import numpy as np
import torch
def sgc_data_generator(features, labels, node_num, select_array, task_num, n_way, k_spt, k_qry):
    x_spt = []
    y_spt = []
    x_qry = []
    y_qry = []
    class1_idx = []
    class2_idx = []

    labels_local = labels.clone().detach()
    select_class = random.sample(select_array, n_way)

    for j in range(node_num):
        if (labels_local[j] == select_class[0]):
            class1_idx.append(j)
            labels_local[j] = 0
        elif (labels_local[j] == select_class[1]):
            class2_idx.append(j)
            labels_local[j] = 1

    for t in range(task_num):
        class1_train = random.sample(class1_idx, k_spt)
        class2_train = random.sample(class2_idx, k_spt)
        class1_test = [n1 for n1 in class1_idx if n1 not in class1_train]
        class2_test = [n2 for n2 in class2_idx if n2 not in class2_train]
        class1_test = random.sample(class1_test, k_qry)
        class2_test = random.sample(class2_test, k_qry)
        train_idx = class1_train + class2_train
        random.shuffle(train_idx)
        test_idx = class1_test + class2_test
        random.shuffle(test_idx)
        x_spt.append(features[train_idx])
        y_spt.append(labels_local[train_idx])
        x_qry.append(features[test_idx])
        y_qry.append(labels_local[test_idx])

    return x_spt, y_spt, x_qry, y_qry


def sgc_data_generator_ind(features, labels, node_num, select_array, task_num, n_way, k_spt, k_qry):
    x_spt = []
    y_spt = []
    x_qry = []
    y_qry = []
    class1_idx = []
    class2_idx = []

    labels_local = labels.clone().detach()
    select_class = random.sample(select_array, n_way)

    for j in range(node_num):
        if (labels_local[j] == select_class[0]):
            class1_idx.append(j)
            labels_local[j] = 0
        elif (labels_local[j] == select_class[1]):
            class2_idx.append(j)
            labels_local[j] = 1

    for t in range(task_num):
        class1_train = random.sample(class1_idx, k_spt)
        class2_train = random.sample(class2_idx, k_spt)
        class1_test = [n1 for n1 in class1_idx if n1 not in class1_train]
        class2_test = [n2 for n2 in class2_idx if n2 not in class2_train]
        class1_test = random.sample(class1_test, k_qry)
        class2_test = random.sample(class2_test, k_qry)
        train_idx = class1_train + class2_train
        random.shuffle(train_idx)
        test_idx = class1_test + class2_test
        random.shuffle(test_idx)
        x_spt.append(train_idx)
        y_spt.append(labels_local[train_idx])
        x_qry.append(test_idx)
        y_qry.append(labels_local[test_idx])

    return x_spt, y_spt, x_qry, y_qry


def get_class_ind(labels):
    labelscp = labels.clone().detach().cpu().numpy()
    class_idx = {}
    nodenum = labelscp.shape[0]
    for i in range(nodenum):

        nodeid = i
        nodelabel = labelscp[i]

        if nodelabel in class_idx:
            class_idx[nodelabel].append(nodeid)

        else:
            class_idx[nodelabel] = []
            class_idx[nodelabel].append(nodeid)

    return class_idx


def data_generator_ind(features, class_idx, node_num, select_array, task_num, n_way, k_spt, k_qry):
    x_spt = []
    y_spt = []
    x_qry = []
    y_qry = []

    for t in range(task_num):
        task_train = []
        task_test = []
        task_trainy = []
        task_testy = []
        ncls = np.random.choice(select_array, n_way, replace=False)
        for cid, c in enumerate(ncls):
            percls_ind = random.sample(class_idx[c], k_spt + k_qry)

            percls_sup = percls_ind[:k_spt]
            percls_qry = percls_ind[k_spt:]
            task_train.extend(percls_sup)
            task_test.extend(percls_qry)

            percls_supy = np.zeros((k_spt,))
            percls_supy.fill(cid)

            percls_qryy = np.zeros((k_qry, ))
            percls_qryy.fill(cid)

            task_trainy.extend(percls_supy)
            task_testy.extend(percls_qryy)

        task_train = np.array(task_train)
        task_test = np.array(task_test)
        task_trainy = np.array(task_trainy)
        task_testy = np.array(task_testy)

        randomidsup = np.random.permutation(n_way * k_spt)
        randomidqry = np.random.permutation(n_way * k_qry)

        task_train = task_train[randomidsup]
        task_trainy = torch.LongTensor(task_trainy[randomidsup]).to(features.device)

        task_test = task_test[randomidqry]
        task_testy =torch.LongTensor( task_testy[randomidqry]).to(features.device)

        x_spt.append(task_train)
        y_spt.append(task_trainy)
        x_qry.append(task_test)
        y_qry.append(task_testy)

    return x_spt, y_spt, x_qry, y_qry


