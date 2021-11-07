import numpy as np
from datetime import datetime
import csv
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch


def seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def plot(results, log_results, prob, num_gt):
    results = np.array(results)
    results = pd.DataFrame(
        results, columns=['episode', 'reward', 'agent_loss', 'time'])
    results.to_csv(log_results + "/episodes.csv", index=False)
    prob_df = pd.DataFrame.from_dict(prob)
    prob_df.to_csv(log_results + "/prob.csv", index=False)

    # Visualization
    sns.lineplot(data=results.reward/num_gt, color="g")
    plt.legend(labels=["Reward"])
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(log_results + '/rewards.png')

def normalize(emb):
    return emb / np.sqrt((emb ** 2).sum(axis=1)).reshape(-1, 1)

def save_results(results_list, path_log):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    results_list.insert(0, dt_string)
    with open(path_log, 'a') as file:
        writer = csv.writer(file)
        writer.writerow(results_list)


def normalize_prob(prob):
    max_len = max([len(v) for _, v in prob.items()])
    for k, v in prob.items():
        # if prob array of one paired state is not equal (due to skip rate)
        if len(v) != max_len:
            prob[k].append(prob[k][-1]) # append the last prob
    return prob


def build_adj_matrix_and_embeddings(is_training):
    # Build adj matrix
    file1 = open('./data/IKAMI/D_W_15K_V2/rel_triples_1.txt', 'r')
    lines1 = file1.readlines()
    lines1 = lines1[:500]
    g1 = []
    for line in lines1:
        g1.append([int(x) for x in line.strip().split("\t")])
    g1 = np.array(g1)

    file2 = open('./data/IKAMI/D_W_15K_V2/rel_triples_2.txt', 'r')
    lines2 = file2.readlines()
    lines2 = lines2[:500]
    g2 = []
    for line in lines2:
        g2.append([int(x) for x in line.strip().split("\t")])
    g2 = np.array(g2)

    mapping_index_1 = {}
    mapping_index_2 = {}
    unique1 = np.unique(g1)
    unique2 = np.unique(g2)
    for i in range(len(unique1)):
        mapping_index_1[unique1[i]] = i

    for i in range(len(unique2)):
        mapping_index_2[unique2[i]] = i

    G1_nodes = len(np.unique(g1))
    G2_nodes = len(np.unique(g2))

    G1_adj_matrix = np.zeros(shape=(G1_nodes, G1_nodes))
    G2_adj_matrix = np.zeros(shape=(G2_nodes, G2_nodes))

    for i in range(len(g1)):
        head = g1[i][0]
        tail = g1[i][1]
        G1_adj_matrix[mapping_index_1[head]][mapping_index_1[tail]] = 1
        G1_adj_matrix[mapping_index_1[tail]][mapping_index_1[head]] = 1
    for i in range(len(g1)):
        head = g2[i][0]
        tail = g2[i][1]
        G2_adj_matrix[mapping_index_2[head]][mapping_index_2[tail]] = 1
        G2_adj_matrix[mapping_index_2[tail]][mapping_index_2[head]] = 1

    # Build embeddings
    emb1 = []
    emb2 = []
    proximi = np.load("./data/IKAMI/D_W_15K_V2/_proximity.npy")
    for i, _ in enumerate(mapping_index_1):
        emb1.append(proximi[i])
    for i, _ in enumerate(mapping_index_2):
        emb2.append(proximi[i])
    emb1 = np.array(emb1)
    emb2 = np.array(emb2)

    # Get ground truth
    tmp_ground_truth = {}
    file_gt = open('./data/IKAMI/D_W_15K_V2/ground_truth.txt', 'r')
    lines = file_gt.readlines()
    gt = []
    for line in lines:
        gt.append([int(x) for x in line.strip().split("\t")])
    gt = np.array(gt)

    for i in range(len(gt)):
        index_x = gt[i][0]
        index_y = gt[i][1]
        if index_x in mapping_index_1 and index_y in mapping_index_2:
            tmp_ground_truth[mapping_index_1[index_x]] = mapping_index_2[index_y]

    ground_truth = {}
    if is_training:
        for key, value in tmp_ground_truth.items():
            ground_truth[key] = value
            if len(ground_truth) == int(0.6*len(tmp_ground_truth)):
                break
    else:
        for key, value in tmp_ground_truth.items():
            ground_truth[key] = value

    return G1_adj_matrix, G2_adj_matrix, emb1, emb2, ground_truth