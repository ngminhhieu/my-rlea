import torch
import sys
import os
import argparse
from tqdm import tqdm
from framework.utils import build_adj_matrix_and_embeddings
from framework.agent import Agent
from framework.utils import save_results
import numpy as np

if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_results',
                        default="./log/test/results/_test",
                        type=str,
                        help='Directory for results')
    parser.add_argument('--weights_path',
                        default="./log/train/weights/_test",
                        type=str,
                        help='Directory for weights')
    parser.add_argument('--cuda',
                        default=1,
                        type=int,
                        help='GPU device')

    args = parser.parse_args()
    if not os.path.exists(args.log_results):
        os.makedirs(args.log_results)

    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')

    print("Loading data...")
    G1_adj_matrix, G2_adj_matrix, emb1, emb2, ground_truth = build_adj_matrix_and_embeddings(False)

    print("Intitializing agent...")
    lr = 0.0001
    agent = Agent(G1_adj_matrix, G2_adj_matrix,
                  emb1.shape[1], 128, 64, gamma=0.99, activation='Sigmoid')
    agent.load_state_dict(torch.load(args.weights_path))
    first_embeddings_torch = torch.from_numpy(
        emb1).type(torch.FloatTensor).to(device)
    second_embeddings_torch = torch.from_numpy(
        emb2).type(torch.FloatTensor).to(device)
    agent.to(device)
    agent.eval()

    print("Testing...")
    training_total_match = 0
    testing_total_match = 0
    training_acc = 0
    testing_acc = 0
    ground_truth = np.array(list(ground_truth.items()))
    training_gt = ground_truth[:int(0.6*len(ground_truth))]
    testing_gt = ground_truth[int(0.6*len(ground_truth)):]
    for i in tqdm(range(len(training_gt)), desc="Evaluate training accuracy"):
        action, p = agent.get_action(first_embeddings_torch,
                                     second_embeddings_torch, [(training_gt[i][0], training_gt[i][1])], False)
        if action == 1:
            training_total_match += 1

    for i in tqdm(range(len(testing_gt)), desc="Evaluate testing accuracy"):
        action, p = agent.get_action(first_embeddings_torch,
                                     second_embeddings_torch, [(testing_gt[i][0], testing_gt[i][1])], False)
        if action == 1:
            testing_total_match += 1


    print("Saving results...")
    training_acc = training_total_match/len(training_gt)
    print("Training accuracy: ", training_acc)
    testing_acc = testing_total_match/len(testing_gt)
    print("Testing accuracy: ", testing_acc)
    save_results([training_acc, testing_acc], args.log_results + "/accuracy.csv")
