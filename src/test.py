import torch
import sys
import os
import argparse
from tqdm import tqdm
from torch.distributions import Categorical
from framework.env import Environment
from framework.agent import Agent, Policy
from framework.utils import save_results
import numpy as np

def get_action(state):
    policy = agent(first_embeddings_torch, second_embeddings_torch, state)
    m = Categorical(policy)
    action = m.sample()
    agent.saved_log_probs.append(m.log_prob(action))
    return action[0].cpu().data.numpy(), policy[0].cpu().data.numpy()

if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_results',
                        default="./log/test/results/_test_3",
                        type=str,
                        help='Directory for results')
    parser.add_argument('--weights_path',
                        default="./log/train/weights/_test_3/best.pt",
                        type=str,
                        help='Directory for weights')
    parser.add_argument('--cuda',
                        default=1,
                        type=int,
                        help='GPU device')
    parser.add_argument('--num_nodes',
                        default=15000,
                        type=int,
                        help='Seed')

    args = parser.parse_args()
    config = args
    if not os.path.exists(args.log_results):
        os.makedirs(args.log_results)

    # device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    print("Loading data...")
    env = Environment(config)

    print("Intitializing agent...")
    lr = 0.0001
    agent = Policy(env.emb1.shape[1])
    agent.load_state_dict(torch.load(args.weights_path))
    first_embeddings_torch = torch.from_numpy(
        env.emb1).type(torch.FloatTensor).to(device)
    second_embeddings_torch = torch.from_numpy(
        env.emb2).type(torch.FloatTensor).to(device)
    agent.to(device)
    agent.eval()

    print("Testing...")
    training_total_match = 0
    testing_total_match = 0
    training_acc = 0
    testing_acc = 0
    training_gt = np.array(list(env.training_gt.items()))
    testing_gt = np.array(list(env.testing_gt.items()))
    for i in tqdm(range(len(training_gt)), desc="Evaluate training accuracy"):
        action, p = get_action([(training_gt[i][0], training_gt[i][1])])
        if action == 1:
            training_total_match += 1

    for i in tqdm(range(len(testing_gt)), desc="Evaluate testing accuracy"):
        action, p = get_action([(testing_gt[i][0], testing_gt[i][1])])
        print(p)
        if action == 1:
            testing_total_match += 1


    print("Saving results...")
    training_acc = training_total_match/len(training_gt)
    print("Training accuracy: ", training_acc)
    testing_acc = testing_total_match/len(testing_gt)
    print("Testing accuracy: ", testing_acc)
    save_results([training_acc, testing_acc], args.log_results + "/accuracy.csv")
