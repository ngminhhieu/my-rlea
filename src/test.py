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
import wandb


def get_log_dir():
    results_dir = "./log/test/results/{}".format(args.num_nodes)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir


def get_action(state):
    policy = agent(first_embeddings_torch, second_embeddings_torch, state)
    m = Categorical(policy)
    action = m.sample()
    agent.saved_log_probs.append(m.log_prob(action))
    return action[0].cpu().data.numpy(), policy[0].cpu().data.numpy()


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_nodes',
                        default=250,
                        type=int,
                        help='Seed')
    parser.add_argument('--project_name',
                        default="rlea",
                        type=str,
                        help='Project name in Wandb')
    parser.add_argument('--team_name',
                        default="bkai",
                        type=str,
                        help='Team name in Wandb')

    args = parser.parse_args()
    results_dir = get_log_dir()
    # device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')
    print("Loading data...")
    env = Environment(args)

    print("Intitializing agent...")
    id_name_wandb = id_name_wandb = "{}".format(args.num_nodes)
    wandb.init(project=args.project_name, entity=args.team_name, config=args,
               id=id_name_wandb, name=id_name_wandb, job_type="train", resume=True)
    best_model = wandb.restore('best.pt')
    device = 'cpu'
    agent = Policy(env.emb1.shape[1])
    agent.load_state_dict(torch.load(best_model.name))
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
    # online
    wandb.log({"Training accuracy": training_acc,
              "Testing accuracy:": testing_acc})
    # local
    save_results([training_acc, testing_acc], results_dir + "/accuracy.csv")
