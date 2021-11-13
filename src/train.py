import torch
from tqdm import tqdm
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from framework.utils import normalize_prob, seed, plot
from framework.env import Environment
from framework.agent import Policy_LR, Policy_GCN
import time
import sys
import argparse
import os
import wandb
import pandas as pd
import matplotlib.pyplot as plt


def get_log_dir():
    results_dir = "./log/train/results/{}".format(args.run_name)
    weight_dir = "./log/train/weight/{}".format(args.run_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    return results_dir, weight_dir


def get_action(state):
    policy = agent(env.data_x, env.data_y, state)
    m = Categorical(policy)
    action = m.sample()
    agent.saved_log_probs.append(m.log_prob(action))
    return action[0].cpu().data.numpy(), policy[0].cpu().data.numpy()


def finish_episode():
    R = 0
    policy_loss = []
    returns = []
    for r in agent.rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(agent.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    policy_loss = torch.cat(policy_loss).sum()
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    del agent.rewards[:]
    del agent.saved_log_probs[:]
    return policy_loss


def train():
    early_stopping = 0
    results = {}
    results["num_gt"] = num_gt
    results["accuracy"] = []
    results["training"] = []  # results of each episode
    results["prob"] = {}  # to check probability of each pair after one episode
    results["policy"] = []
    for k, v in env.training_gt.items():
        results["prob"][(k, v)] = [0]

    print("Training...")
    for ep in tqdm(range(1, args.episode + 1)):
        start_episode = time.time()

        # Reset environment
        state, ep_reward = env.reset(), 0
        total_reward = num_gt * args.tp + \
            (len(env.list_state) - num_gt) * args.tn
        # Get policy, action and reward
        while True:
            action, policy = get_action([state])
            if ep == args.episode:
                results["policy"].append(policy)
                print(policy)

            if env.is_match(state):
                results["prob"][state].append(policy[1])
            next_state, reward, done = env.step(action)

            # add reward
            agent.rewards.append(reward)
            ep_reward += reward
            if done:
                break

            # next state
            state = next_state

        # print("count: ", env.count)
        # just for storing "prob" results
        results["prob"] = normalize_prob(results["prob"])
        wandb.log({"reward": ep_reward})
        # Train model
        loss = finish_episode()
        end_episode = time.time()

        results["training"].append(
            [ep, ep_reward, loss.cpu().detach().numpy(), end_episode - start_episode])
        results["accuracy"].append(
            [env.info["tp"], env.info["tn"], env.info["fp"], env.info["fn"]])
        torch.save(agent.state_dict(), weight_dir + "/best.pt")
        # Monitoring
        if ep % 10 == 0:
            print("Episode: {}   Reward: {}/{}   Agent loss: {}".format(ep,
                  ep_reward, total_reward, loss.cpu().detach().numpy()))
        if ep % 10 == 0:
            print(env.info)

        # Early stopping
        if ep_reward == total_reward:
            early_stopping += 1
        else:
            early_stopping = 0
        if early_stopping == args.early_stopping:
            print("Early stopping")
            print("Goal reached! Reward: {}/{}".format(ep_reward, total_reward))
            break

    return results, agent


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr',
                        default=0.01,
                        type=float,
                        help='Learning rate')
    parser.add_argument('--gamma',
                        default=0.99,
                        type=float,
                        help='Discount factor')
    parser.add_argument('--early_stopping',
                        default=50,
                        type=int,
                        help='Early stopping')
    parser.add_argument('--episode',
                        default=1000,
                        type=int,
                        help='Episode')
    parser.add_argument('--seed',
                        default=52,
                        type=int,
                        help='Seed')
    parser.add_argument('--tp',
                        default=10,
                        type=int,
                        help='True positive score')
    parser.add_argument('--tn',
                        default=1,
                        type=int,
                        help='True negative score')
    parser.add_argument('--fp',
                        default=-3,
                        type=int,
                        help='False positive score')
    parser.add_argument('--fn',
                        default=-2,
                        type=int,
                        help='False negative score')
    parser.add_argument('--num_nodes',
                        default=500,
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
    parser.add_argument('--run_name',
                        default="test",
                        type=str,
                        help='Run name for a training case')

    args = parser.parse_args()
    seed(args)
    results_dir, weight_dir = get_log_dir()

    print("Beginning the training process...")
    id_name_wandb = "{}".format(args.run_name)
    run = wandb.init(project=args.project_name, entity=args.team_name, config=args,
                     id=id_name_wandb, name=id_name_wandb, job_type="train", resume=True)
    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    print("Building environment...")
    env = Environment(args)
    num_gt = len(env.training_gt)
    print("Num nodes in G1: ", len(env.g1_adj_matrix))
    print("Num nodes in G2: ", len(env.g2_adj_matrix))
    print("Num ground_truth: ", num_gt)


    print("Intitializing agent...")
    # agent = Policy_LR(env.emb1.shape[1])
    agent = Policy_GCN(env.emb1.shape[1], 128, 64)

    optimizer = optim.Adam(agent.parameters(), lr=args.lr)
    eps = np.finfo(np.float32).eps.item()
    agent.to(device)
    agent.train()
    results, agent = train()

    print("Saving results...")
    table_acc = pd.DataFrame(data=results["accuracy"], columns=[
                             "tp", "tn", "fp", "fn"])
    plt.plot(table_acc.index, table_acc["tp"], label="tp")
    plt.plot(table_acc.index, table_acc["tn"], label="tn")
    plt.plot(table_acc.index, table_acc["fp"], label="fp")
    plt.plot(table_acc.index, table_acc["fn"], label="fn")
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Number of cases")
    plt.title("Accuracy over episodes")

    wandb.log({"accuracy": plt})
    torch.save(agent.state_dict(), wandb.run.dir + "/best.pt")
    torch.save(agent.state_dict(), weight_dir + "/best.pt")
    plot(results["training"], results["prob"], results["num_gt"], results_dir)
    print("Done!")
