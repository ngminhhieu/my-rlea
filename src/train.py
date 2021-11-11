import torch
from tqdm import tqdm
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from framework.utils import  normalize_prob, seed, plot
from framework.env import Environment
from framework.agent import Policy
import time
import sys
import argparse
import os
from sklearn.preprocessing import MinMaxScaler
import wandb
import pandas as pd


def get_action(state):
    policy = agent(first_embeddings_torch, second_embeddings_torch, state)
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
        total_reward = num_gt * args.tp + (len(env.list_state) - num_gt) * args.tn
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

            # just for storing "prob" results
        results["prob"] = normalize_prob(results["prob"])
        wandb.log({"reward": ep_reward})
        # Train model
        loss = finish_episode()
        end_episode = time.time()

        results["training"].append([ep, ep_reward, loss.cpu().detach().numpy(), end_episode - start_episode])
        results["accuracy"].append([env.info["tp"], env.info["tn"], env.info["fp"], env.info["fn"]])
        torch.save(agent.state_dict(), args.log_weights + "/best.pt")
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
    parser.add_argument('--log_results',
                        default="./log/train/results/_test_3",
                        type=str,
                        help='Directory for results')
    parser.add_argument('--log_weights',
                        default="./log/train/weights/_test_3",
                        type=str,
                        help='Directory for weights')
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
                        default=500,
                        type=int,
                        help='Episode')
    parser.add_argument('--tl',
                        default=0,
                        type=int,
                        help='Transfer learning')
    parser.add_argument('--seed',
                        default=52,
                        type=int,
                        help='Seed')
    parser.add_argument('--tp',
                        default=10,
                        type=int,
                        help='Seed')
    parser.add_argument('--tn',
                        default=1,
                        type=int,
                        help='Seed')
    parser.add_argument('--fp',
                        default=-3,
                        type=int,
                        help='Seed')
    parser.add_argument('--fn',
                        default=-2,
                        type=int,
                        help='Seed')
    parser.add_argument('--num_nodes',
                        default=15000,
                        type=int,
                        help='Seed')

    args = parser.parse_args()
    seed(args)
    wandb.init(config=args)
    config = wandb.config
    if not os.path.exists(args.log_results):
        os.makedirs(args.log_results)
    if not os.path.exists(args.log_weights):
        os.makedirs(args.log_weights)
    print("Beginning the training process...")
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print("Loading data...")
    env = Environment(config)
    num_gt = len(env.training_gt)
    print("Num nodes in G1: ", len(env.g1_adj_matrix))
    print("Num nodes in G2: ", len(env.g2_adj_matrix))
    print("Num ground_truth: ", num_gt)
    print("Building environment...")
    
    first_embeddings_torch = torch.from_numpy(
        env.emb1).type(torch.FloatTensor).to(device)
    second_embeddings_torch = torch.from_numpy(
        env.emb2).type(torch.FloatTensor).to(device)

    print("Intitializing agent...")
    agent = Policy(env.emb1.shape[1])
    # transfer learning
    if args.tl:
        agent.load_state_dict(torch.load(args.weights_path))

    optimizer = optim.Adam(agent.parameters(), lr=args.lr)
    eps = np.finfo(np.float32).eps.item()
    agent.to(device)
    agent.train()
    results, agent = train()

    print("Saving results...")
    table_acc = pd.DataFrame(data=results["accuracy"], columns = ["tp", "tn", "fp", "fn"])
    wandb.log({"accuracy": wandb.plot.line_series(
                       xs= [i for i in range(len(results["accuracy"]))], 
                       ys= [table_acc[c][:] for c in table_acc.columns],
                       keys=["tp", "tn", "fp", "fn"],
                       title="Accuracy over episodes",
                       xname="Episode")})
    table_prob = wandb.Table(data=results["policy"], columns = ["unmatch", "match"])
    wandb.log({"policy": wandb.plot.scatter(table_prob, "unmatch", "match")})
    torch.save(agent.state_dict(), args.log_weights + "/best.pt")
    plot(results["training"], args.log_results,
         results["prob"], results["num_gt"])
    print("Done!")
