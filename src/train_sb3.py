import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import torch
from tqdm import tqdm
import torch.optim as optim
from torch.distributions import Categorical
from framework.utils import build_adj_matrix_and_embeddings, normalize_prob, seed, plot
from framework.env import SequentialMatchingEnv
from framework.agent import Agent
from framework.env import isAligned
import time
import sys
import argparse
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def get_utils():
    checkpoint_callbacks = CheckpointCallback(save_freq=1000, save_path="./log/", name_prefix="sb3")

def make_env(args):
    seed(args)    
    env = SequentialMatchingEnv(seed=args.seed)
    return env

def create_training_data():
    G1_adj_matrix, G2_adj_matrix, emb1, emb2, ground_truth = build_adj_matrix_and_embeddings(True)
    scaler = MinMaxScaler()
    scaler.fit(emb1)
    emb1 = scaler.transform(emb1)
    emb2 = scaler.transform(emb2)
    num_gt = len(ground_truth)

    return G1_adj_matrix, G2_adj_matrix, emb1, emb2, ground_truth, num_gt

def define_model(model_type):
    if model_type == 'ppo':
        MODEL = PPO
    elif model_type == 'a2c':
        MODEL = A2C
    else:
        MODEL = None
    return MODEL

def train(args):
    print("Beginning the training process...")
    env=  make_env(args)
    early_stopping = 0
    results = []
    
    MODEL = define_model(args.model)
    
    model = MODEL('MlpPolicy', env, n_steps=4)

    all_episode_rewards = []
    for ep in tqdm(range(1, args.episode+1)):
        start_episode = time.time()
        episode_rewards = []
        state, ep_reward, ep_tp, ep_tn, ep_fp, ep_fn  = env.reset(ep), 0, 0, 0, 0, 0
        state = np.array(state).reshape(2,1)
        while True:
            # import pdb; pdb.set_trace()
            action, _state = model.predict(state)
            next_state, reward, done, info = env.step(action)

            tp, tn, fp, fn = info['tp'], info['tn'], info['fp'], info['fn']
            next_state = np.array(next_state).reshape(2,1)

            # add reward
            episode_rewards.append(reward)
            ep_reward += reward

            ep_tp += tp
            ep_tn += tn
            ep_fp += fp
            ep_fn += fn

            if done:
                break            
            state = next_state

        all_episode_rewards.append(sum(episode_rewards))
        end_episode = time.time()

        accuracy = (ep_tp + ep_tn) / (ep_tp + ep_tn + ep_fn + ep_fn)
        precision = ep_tp / (ep_tp + ep_fp)
        recall = ep_tp / (ep_tp + ep_fn) 
        results.append({'episode': ep, 'reward' :ep_reward, 'time': end_episode - start_episode, 'accuracy': accuracy, 'precision': precision, 'recall': recall})

                # Early stopping
        if ep_reward == num_gt:
            early_stopping += 1
        else:
            early_stopping = 0
        if early_stopping == args.early_stopping:
            print("Early stopping")
            print("Goal reached! Reward: {}/{}".format(ep_reward, num_gt))
            break

    mean_episode_reward = np.mean(all_episode_rewards)
    
    df = pd.DataFrame(results)
    df.to_csv("./log/train/results/sb3/log.csv", columns=['episode', 'reward', 'time', 'accuracy', 'precision', 'recall'])

    print('mean_episode_reward: {}'.format(mean_episode_reward))
    return mean_episode_reward


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_results',
                        default="./log/train/results/sb3",
                        type=str,
                        help='Directory for results')
    parser.add_argument('--model',
                        default='ppo',
                        type=str,
                        help="Model type: Including (ppo, a2c)")
    parser.add_argument('--log_weights',
                        default="./log/train/weights/sb3",
                        type=str,
                        help='Directory for weights')
    parser.add_argument('--lr',
                        default=0.001,
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
    parser.add_argument('--tl',
                        default=0,
                        type=int,
                        help='Transfer learning')
    parser.add_argument('--seed',
                        default=52,
                        type=int,
                        help='Seed')

    parser.add_argument('--cuda',
                        default=2,
                        type=int,
                        help='GPU device')

    args = parser.parse_args()
    seed(args)
    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.log_results):
        os.makedirs(args.log_results)
    if not os.path.exists(args.log_weights):
        os.makedirs(args.log_weights)

    # init device
    device = torch.device('cpu')

    # data 
    G1_adj_matrix, G2_adj_matrix, emb1, emb2, ground_truth, num_gt = create_training_data()
    
    print("Intitializing agent...")
    agent = Agent(G1_adj_matrix, G2_adj_matrix, emb1.shape[1], 16, 1, activation="Sigmoid")


    optimizer = optim.Adam(agent.parameters(), lr=args.lr)
    eps = np.finfo(np.float32).eps.item()
    agent.to(device)
    agent.train()

    results = train(args)
    
    print("Saving results...")
    torch.save(agent.state_dict(), args.log_weights + "/best.pt")
    print("Done!")
    
    

