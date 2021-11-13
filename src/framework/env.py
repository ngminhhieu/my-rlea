# đổi lại thành dạng hash table
from framework.utils import subgraph_build_adj_matrix_and_embeddings
import numpy as np
from sklearn import preprocessing

class Environment():
    def __init__(self, config):
        self.config = config
        G1_adj_matrix, G2_adj_matrix, emb1, emb2, training_gt, testing_gt, data_x, data_y = subgraph_build_adj_matrix_and_embeddings(config.num_nodes)
        self.g1_adj_matrix = G1_adj_matrix
        self.g2_adj_matrix = G2_adj_matrix
        self.data_x = data_x
        self.data_y = data_y
        self.emb1 = preprocessing.normalize(emb1)
        self.emb2 = preprocessing.normalize(emb2)
        self.true = 1
        self.false = 0
        self.training_gt = training_gt
        self.testing_gt = testing_gt
        self.info = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
        cosim_hash_table, prob_skip_hash_table = self.get_cosim_hash_table()
        self.hash_table, self.hash_skip_rate = self.get_hash_table(training_gt, cosim_hash_table, prob_skip_hash_table)
        self.list_state = self.get_list_state(self.hash_table, self.hash_skip_rate)
        self.count = 0

        
    def reset(self, ep_num=1):
        """
          Important: The observation must be numpy array 
          : return: (np.array)
        """
        self.list_state = self.get_list_state(
            self.hash_table, self.hash_skip_rate, ep_num=ep_num)
        self.info = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
        self.count = 0
        return self.list_state[0]

    def step(self, action):
        current_state = self.list_state[0]
        if self.is_match(current_state):
            self.count += 1
        self.list_state.pop(0)
        if action == self.true:            
            if self.is_match(current_state):
                score = self.config.tp
                self.info['tp'] += 1 
            elif not self.is_match(current_state):
                score = self.config.fp
                self.info['fp'] += 1 
        elif action == self.false:
            if self.is_match(current_state):
                score = self.config.fn
                self.info['fn'] += 1 
            elif not self.is_match(current_state):
                score = self.config.tn
                self.info['tn'] += 1
        else:
            raise ValueError(
                "Received invalid action={} which is not part of the action space".format(action))

        if len(self.list_state) == 0:
            next_state = (None, None)  # khong con phan tu nao trong G1
            done = True
        else:
            # get next state id and next state embedding
            next_state = self.list_state[0]
            done = False

        return next_state, score, done


    def is_match(self, state):
        true_state = (state[0], self.training_gt[state[0]])
        if state == true_state:
            return True
        return False 

    @staticmethod
    def get_hash_table(ground_truth, cosim_hash_table, prob_skip_hash_table):
        '''
            Input: groundtruth,
            Output: hashtable of corresponding  and hashtable of skipping rate 
        '''
        hash_mapping_node = {}
        hash_skip_rate = {}

        for source_node, target_node in ground_truth.items():
            hash_mapping_node[source_node] = cosim_hash_table[target_node]
            hash_skip_rate[source_node] = prob_skip_hash_table[target_node]
        return hash_mapping_node, hash_skip_rate


    def get_cosim_hash_table(self, top_k=5): # Kết quả tốt nhất là k=5 khi thực hiện với linear
        # get hashtable of nearest node corresponding to each source node 
        emb1 = preprocessing.normalize(self.emb1)
        emb2 = preprocessing.normalize(self.emb2)
        sim_mat = np.matmul(emb1, emb2.T) 
        cosim_hash_table = {}
        prob_skip_hash_table = {} 
        for i in range(len(emb1)):
            rank = list(np.argpartition(sim_mat[i, :], -top_k)[-top_k:])
            rank.append(i) # Ke ca co roi cung append. De tang so luong cap true positive
            cosim_hash_table.update({i: [ j for j in rank ]})
            max_sim_score = max(sim_mat[i,rank] ) 
            lst_sim_mat = [ max_sim_score -i for i in sim_mat[i, rank] ]
            prob_skip_hash_table.update({i: lst_sim_mat})

        return cosim_hash_table, prob_skip_hash_table 

    def get_list_state(self, hash_table, hash_skip_rate, ep_num=1, min_skip_rate=0.05, basic_skip_rate=0.9, discount_ratio=0.9):
        # calculate skip rate at each episode and eliminate 
        lst_state = []
        for key in hash_table.keys():
            lst_key_2 = hash_table[key]
            lst_prob = hash_skip_rate[key]

            for i in range(len(lst_key_2)-1): # Vi append groundtruth ở cuối mảng nên bỏ ra để không bị skip
                # idea 2: skip rate
                prob = max(min_skip_rate, (1 - discount_ratio ** (ep_num -1)) * basic_skip_rate * lst_prob[i] ) #  max(pmin s, η^(t−1)ps * di)            
                cur_rand = np.random.rand() 
                if(prob < cur_rand):
                    lst_state.append((key, lst_key_2[i]))
            lst_state.append((key, lst_key_2[len(lst_key_2)-1])) # Không duyệt nên hết vòng phải cho vào list states
        return lst_state


    def popCurrentState(lst_state, current_state):
        key1, key2 = current_state[0], current_state[1]
        lst_remove = []
        for state in lst_state:
            key_state_1 = state[0]
            key_state_2 = state[1]

            if key_state_1 == key1 or key_state_2 == key2:
                lst_remove.append(state)
        lst_final = list(set(lst_state) - set(lst_remove))
        return lst_final