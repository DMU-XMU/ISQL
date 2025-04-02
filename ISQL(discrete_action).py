import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import argparse
import torch.optim as optim
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_
import math
import os
from tqdm import *
import pandas as pd
from pandas import DataFrame
from sklearn.cluster import KMeans
from joblib import dump, load
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle
os.environ['CUDA_VISIBLE_DEVICES']='1'

REWARD_THRESHOLD = 20
reg_lambda = 5
per_alpha = 0.6 # PER hyperparameter
per_epsilon = 0.01 # PER hyperparameter
batch_size = 32
# num_steps = 70000 # How many steps to train for
load_model = False #Whether to load a saved model.
tau = 0.001 #Rate to update target network toward primary network
save_results = False
clip_reward = False

###############################

action_map = {}
count = 0
for iv in range(5):
    for vaso in range(5):
        action_map[(iv,vaso)] = count
        count += 1

class DiscreteActor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_size=128):
        super(DiscreteActor, self).__init__()
       
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_logits = self.fc3(x)
        action_probs = self.softmax(action_logits)
        return action_probs
    

    def get_actions(self, state, epsilon=1e-6):
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        actions = dist.sample()
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probs = torch.log(action_probs + z)
        max_action = torch.argmax(action_probs, dim=1)
        
        return max_action, actions.detach(), action_probs, log_action_probs

    def act_sample(self, state, epsilon=1e-6):
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        actions = dist.sample()
        return actions.detach()

    def act_max(self, state, epsilon=1e-6):
        action_probs = self.forward(state)
        max_action = torch.argmax(action_probs, dim=1)
        return max_action.detach()

    def evaluate(self, state, actions):
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        log_action_probs = dist.log_prob(actions)
        return log_action_probs

class Critic(nn.Module):
    """Critic (Value) Model."""
    def __init__(self, state_size, action_size, hidden_size=128, seed=1):
        super(Critic, self).__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class Value(nn.Module):
    """Value (Value) Model."""
    def __init__(self, state_size, hidden_size=128):
        super(Value, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DiscreteIQL(object):
    def __init__(
        self,
        *,
        state_dim,
        num_actions,
        hidden_dim,
        lr=3e-4,
        gamma,
        target_update_frequency=1,
        tau=0.005,
        target_value_clipping=False,
        expectile: float = 1.0,
        temperature: float = 0.1,
        per_flag,
        reg_flag,
        act_max_flag,
        huber_flag,
        df,
        val_df,
        test_df,
        state_features,
        **kwargs,
    ):
        super().__init__()
        self.lr = lr
        self.gamma = gamma
        self.num_actions = num_actions
        self.per_flag = per_flag
        self.reg_flag = reg_flag
        self.act_max_flag = act_max_flag

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Actor Network 
        self.actor_local = DiscreteActor(state_dim, num_actions, hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr)     
        
        # Critic Network (w/ Target Network)
        self.critic1 = Critic(state_dim, num_actions, hidden_dim, 2).to(self.device)
        self.critic2 = Critic(state_dim, num_actions, hidden_dim, 1).to(self.device)
        
        assert self.critic1.parameters() != self.critic2.parameters()
        
        self.critic1_target = Critic(state_dim, num_actions, hidden_dim).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Critic(state_dim, num_actions, hidden_dim).to(self.device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Freeze target network so we don't accidentally train it
        for param in self.critic1_target.parameters():
            param.requires_grad = False

        for param in self.critic2_target.parameters():
            param.requires_grad = False

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.lr) 
        
        self.value_net = Value(state_size=state_dim, hidden_size=hidden_dim).to(self.device)
        
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.lr)
        self.iterations = 0
        self.tau = tau
        self.expectile = expectile
        self.temperature = temperature
        self.soft_update_every = 1
        self.huber_flag = huber_flag
        self.df = df
        self.val_df = val_df
        self.test_df = test_df
        self.state_features = state_features

    # 保存模型
    def save_policy(self, filename):
        actor_state_dict = self.actor_local.state_dict()
        critic1_state_dict = self.critic1.state_dict()
        critic2_state_dict = self.critic2.state_dict()
        value_net_state_dict = self.value_net.state_dict()
        checkpoint = {
            'actor': actor_state_dict,
            'critic1': critic1_state_dict,
            'critic2': critic2_state_dict,
            'value_net': value_net_state_dict,
            'iterations': self.iterations,
            'tau': self.tau,
            'temperature': self.temperature,
        }
        torch.save(checkpoint, filename)

    def load_policy(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor_local.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.value_net.load_state_dict(checkpoint['value_net'])

        self.iterations = checkpoint['iterations']
        self.tau = checkpoint['tau']
        self.temperature = checkpoint['temperature']

    def calc_abs_error(self, states, actions, rewards, notdone, next_states):
        next_v = self.value_net(next_states).detach()
        next_v[next_v > REWARD_THRESHOLD] = REWARD_THRESHOLD
        next_v[next_v < -REWARD_THRESHOLD] = -REWARD_THRESHOLD
        q_target = rewards + (self.gamma * notdone * next_v.squeeze(-1))

        q1 = self.critic1(states).gather(1, actions.long().unsqueeze(-1)).squeeze(-1)
        q2 = self.critic2(states).gather(1, actions.long().unsqueeze(-1)).squeeze(-1)

        abs_error1 = torch.abs(q_target - q1)
        abs_error2 = torch.abs(q_target - q2)
        abs_error = torch.min(abs_error1, abs_error2)

        return abs_error.detach()

    def calc_q_loss(self, states, actions, rewards, notdone, next_states, imp_sampling_weights):
        next_v = self.value_net(next_states).detach()
        next_v[next_v > REWARD_THRESHOLD] = REWARD_THRESHOLD
        next_v[next_v < -REWARD_THRESHOLD] = -REWARD_THRESHOLD
        q_target = rewards + (self.gamma * notdone * next_v.squeeze(-1))

        q1 = self.critic1(states).gather(1, actions.long().unsqueeze(-1)).squeeze(-1)
        q2 = self.critic2(states).gather(1, actions.long().unsqueeze(-1)).squeeze(-1)

        if self.per_flag:
            imp_sampling_weights_tensor = torch.tensor(imp_sampling_weights).to(self.device)
            if self.huber_flag:
                critic1 = F.smooth_l1_loss(q1, q_target.detach(), reduction='none')
                critic1_loss = torch.mean(critic1 * imp_sampling_weights_tensor)
                critic2 = F.smooth_l1_loss(q2, q_target.detach(), reduction='none')
                critic2_loss = torch.mean(critic2 * imp_sampling_weights_tensor)
            else:
                td_error1 = ((q_target.detach() - q1)**2)
                td_error2 = ((q_target.detach() - q2)**2)
                critic1_loss = torch.mean(torch.multiply(td_error1, imp_sampling_weights_tensor))
                critic2_loss = torch.mean(torch.multiply(td_error2, imp_sampling_weights_tensor))
        else:
            if self.huber_flag:
                critic1_loss = F.smooth_l1_loss(q1, q_target.detach())
                critic2_loss = F.smooth_l1_loss(q2, q_target.detach())
            else:
                critic1_loss = ((q1 - q_target.detach())**2).mean() 
                critic2_loss = ((q2 - q_target.detach())**2).mean()

        if self.reg_flag:
            q1_reg = torch.abs(q1) - REWARD_THRESHOLD
            q2_reg = torch.abs(q2) - REWARD_THRESHOLD
            q1_reg[q1_reg < 0] = 0
            q2_reg[q2_reg < 0] = 0
            q1_reg_term = torch.sum(q1_reg)
            q2_reg_term = torch.sum(q2_reg)
            critic1_loss = critic1_loss + reg_lambda * q1_reg_term
            critic2_loss = critic2_loss + reg_lambda * q2_reg_term
        
        return critic1_loss, critic2_loss

    def calc_policy_loss(self, states, actions):
        with torch.no_grad():
            v = self.value_net(states)
            q1 = self.critic1_target(states).gather(1, actions.long().unsqueeze(-1))
            q2 = self.critic2_target(states).gather(1, actions.long().unsqueeze(-1))
            min_Q = torch.min(q1,q2)

        exp_a = torch.exp((min_Q.detach() - v) * self.temperature)
        exp_a = torch.min(exp_a, torch.FloatTensor([100.0]).to(self.device)).squeeze(-1)

        log_probs = self.actor_local.evaluate(states, actions.squeeze(-1))
        actor_loss = -(exp_a * log_probs).mean()

        return actor_loss

    def calc_value_loss(self, states, actions):
        with torch.no_grad():
            q1 = self.critic1_target(states).gather(1, actions.long().unsqueeze(-1))
            q2 = self.critic2_target(states).gather(1, actions.long().unsqueeze(-1))
            min_Q = torch.min(q1,q2)
        
        value = self.value_net(states)
        diff = torch.squeeze(min_Q.detach() - value, dim=1)
        value_loss = self.expectile_loss(diff, expectile=self.expectile).mean()
        return value_loss

    def expectile_loss(self, diff, expectile=0.8):
        weight = torch.where(diff > 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def train(self, states, actions, rewards, next_states, done_flags, imp_sampling_weights):
        state, action, next_state, reward, notdone = states, actions, next_states, rewards, 1- done_flags
        # update value net
        self.value_optimizer.zero_grad()
        value_loss = self.calc_value_loss(state, action)
        value_loss.backward()
        self.value_optimizer.step()

        # update critic
        critic1_loss, critic2_loss = self.calc_q_loss(state, action, reward, notdone, next_state, imp_sampling_weights)
        # critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        # critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # update actor
        actor_loss = self.calc_policy_loss(state, action)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # calculate abs_error
        abs_error = self.calc_abs_error(state, action, reward, notdone, next_state)

        self.iterations += 1
        # target函数参数更新
        if self.iterations % self.soft_update_every == 0:
            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic1, self.critic1_target)
            self.soft_update(self.critic2, self.critic2_target)
        
        return abs_error

    def do_eval_iql(self, eval_type):
        gen = self.process_eval_batch(size=1000, eval_type=eval_type)
        phys_q_ret = []
        actions_ret = []
        agent_q_ret = []
        actions_taken_ret = []
        agent_qsa_ret = []
        pi_e_ret = []
        error_ret = 0

        for b in gen:
            states, actions, rewards, next_states, done_flags, _ = b
            # firstly get the chosen actions at the next timestep
            if self.act_max_flag:
                actions_from_q1 = self.actor_local.act_max(next_states)
            else:
                actions_from_q1 = self.actor_local.act_sample(next_states) 
            # Q values for the next timestep from target network, as part of the Double DQN update
            Q2_1 = self.critic1_target(next_states) #
            Q2_2 = self.critic2_target(next_states)
            Q2 = torch.min(Q2_1, Q2_2)

            # handles the case when a trajectory is finished
            end_multiplier = 1 - done_flags

            double_q_value = Q2[range(len(Q2)), actions_from_q1]

            targetQ = rewards + (self.gamma * double_q_value * end_multiplier)

            q_output1 = self.critic1(states)
            q_output2 = self.critic2(states)
            q_output = torch.min(q_output1, q_output2) 

            pi_e = self.actor_local(states)
            
            temp_q_value = q_output[range(len(q_output)), actions.long()]
            #self.predict = tf.argmax(self.q_output, 1, name='predict')  # vector of length batch size

            if self.act_max_flag:
                actions_taken = self.actor_local.act_max(states)
            else:
                actions_taken = self.actor_local.act_sample(states)

            abs_error = targetQ - temp_q_value

            # return the relevant q values and actions
            phys_q = q_output[range(len(q_output)), actions.long()]  
            agent_q = q_output[range(len(q_output)), actions_taken.long()]
            # error = torch.mean(abs_error)

            # update the return vals
            phys_q_ret.extend(phys_q)
            actions_ret.extend(actions)
            agent_q_ret.extend(agent_q)  # q
            actions_taken_ret.extend(actions_taken)  # a
            # error_ret += error
            agent_qsa_ret.extend(q_output)  # qsa
            pi_e_ret.extend(pi_e)

        return agent_qsa_ret, phys_q_ret, actions_ret, agent_q_ret, actions_taken_ret, pi_e_ret

    def save_results(self, type=None):
        # get the chosen actions for the train, val, and test set when training is complete.
        if type == None:
            print("the type is None!")
        elif  type=='test':
            agent_qsa_test, phys_q_test, _, agent_q_test, agent_actions_test, agent_pi_e_test = self.do_eval_iql(eval_type='test')

        return agent_qsa_test, phys_q_test, _, agent_q_test, agent_actions_test, agent_pi_e_test

    def save_results_all_q(self):
        agent_qsa_train, phys_q_train, _, agent_q_train, agent_actions_train, agent_pi_e_train = self.do_eval_iql(eval_type='train')
        agent_qsa_val, phys_q_val, _, agent_q_val, agent_actions_val, agent_pi_e_val = self.do_eval_iql(eval_type='val')
        agent_qsa_test, phys_q_test, _, agent_q_test, agent_actions_test, agent_pi_e_test = self.do_eval_iql(eval_type='test')
        return phys_q_train, phys_q_val, phys_q_test

    def save_results_all_qsa(self):
        agent_qsa_train, phys_q_train, _, agent_q_train, agent_actions_train, agent_pi_e_train = self.do_eval_iql(eval_type='train')
        agent_qsa_val, phys_q_val, _, agent_q_val, agent_actions_val, agent_pi_e_val= self.do_eval_iql(eval_type='val')
        agent_qsa_test, phys_q_test, _, agent_q_test, agent_actions_test, agent_pi_e_test = self.do_eval_iql(eval_type='test')
        return agent_qsa_train, agent_qsa_val, agent_qsa_test

    def hard_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)
        
    def soft_update(self, local_model , target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def process_train_batch(self, size):
        if self.per_flag:
            # uses prioritised exp replay
            a = self.df.sample(n=size, weights=self.df['prob'])
        else:
            a = self.df.sample(n=size)
        states = None
        actions = None
        rewards = None
        next_states = None
        done_flags = None
        for i in a.index:
            cur_state = a.loc[i, self.state_features]
            iv = int(a.loc[i, 'iv_input'])
            vaso = int(a.loc[i, 'vaso_input'])
            action = action_map[iv, vaso]
            reward = a.loc[i, 'reward']

            if clip_reward:
                if reward > 1: reward = 1
                if reward < -1: reward = -1

            if i != self.df.index[-1]:
                # if not terminal step in trajectory
                if self.df.loc[i, 'icustayid'] == self.df.loc[i + 1, 'icustayid']:
                    next_state = self.df.loc[i + 1, self.state_features]
                    done = 0
                else:
                    # trajectory is finished
                    next_state = np.zeros(len(cur_state))
                    done = 1
            else:
                # last entry in df is the final state of that trajectory
                next_state = np.zeros(len(cur_state))
                done = 1

            if states is None:
                states = copy.deepcopy(cur_state)
            else:
                states = np.vstack((states, cur_state))

            if actions is None:
                actions = [action]
            else:
                actions = np.vstack((actions, action))

            if rewards is None:
                rewards = [reward]
            else:
                rewards = np.vstack((rewards, reward))

            if next_states is None:
                next_states = copy.deepcopy(next_state)
            else:
                next_states = np.vstack((next_states, next_state))

            if done_flags is None:
                done_flags = [done]
            else:
                done_flags = np.vstack((done_flags, done))
        # return (states, np.squeeze(actions), np.squeeze(rewards), next_states, np.squeeze(done_flags), a)
        return (torch.tensor(states).to(self.device, dtype=torch.float32), torch.tensor(np.squeeze(actions), dtype=torch.int32).to(self.device), torch.tensor(np.squeeze(rewards)).to(self.device,dtype=torch.float32), torch.tensor(next_states).to(self.device, dtype=torch.float32), torch.tensor(np.squeeze(done_flags)).to(self.device), a)
    # extract chunks of length size from the relevant dataframe, and yield these to the caller
    def process_eval_batch(self, size, eval_type=None):
        if eval_type is None:
            raise Exception('Provide eval_type to process_eval_batch')
        elif eval_type == 'train':
            a = self.df.copy()
        elif eval_type == 'val':
            a = self.val_df.copy()
        elif eval_type == 'test':
            a = self.test_df.copy()
        else:
            raise Exception('Unknown eval_type')
        count = 0
        while count < len(a.index):
            states = None
            actions = None
            rewards = None
            next_states = None
            done_flags = None

            start_idx = count
            end_idx = min(len(a.index), count + size)
            segment = a.index[start_idx:end_idx]

            for i in segment:
                cur_state = a.loc[i, self.state_features]
                iv = int(a.loc[i, 'iv_input'])
                vaso = int(a.loc[i, 'vaso_input'])
                action = action_map[iv, vaso]
                reward = a.loc[i, 'reward']

                if clip_reward:
                    if reward > 1: reward = 1
                    if reward < -1: reward = -1

                if i != a.index[-1]:
                    # if not terminal step in trajectory
                    if a.loc[i, 'icustayid'] == a.loc[i + 1, 'icustayid']:
                        next_state = a.loc[i + 1, self.state_features]
                        done = 0
                    else:
                        # trajectory is finished
                        next_state = np.zeros(len(cur_state))
                        done = 1
                else:
                    # last entry in df is the final state of that trajectory
                    next_state = np.zeros(len(cur_state))
                    done = 1

                if states is None:
                    states = copy.deepcopy(cur_state)
                else:
                    states = np.vstack((states, cur_state))

                if actions is None:
                    actions = [action]
                else:
                    actions = np.vstack((actions, action))

                if rewards is None:
                    rewards = [reward]
                else:
                    rewards = np.vstack((rewards, reward))

                if next_states is None:
                    next_states = copy.deepcopy(next_state)
                else:
                    next_states = np.vstack((next_states, next_state))

                if done_flags is None:
                    done_flags = [done]
                else:
                    done_flags = np.vstack((done_flags, done))

            yield (torch.tensor(states).to(self.device,dtype=torch.float32), torch.tensor(np.squeeze(actions)).to(self.device,dtype=torch.int32), torch.tensor(np.squeeze(rewards)).to(self.device), torch.tensor(next_states).to(self.device,dtype=torch.float32), torch.tensor(np.squeeze(done_flags)).to(self.device), a)
            count += size


    def d3qn_softmax(self, x, axis=1):
        x_stack = torch.stack(x, dim=0)
        return torch.softmax(x_stack, dim=axis)

    def WDR_estimator(self, Pi_e, Pi_b, Q_agent, df_test):
        unique_ids = df_test['icustayid'].unique()
        rho_all = []
        DR = []
        V_WDR = 0
        ind = 0
        for uid in unique_ids:
            rho = []
            traj = df_test.loc[df_test['icustayid'] == uid]
            for t in range(len(traj)):
                iv = df_test.loc[ind, 'iv_input']
                vaso = df_test.loc[ind, 'vaso_input']
                phys_action = action_map[(iv, vaso)] 
                if np.isclose(Pi_b[ind][phys_action], 0.0):
                    rho_t = Pi_e[ind][phys_action] / 0.001
                else:
                    rho_t = Pi_e[ind][phys_action] / Pi_b[ind][phys_action]  # df_test['phys_prob'][ind] 
                rho.append(rho_t)
                ind += 1
            rho_all.append(rho)
        max_H = max(len(rho) for rho in rho_all)
        rho_cum = torch.zeros((len(unique_ids), max_H))
        for i, rho in enumerate(rho_all):
            rho_tmp = torch.ones(max_H)
            index = 0
            for rho_index in rho:
                rho_tmp[index] = rho_index
                index += 1
            rho_cum[i] = torch.cumprod(rho_tmp, dim=0)  # 累乘（uids，H）
        weights = torch.mean(rho_cum, axis=0)

        ind = 0
        n_traj = 0
        for uid in unique_ids:
            trajectory = df_test.loc[df_test['icustayid'] == uid]
            rho_cumulative = rho_cum[n_traj]
            V_WDR = 0
            for t in range(len(trajectory)):
                iv = df_test.loc[ind, 'iv_input']
                vaso = df_test.loc[ind, 'vaso_input']
                phys_action = action_map[(iv, vaso)] 
                Q_hat = Q_agent[ind][phys_action]  # test_set <s,a,r>
                V_hat = torch.sum(Q_agent[ind] * Pi_e[ind], dim=0)
                # V_hat = Q_agent[ind].max()
                r_t = df_test['reward'][ind]
                rho_1t = rho_cumulative[t] / weights[t]
                if t == 0:
                    rho_1t_1 = 1.0
                else:
                    rho_1t_1 = rho_cumulative[t - 1] / weights[t - 1]
                # V_WDR = V_WDR + np.power(gamma, t) * (rho_1t * r_t - (rho_1t * Q_hat - rho_1t_1 * V_hat))
                V_WDR = V_WDR + torch.pow(torch.tensor(self.gamma), t) * (rho_1t * r_t - (rho_1t * Q_hat - rho_1t_1 * V_hat))
                ind += 1
            DR.append(V_WDR)
            n_traj += 1
        return DR
        

# The main training loop is here
def main(args):
    av_q_list = []
    WDR_estimator_list = []
    num_actions = 25

    # load dataset
    file_path = os.path.join(args.data_file, 'state_features.txt')
    with open(file_path) as f:
        state_features = f.read().split()
    print (state_features)

    train_data_path = os.path.join(args.data_file, 'rl_train_data_final_cont.csv')
    val_data_path = os.path.join(args.data_file, 'rl_val_data_final_cont.csv')
    test_data_path = os.path.join(args.data_file, 'rl_test_data_final_cont.csv')

    df = pd.read_csv(train_data_path)
    val_df = pd.read_csv(val_data_path)
    test_df = pd.read_csv(test_data_path)

    phy_data_path = os.path.join(args.data_file, 'test_policy_KNN.p')
    phys_policy_test_KNN = pickle.load(open(phy_data_path, "rb"))
    Pi_b = phys_policy_test_KNN
    

    beta_start = 0.9
    df['prob'] = abs(df['reward'])
    temp = 1.0/df['prob']
    df['imp_weight'] = pow((1.0/len(df) * temp), beta_start)

    state_dim = len(state_features)
    policy = DiscreteIQL(
            state_dim=state_dim,
            num_actions=num_actions,
            df=df,
            val_df=val_df,
            test_df=test_df,
            state_features=state_features,
            **vars(args), 
        )

    # Make a path for our model to be saved in.
    if not os.path.exists(args.save_file):
        os.makedirs(args.save_file)

    if load_model == True:
        print('Trying to load model...')
        # policy.load_policy(save_model_path)

    net_loss = 0.0
    for i in tqdm(range(args.num_steps)):
        if save_results:
            print("Calling do save results")
            policy.do_save_results()
            break

        states, actions, rewards, next_states, done_flags, sampled_df = policy.process_train_batch(batch_size)
        

        # Calculate the importance sampling weights for PER
        imp_sampling_weights = np.array(sampled_df['imp_weight'] / float(max(df['imp_weight'])))
        imp_sampling_weights[np.isnan(imp_sampling_weights)] = 1
        imp_sampling_weights[imp_sampling_weights <= 0.001] = 0.001


        # Train with the batch
        error = policy.train(states, actions, rewards, next_states, done_flags, imp_sampling_weights)

        net_loss += sum(error)

        # Set the selection weight/prob to the abs prediction error and update the importance sampling weight
        new_weights = pow((error + per_epsilon), per_alpha)
        df.loc[df.index.isin(sampled_df.index), 'prob'] = new_weights.cpu().numpy()
        temp = 1.0 / new_weights.cpu().numpy()
        df.loc[df.index.isin(sampled_df.index), 'imp_weight'] = pow(((1.0 / len(df)) * temp), beta_start)

        if i % 1000 == 0  :             # WDR 评估
            print("WDR Evaluation")
            agent_qsa, _, _, agent_q_test, agent_actions_test, Pi_e = policy.save_results('test')
            WDR = policy.WDR_estimator(Pi_e, Pi_b, agent_qsa, test_df)
            WDR_clipped = torch.clamp(torch.tensor(WDR), min=-15, max=15)
            mean_WDR = torch.mean(torch.where(torch.isnan(WDR_clipped), torch.tensor(0.0), WDR_clipped))
            print('step:',i, 'WDR:',mean_WDR)
            WDR_estimator_list.append(mean_WDR)

    filename = f"WDR_list_expectile_{args.expectile}_temperature_{args.temperature}_per_{args.per_flag}_reg_{args.reg_flag}_actmax_{args.act_max_flag}_huber_{args.huber_flag}.p"
    with open(args.save_file + filename, 'wb') as f:
        pickle.dump(WDR_estimator_list, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser = add_data_specific_args(parser)
    parser.add_argument('--num_steps', type=int, default=70000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--tau", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expectile", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--per_flag", action="store_true")
    parser.add_argument("--reg_flag", action="store_true")
    parser.add_argument("--act_max_flag", action="store_true")
    parser.add_argument("--huber_flag", action="store_true")
    parser.add_argument("--data_file", type=str,help='Path input data file')
    parser.add_argument("--save_file", type=str,help='Path save data file')

    args = parser.parse_args()
    main(args)