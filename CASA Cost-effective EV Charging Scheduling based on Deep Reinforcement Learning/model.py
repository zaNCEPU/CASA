import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

random.seed(6)
np.random.seed(6)

class NET(nn.Module):
    def __init__(self, n_actions, n_features):
        super(NET, self).__init__()
        self.fc1 = nn.Linear(n_features, 10)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.fc2 = nn.Linear(10, 10)
        self.out = nn.Linear(10, n_actions)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.relu(self.fc2(x))
        q_evaluate = self.out(x)

        return q_evaluate


class baseline_DQN():
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=50,
            memory_size=800,
            batch_size=30,
            e_greedy_increment=0.006,
            # output_graph=False,
    ):
        self.n_actions = n_actions  # if +1: allow to reject jobs
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0.01 if e_greedy_increment is not None else self.epsilon_max

        self.learn_step_counter = 0  # total learning step
        self.replay_buffer = deque()  # init experience replay [s, a, r, s_, done]

        # consist of [target_net, evaluate_net]
        self.eval_net, self.target_net = NET(n_actions, n_features), NET(n_actions, n_features)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        pro = np.random.uniform()
        if pro < self.epsilon:
            actions_value = self.eval_net.forward(state)
            action = torch.max(actions_value, 1)[1]
            # print('pro: ', pro, ' q-values:', actions_value, '  best_action:', action)
            # print('  best_action:', action)
        else:
            action = np.random.randint(0, self.n_actions)
            # print('pro: ', pro, '  rand_action:', action)
            # print('  rand_action:', action)
        return action

    def store_transition(self, s, a, r, s_):
        one_hot_action = np.zeros(self.n_actions)
        one_hot_action[a] = 1
        self.replay_buffer.append((s, one_hot_action, r, s_))
        if len(self.replay_buffer) > self.memory_size:
            self.replay_buffer.popleft()

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:  # 50
            self.target_net.load_state_dict(self.eval_net.state_dict())
            # print('-------------target_params_replaced------------------')

        # sample batch memory from all memory: [s, a, r, s_]
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        b_s = torch.FloatTensor([data[0] for data in minibatch])
        b_a = []

        for data in minibatch:
            for i in range(0, len(data[1])):
                if (data[1][i] == 1):
                    b_a.append(i)
        b_a = torch.LongTensor(b_a)
        b_a = b_a.reshape(30, 1)
        b_r = torch.FloatTensor([data[2] for data in minibatch]).reshape(30, 1)
        b_s_ = torch.FloatTensor([data[3] for data in minibatch])

        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + self.gamma * q_next.max(1)[0].reshape(30, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # increasing epsilon
        if self.epsilon < self.epsilon_max:
            self.epsilon = self.epsilon + self.epsilon_increment
        else:
            self.epsilon = self.epsilon_max
        # print('epsilon:', self.epsilon)
        self.learn_step_counter += 1


class baselines:
    def __init__(self, n_actions, CPtypes):
        self.n_actions = n_actions
        self.CPtypes = np.array(CPtypes)  # change list to numpy

        # parameters for sensible policy
        self.sensible_updateT = 5
        self.sensible_counterT = 1
        self.sensible_discount = 0.7  # 0.7 is best, 0.5 and 0.6 OK
        self.sensible_W = np.zeros(self.n_actions)
        self.sensible_probs = np.ones(self.n_actions) / self.n_actions
        self.sensible_probsCumsum = self.sensible_probs.cumsum()
        self.sensible_sumDurations = np.zeros((2, self.n_actions))  # row 1: jobNum   row 2: sum duration

    def random_choose_action(self):  # random policy
        action = np.random.randint(self.n_actions)  # [0, n_actions)
        return action

    def RR_choose_action(self, EV_count):  # round robin policy
        action = (EV_count - 1) % self.n_actions
        return action

    def early_choose_action(self, idleTs):  # earliest policy
        action = np.argmin(idleTs)
        return action
