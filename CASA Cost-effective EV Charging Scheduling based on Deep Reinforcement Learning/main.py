import torch
import numpy as np
from env import SchedulingEnv
from model import baseline_DQN, baselines
from utils import get_args, plot_SR_figure

args = get_args()

performance_lamda = np.zeros(args.Baseline_num)
performance_success = np.zeros(args.Baseline_num)
performance_util = np.zeros(args.Baseline_num)
performance_finishT = np.zeros(args.Baseline_num)
performance_cost = np.zeros(args.Baseline_num)

# Environment
env = SchedulingEnv(args)

# DQN agent
brainRL = baseline_DQN(env.actionNum, env.s_features)

# Baselines
brainOthers = baselines(env.actionNum, env.CPtypes)

global_step = 0
my_learn_step = 0
DQN_Reward_list = []
for episode in range(args.Epoch):
    print('----------------------------Episode', episode, '----------------------------')
    job_c = 1
    performance_c = 0
    env.reset(args)
    performance_respTs = []
    while True:
        global_step += 1

        finish, job_attrs = env.workload(job_c)

        DQN_state = env.getState(job_attrs, 4)
        # print(DQN_state)

        if global_step != 1:
            brainRL.store_transition(last_state, last_action, last_reward, DQN_state)
        action_DQN = brainRL.choose_action(DQN_state)  # choose action
        reward_DQN = env.feedback(job_attrs, action_DQN, 4)
        if episode == 1:
            DQN_Reward_list.append(reward_DQN)
        if (global_step > args.Dqn_start_learn) and (global_step % args.Dqn_learn_interval == 0):  # learn
            brainRL.learn()

        last_state = DQN_state
        last_action = action_DQN
        last_reward = reward_DQN

        # random policy
        state_Ran = env.getState(job_attrs, 1)
        action_random = brainOthers.random_choose_action()
        reward_random = env.feedback(job_attrs, action_random, 1)
        # round robin policy
        state_RR = env.getState(job_attrs, 2)
        action_RR = brainOthers.RR_choose_action(job_c)
        reward_RR = env.feedback(job_attrs, action_RR, 2)
        # earliest policy
        idleTimes = env.get_CP_idleT(3)  # get charger state
        action_early = brainOthers.early_choose_action(idleTimes)
        reward_early = env.feedback(job_attrs, action_early, 3)

        if job_c % 500 == 0:
            acc_Rewards = env.get_accumulateRewards(args.Baseline_num, performance_c, job_c)
            cost = env.get_accumulateCost(args.Baseline_num, performance_c, job_c)
            finishTs = env.get_FinishTimes(args.Baseline_num, performance_c, job_c)
            avg_exeTs = env.get_executeTs(args.Baseline_num, performance_c, job_c)
            avg_waitTs = env.get_waitTs(args.Baseline_num, performance_c, job_c)
            avg_respTs = env.get_responseTs(args.Baseline_num, performance_c, job_c)
            performance_respTs.append(avg_respTs)
            successTs = env.get_successTimes(args.Baseline_num, performance_c, job_c)
            performance_c = job_c

        job_c += 1
        if finish:
            break

    # episode performance
    startP = 2000

    total_Rewards = env.get_totalRewards(args.Baseline_num, startP)
    avg_allRespTs = env.get_total_responseTs(args.Baseline_num, startP)
    total_success = env.get_totalSuccess(args.Baseline_num, startP)
    avg_util = env.get_avgUtilitizationRate(args.Baseline_num, startP)
    total_Ts = env.get_totalTimes(args.Baseline_num, startP)
    total_cost = env.get_totalCost(args.Baseline_num, startP)
    print('total performance (after 2000 jobs):')
    for i in range(len(args.Baselines)):
        name = "[" + args.Baselines[i] + "]"
        print(name + " reward:", total_Rewards[i], ' avg_responseT:', avg_allRespTs[i],
              'success_rate:', total_success[i], ' utilizationRate:', avg_util[i], ' finishT:', total_Ts[i], 'Cost:',
              total_cost[i])

    if episode != 0:
        performance_lamda[:] += env.get_total_responseTs(args.Baseline_num, 0)
        performance_success[:] += env.get_totalSuccess(args.Baseline_num, 0)
        performance_util[:] += env.get_avgUtilitizationRate(args.Baseline_num, 0)
        performance_finishT[:] += env.get_totalTimes(args.Baseline_num, 0)
        performance_cost += env.get_totalCost(args.Baseline_num, 0)
print('')

print('---------------------------- Final results ----------------------------')
performance_lamda = np.around(performance_lamda / (args.Epoch - 1), 3)
performance_success = np.around(performance_success / (args.Epoch - 1), 3)
performance_util = np.around(performance_util / (args.Epoch - 1), 3)
performance_finishT = np.around(performance_finishT / (args.Epoch - 1), 3)
performance_cost = np.around(performance_cost / (args.Epoch - 1), 3)
print('avg_responseT:')
print(performance_lamda)
print('success_rate:')
print(performance_success)
print('utilizationRate:')
print(performance_util)
print('finishT:')
print(performance_finishT)
print('Cost:')
print(performance_cost)

