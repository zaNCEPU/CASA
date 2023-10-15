import numpy as np
from scipy import stats

np.random.seed(3)

class SchedulingEnv:
    def __init__(self, args):
        # Environment Settings
        self.policy_num = len(args.Baselines)

        # charge pile
        self.CPtypes = args.CP_Type
        self.CPnum = args.CP_Num
        assert self.CPnum == len(self.CPtypes)
        self.CPcapacity = args.CP_capacity
        self.actionNum = args.CP_Num
        self.s_features = 1 + args.CP_Num
        self.CPAcc = args.CP_Acc
        self.CPCost = args.CP_Cost

        # EV
        self.jobMI = args.Job_len_Mean
        self.jobMI_std = args.Job_len_Std
        self.job_type = args.Job_Type
        self.jobNum = args.Job_Num
        self.lamda = args.lamda
        self.arrival_Times = np.zeros(self.jobNum)
        self.jobsMI = np.zeros(self.jobNum)
        self.lengths = np.zeros(self.jobNum)
        self.types = np.zeros(self.jobNum)
        self.ddl = np.ones(self.jobNum) * args.Job_ddl
        # generate workload
        self.gen_workload(self.lamda)

        # 1-chargerID  2-start time  3-wait time  4-waitT+exeT  5-leave time
        # 6-reward  7-actual_exeT  8- success  9-reject

        # Random
        self.RAN_events = np.zeros((9, self.jobNum))
        self.RAN_CP_events = np.zeros((2, self.CPnum))
        # Round Robin
        self.RR_events = np.zeros((9, self.jobNum))
        self.RR_CP_events = np.zeros((2, self.CPnum))
        # Earliest
        self.early_events = np.zeros((9, self.jobNum))
        self.early_CP_events = np.zeros((2, self.CPnum))
        # DQN
        self.DQN_events = np.zeros((9, self.jobNum))
        self.DQN_CP_events = np.zeros((2, self.CPnum))


    def gen_workload(self, lamda):
        intervalT = stats.expon.rvs(scale=1 / lamda, size=self.jobNum)
        print("intervalT mean: ", round(np.mean(intervalT), 3),
              '  intervalT SD:', round(np.std(intervalT, ddof=1), 3))

        self.arrival_Times = np.around(intervalT.cumsum(), decimals=3)
        last_arrivalT = self.arrival_Times[- 1]
        print('last EV arrivalT:', round(last_arrivalT, 3))

        self.jobsMI = np.random.normal(self.jobMI, self.jobMI_std, self.jobNum)
        self.jobsMI = self.jobsMI.astype(int)
        print("Demand mean: ", round(np.mean(self.jobsMI), 3), '  Demand SD:', round(np.std(self.jobsMI, ddof=1), 3))
        self.lengths = self.jobsMI / self.CPcapacity
        print("exeT mean: ", round(np.mean(self.lengths), 3), '  exeT SD:', round(np.std(self.lengths, ddof=1), 3))

        types = np.zeros(self.jobNum)
        for i in range(self.jobNum):
            if np.random.uniform() < self.job_type:
                types[i] = 0  # urgent EV user
            else:
                types[i] = 1  # un-urgent EV user
        self.types = types

    def reset(self, args):
        # if each episode generates new workload
        self.arrival_Times = np.zeros(self.jobNum)
        self.jobsMI = np.zeros(self.jobNum)
        self.lengths = np.zeros(self.jobNum)
        self.types = np.zeros(self.jobNum)
        self.ddl = np.ones(self.jobNum) * args.Job_ddl
        self.job_type = args.Job_Type
        self.gen_workload(args.lamda)
        self.CPtypes = args.CP_Type

        # reset all records
        self.SAIRL_events = np.zeros((9, self.jobNum))
        self.SAIRL_CP_events = np.zeros((2, self.CPnum))
        self.RAN_events = np.zeros((9, self.jobNum))
        self.RAN_CP_events = np.zeros((2, self.CPnum))
        self.RR_events = np.zeros((9, self.jobNum))
        self.RR_CP_events = np.zeros((2, self.CPnum))
        self.early_events = np.zeros((9, self.jobNum))
        self.early_CP_events = np.zeros((2, self.CPnum))
        self.DQN_events = np.zeros((9, self.jobNum))
        self.DQN_CP_events = np.zeros((2, self.CPnum))
        self.suit_events = np.zeros((9, self.jobNum))
        self.suit_CP_events = np.zeros((2, self.CPnum))
        self.sensible_events = np.zeros((9, self.jobNum))
        self.sensible_CP_events = np.zeros((2, self.CPnum))


    def workload(self, EV_count):
        arrival_time = self.arrival_Times[EV_count - 1]
        length = self.lengths[EV_count - 1]
        jobType = self.types[EV_count - 1]
        ddl = self.ddl[EV_count - 1]
        if EV_count == self.jobNum:
            finish = True
        else:
            finish = False
        job_attributes = [EV_count - 1, arrival_time, length, jobType, ddl]
        return finish, job_attributes


    def feedback(self, job_attrs, action, policyID):
        job_id = job_attrs[0]
        arrival_time = job_attrs[1]
        length = job_attrs[2]
        job_type = job_attrs[3]
        ddl = job_attrs[4]  # QOS
        acc = self.CPAcc[action]  #
        cost = self.CPCost[action]  #
        if job_type == self.CPtypes[action]:
            real_length = length / acc
        else:
            real_length = (length * 2) / acc

        if policyID == 1:
            idleT = self.RAN_CP_events[0, action]
        elif policyID == 2:
            idleT = self.RR_CP_events[0, action]
        elif policyID == 3:
            idleT = self.early_CP_events[0, action]
        elif policyID == 4:
            idleT = self.DQN_CP_events[0, action]

        # waitT & start exeT
        if idleT <= arrival_time:  # if no waitT
            waitT = 0
            startT = arrival_time
        else:
            waitT = idleT - arrival_time
            startT = idleT

        TOU_3 = 0.3946
        TOU_6 = 0.6950
        TOU_9 = 1.0044
        TOU_12 = 0.6950
        TOU_15 = 1.0044
        TOU_18 = 0.6950
        TOU_21 = 0.3946
        t = arrival_time % 24
        if 0 <= t < 6:
            price = TOU_3
        if 6 <= t < 9:
            price = TOU_6
        if 9 <= t < 12:
            price = TOU_9
        if 12 <= t < 15:
            price = TOU_12
        if 15 <= t < 18:
            price = TOU_15
        if 18 <= t < 21:
            price = TOU_18
        if 21 <= t < 24:
            price = TOU_21

        durationT = waitT + real_length  # waitT+exeT
        leaveT = startT + real_length  # leave T
        new_idleT = leaveT  # update charger idle time

        # reward
        cost = real_length * cost + length * price
        reward = (1 + np.exp(1.5 - cost)) * length / durationT
        # reward = (1/cost) * (length) / durationT
        # reward = - durationT / length
        # reward = (length) / (durationT * cost)

        # whether success
        success = 1 if durationT <= ddl else 0

        if policyID == 1:  # random
            self.RAN_events[0, job_id] = action
            self.RAN_events[2, job_id] = waitT
            self.RAN_events[1, job_id] = startT
            self.RAN_events[3, job_id] = durationT
            self.RAN_events[4, job_id] = leaveT
            self.RAN_events[5, job_id] = reward
            self.RAN_events[6, job_id] = real_length
            self.RAN_events[7, job_id] = success
            self.RAN_events[8, job_id] = cost + np.random.rand() / 100
            # update charger info
            self.RAN_CP_events[1, action] += 1
            self.RAN_CP_events[0, action] = new_idleT
            # print('CP_after:', self.RAN_CP_events[0, action])
        elif policyID == 2:  # Round-Robin
            self.RR_events[0, job_id] = action
            self.RR_events[2, job_id] = waitT
            self.RR_events[1, job_id] = startT
            self.RR_events[3, job_id] = durationT
            self.RR_events[4, job_id] = leaveT
            self.RR_events[5, job_id] = reward
            self.RR_events[6, job_id] = real_length
            self.RR_events[7, job_id] = success
            self.RR_events[8, job_id] = cost + np.random.rand() / 100
            # update charger info
            self.RR_CP_events[1, action] += 1
            self.RR_CP_events[0, action] = new_idleT
        elif policyID == 3:  # Earliest
            self.early_events[0, job_id] = action
            self.early_events[2, job_id] = waitT
            self.early_events[1, job_id] = startT
            self.early_events[3, job_id] = durationT
            self.early_events[4, job_id] = leaveT
            self.early_events[5, job_id] = reward
            self.early_events[6, job_id] = real_length
            self.early_events[7, job_id] = success
            self.early_events[8, job_id] = cost + np.random.rand() / 100
            # update charger info
            self.early_CP_events[1, action] += 1
            self.early_CP_events[0, action] = new_idleT
        elif policyID == 4:  # DQN
            self.DQN_events[0, job_id] = action
            self.DQN_events[2, job_id] = waitT
            self.DQN_events[1, job_id] = startT
            self.DQN_events[3, job_id] = durationT
            self.DQN_events[4, job_id] = leaveT
            self.DQN_events[5, job_id] = reward
            self.DQN_events[6, job_id] = real_length
            self.DQN_events[7, job_id] = success
            self.DQN_events[8, job_id] = cost + np.random.rand() / 100
            # update charger info
            self.DQN_CP_events[1, action] += 1
            self.DQN_CP_events[0, action] = new_idleT
        return reward


    def get_CP_idleT(self, policyID):
        if policyID == 3:
            idleTimes = self.early_CP_events[0, :]
        elif policyID == 4:
            idleTimes = self.DQN_CP_events[0, :]
        elif policyID == 5:
            idleTimes = self.suit_CP_events[0, :]
        elif policyID == 1:
            idleTimes = self.RAN_CP_events[0, :]
        elif policyID == 7:
            idleTimes = self.SAIRL_CP_events[0, :]
        elif policyID == 2:
            idleTimes = self.RR_CP_events[0, :]
        return idleTimes


    def getState(self, job_attrs, policyID):
        arrivalT = job_attrs[1]
        length = job_attrs[2]
        job_type = job_attrs[3]
        state_job = [job_type]
        if policyID == 4:  # DQN
            idleTimes = self.get_CP_idleT(4)
        elif policyID == 1:  # random
            idleTimes = self.get_CP_idleT(1)
        elif policyID == 2:  # RR
            idleTimes = self.get_CP_idleT(2)
        elif policyID == 3:  # used for as
            idleTimes = self.get_CP_idleT(3)
        elif policyID == 5:  # suitable
            idleTimes = self.get_CP_idleT(5)
        elif policyID == 7:  # SAIRL
            idleTimes = self.get_CP_idleT(7)
        waitTimes = [t - arrivalT for t in idleTimes]
        waitTimes = np.maximum(waitTimes, 0)
        state = np.hstack((state_job, waitTimes))  # [job_type,CP_waitime]
        # print("state:",state)
        return state


    def get_accumulateRewards(self, policies, start, end):

        rewards = np.zeros(policies)
        rewards[0] = sum(self.RAN_events[5, start:end])
        rewards[1] = sum(self.RR_events[5, start:end])
        rewards[2] = sum(self.early_events[5, start:end])
        rewards[3] = sum(self.DQN_events[5, start:end])
        return np.around(rewards, 2)

    def get_accumulateCost(self, policies, start, end):

        Cost = np.zeros(policies)
        Cost[0] = sum(self.RAN_events[8, start:end])
        Cost[1] = sum(self.RR_events[8, start:end])
        Cost[2] = sum(self.early_events[8, start:end])
        Cost[3] = sum(self.DQN_events[8, start:end])
        return np.around(Cost, 2)

    def get_FinishTimes(self, policies, start, end):
        finishT = np.zeros(policies)
        finishT[0] = max(self.RAN_events[4, start:end])
        finishT[1] = max(self.RR_events[4, start:end])
        finishT[2] = max(self.early_events[4, start:end])
        finishT[3] = max(self.DQN_events[4, start:end])
        return np.around(finishT, 2)

    def get_executeTs(self, policies, start, end):
        executeTs = np.zeros(policies)
        executeTs[0] = np.mean(self.RAN_events[6, start:end])
        executeTs[1] = np.mean(self.RR_events[6, start:end])
        executeTs[2] = np.mean(self.early_events[6, start:end])
        executeTs[3] = np.mean(self.DQN_events[6, start:end])
        return np.around(executeTs, 3)

    def get_waitTs(self, policies, start, end):
        waitTs = np.zeros(policies)
        waitTs[0] = np.mean(self.RAN_events[2, start:end])
        waitTs[1] = np.mean(self.RR_events[2, start:end])
        waitTs[2] = np.mean(self.early_events[2, start:end])
        waitTs[3] = np.mean(self.DQN_events[2, start:end])
        return np.around(waitTs, 3)

    def get_responseTs(self, policies, start, end):
        respTs = np.zeros(policies)
        respTs[0] = np.mean(self.RAN_events[3, start:end])
        respTs[1] = np.mean(self.RR_events[3, start:end])
        respTs[2] = np.mean(self.early_events[3, start:end])
        respTs[3] = np.mean(self.DQN_events[3, start:end])
        return np.around(respTs, 3)

    def get_successTimes(self, policies, start, end):
        successT = np.zeros(policies)
        successT[0] = sum(self.RAN_events[7, start:end]) / (end - start)
        successT[1] = sum(self.RR_events[7, start:end]) / (end - start)
        successT[2] = sum(self.early_events[7, start:end]) / (end - start)
        successT[3] = sum(self.DQN_events[7, start:end]) / (end - start)
        successT = np.around(successT, 3)
        return successT

    def get_rejectTimes(self, policies, start, end):
        reject = np.zeros(policies)
        reject[0] = sum(self.RAN_events[8, start:end])
        reject[1] = sum(self.RR_events[8, start:end])
        reject[2] = sum(self.early_events[8, start:end])
        reject[3] = sum(self.DQN_events[8, start:end])
        return np.around(reject, 2)

    def get_totalRewards(self, policies, start):
        rewards = np.zeros(policies)
        rewards[0] = sum(self.RAN_events[5, start:self.jobNum])
        rewards[1] = sum(self.RR_events[5, start:self.jobNum])
        rewards[2] = sum(self.early_events[5, start:self.jobNum])
        rewards[3] = sum(self.DQN_events[5, start:self.jobNum])
        return np.around(rewards, 2)

    def get_totalTimes(self, policies, start):
        finishT = np.zeros(policies)
        finishT[0] = max(self.RAN_events[4, :]) - self.arrival_Times[start]
        finishT[1] = max(self.RR_events[4, :]) - self.arrival_Times[start]
        finishT[2] = max(self.early_events[4, :]) - self.arrival_Times[start]
        finishT[3] = max(self.DQN_events[4, :]) - self.arrival_Times[start]
        return np.around(finishT, 2)

    def get_avgUtilitizationRate(self, policies, start):
        avgRate = np.zeros(policies)  # sum(real_length)/ totalT*CPnum
        avgRate[0] = sum(self.RAN_events[6, start:self.jobNum]) / (
                (max(self.RAN_events[4, :]) - self.arrival_Times[start]) * self.CPnum)
        avgRate[1] = sum(self.RR_events[6, start:self.jobNum]) / (
                (max(self.RR_events[4, :]) - self.arrival_Times[start]) * self.CPnum)
        avgRate[2] = sum(self.early_events[6, start:self.jobNum]) / (
                (max(self.early_events[4, :]) - self.arrival_Times[start]) * self.CPnum)
        avgRate[3] = sum(self.DQN_events[6, start:self.jobNum]) / (
                (max(self.DQN_events[4, :]) - self.arrival_Times[start]) * self.CPnum)
        return np.around(avgRate, 3)

    def get_all_responseTs(self, policies):
        respTs = np.zeros((policies, self.jobNum))
        respTs[0, :] = self.RAN_events[3, :]
        respTs[1, :] = self.RR_events[3, :]
        respTs[2, :] = self.early_events[3, :]
        respTs[3, :] = self.DQN_events[3, :]
        return np.around(respTs, 3)

    def get_total_responseTs(self, policies, start):
        respTs = np.zeros(policies)
        respTs[0] = np.mean(self.RAN_events[3, start:self.jobNum])
        respTs[1] = np.mean(self.RR_events[3, start:self.jobNum])
        respTs[2] = np.mean(self.early_events[3, start:self.jobNum])
        respTs[3] = np.mean(self.DQN_events[3, start:self.jobNum])
        return np.around(respTs, 3)

    def get_totalSuccess(self, policies, start):
        successT = np.zeros(policies)  # sum(self.RAN_events[7, 3000:-1])/(self.jobNum - 3000)
        successT[0] = sum(self.RAN_events[7, start:self.jobNum]) / (self.jobNum - start + 1)
        successT[1] = sum(self.RR_events[7, start:self.jobNum]) / (self.jobNum - start + 1)
        successT[2] = sum(self.early_events[7, start:self.jobNum]) / (self.jobNum - start + 1)
        successT[3] = sum(self.DQN_events[7, start:self.jobNum]) / (self.jobNum - start + 1)
        return np.around(successT, 3)

    def get_totalCost(self, policies, start):

        Cost = np.zeros(policies)
        Cost[0] = sum(self.RAN_events[8, start:self.jobNum]) / (self.jobNum - start + 1)
        Cost[1] = sum(self.RR_events[8, start:self.jobNum]) / (self.jobNum - start + 1)
        Cost[2] = sum(self.early_events[8, start:self.jobNum]) / (self.jobNum - start + 1)
        Cost[3] = sum(self.DQN_events[8, start:self.jobNum]) / (self.jobNum - start + 1)
        return np.around(Cost, 3)
