import argparse

def parameter_parser():

    parser = argparse.ArgumentParser(description="SAIRL")

    parser.add_argument("--Baselines",
                        type=list,
                        default=['Random', 'Round-Robin', 'Earliest', 'DQN'],
                        help="Experiment Baseline")

    parser.add_argument("--Baseline_num",
                        type=int,
                        default=4,
                        help="Number of baselines")

    parser.add_argument("--Epoch",
                        type=int,
                        default=10,
                        help="Training Epochs")

    parser.add_argument("--Dqn_start_learn",
                        type=int,
                        default=500,
                        help="Iteration start Learn for normal dqn")

    parser.add_argument("--Dqn_learn_interval",
                        type=int,
                        default=1,
                        help="DQN learning interval")

    parser.add_argument("--Lr_DQN",
                        type=float,
                        default=0.001,
                        help="DQN learning rate")

    parser.add_argument("--CP_Type",
                        type=list,
                        default=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                        help="Charge Piles Type")

    parser.add_argument("--CP_Cost",
                        type=list,
                        default=[1, 1, 4, 4, 6, 1, 1, 4, 4, 6],
                        help="Charge Piles Loss Cost")

    parser.add_argument("--CP_Acc",
                        type=list,
                        default=[1, 1, 1.1, 1.1, 1.2, 1, 1, 1.1, 1.1, 1.2],
                        help="Charge Piles charging power")

    parser.add_argument("--CP_Num",
                        type=int,
                        default=10,
                        help="The number of Charge Piles")

    parser.add_argument("--CP_capacity",
                        type=int,
                        default=1000,
                        help="Charge Piles capacity")

    parser.add_argument("--lamda",
                        type=int,
                        default=15,
                        help="The parameter used to control the length of each charging jobs.")

    parser.add_argument("--Job_Type",
                        type=float,
                        default=0.5,
                        help="The parameter used to control the type of each charging jobs.")

    parser.add_argument("--Job_Num",
                        type=int,
                        default=8000,
                        help="The total number of charging jobs.")

    parser.add_argument("--Job_len_Mean",
                        type=int,
                        default=200,
                        help="The mean value of the normal distribution.")

    parser.add_argument("--Job_len_Std",
                        type=int,
                        default=20,
                        help="The std value of the normal distribution.")

    parser.add_argument("--Job_ddl",
                        type=float,
                        default=0.25,
                        help="Deadline time of each charging jobs")

    return parser.parse_args()
