import matplotlib.pyplot as plt
from param_parser import parameter_parser
import numpy as np

def get_args():
    args = parameter_parser()
    args.Baseline_num = len(args.Baselines)
    return args

def plot_SR_figure(SR_result, args, lamda, fig_type):
    assert fig_type in ["SR", "UR", "ART"]
    draw_success = np.array(SR_result) * 100
    draw_success = np.around(draw_success, 1)
    print(draw_success.shape)
    x = list(range(draw_success.shape[0]))
    width = 0.95 / args.Baseline_num
    for i in range(args.Baseline_num):
        if i != 0:
            for j in range(len(x)):
                x[j] = x[j] + width
        bars = plt.bar(x, draw_success[:, i], width=width, label=args.Baselines[i])
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, height, ha='center', va='bottom', fontsize=15)
            bar.set_edgecolor('white')

    plt.xlabel('Mean Arrival Rate', fontsize=30)
    #plt.xlabel('Job Type Proportion', fontsize=25)
    if fig_type == 'UR':
        plt.ylabel('Utilization Rate(%)', fontsize=30)
    elif fig_type == 'SR':
        plt.ylabel('Success Rate(%)', fontsize=30)
    elif fig_type == 'ART':
        plt.ylabel('Average Response Time(%)', fontsize=30)
    x_sticks = np.linspace(0, draw_success.shape[0] - 1, draw_success.shape[0])
    plt.xticks(x_sticks + 5 * width, lamda, fontsize=30)
    plt.yticks(fontsize=30)
    #plt.legend(..., fontsize=20)
    plt.legend(loc='best')
    plt.grid(True, linestyle="-.", linewidth=1)
    plt.show()