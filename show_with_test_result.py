import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt

file_list = glob.glob('./log/*/*/logs.mat')
# file_list = [name for name in file_list if 'WRA' in name]
file_list = [name for name in file_list if 'GAN-I' in name]
# file_list = [name for name in file_list if 'R_0.1__' in name]
# mode = 'full'  # full, all
mode = 'test'  # full, all
legend_lable = []

colormap = plt.cm.gist_ncar
# plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, len(file_list))])

for name in file_list:
    returns = loadmat(name)['test_returns']
    print(name + '\n')
    print(returns)
    print('\n\n')
    # import pdb; pdb.set_trace()
    if mode in ['test', 'all']:
        plt.plot(returns.reshape([-1]))
        legend_lable.append('test_' + name.split('/')[2])

    returns = loadmat(name)['returns']
    if mode in ['full', 'all']:
        plt.plot(returns.reshape([-1]))
        legend_lable.append('full_' + name.split('/')[2])

plt.legend(legend_lable)
plt.show()
