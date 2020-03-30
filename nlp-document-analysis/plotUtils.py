import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from pylab import rcParams
rcParams['figure.figsize'] = 20, 10


def plot_bar(dfx, dfy, labx, laby, filename):
	# plt.figure(figsize=(20,10))
	fig = plt.figure()
	plt.bar(dfx, dfy)
	plt.xticks(rotation=90)
	# plt.xlabel(labx)
	# plt.ylabel(laby)
	# plt.xticks(dfx, dfx)
	# plt.yticks(dfy)
	# plt.title('Most Used 20 Words')
	plt.tight_layout()
	plt.savefig(filename+'.png')