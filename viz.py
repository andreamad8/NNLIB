from __future__ import print_function
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import pickle
import seaborn as sns

plt.rc('text', usetex=True)
plt.rc('font', family='Times-Roman')
sns.set_style(style='white')

def LRvdMOM():
	plot_arr=pickle.load( open( "Model/LRvsMOM100.p", "rb" ) )
	pair=pickle.load( open( "Model/pairLRvsMOM100.p", "rb" ) )

	fig = plt.figure()
	for i in range(1,17):
		
		ax = fig.add_subplot(4, 4, i)
		plt.title("Learning rate: %s, Momentum: %s"%(pair[i-1][0],pair[i-1][1]))

		ax.plot(plot_arr[i-1],linewidth=1.95, alpha=0.7, color='red')
		ax.set_ylabel("Loss")
		ax.set_xlabel("Epoch")

		ax.yaxis.grid(True)
		inset = inset_axes(ax, width="60%", height="60%")
		inset.plot(plot_arr[i-1],linewidth=1.95, alpha=0.7, color='red')	
		inset.set_xlim([0, 60])
		inset.yaxis.grid(True)		

	mng = plt.get_current_fig_manager()                                         
	mng.window.showMaximized()
	fig.tight_layout()
	plt.show()


def HvsL():
	plot_arr=pickle.load( open( "Model/HUvsLAMBDA03.p", "rb" ) )
	pair=pickle.load( open( "Model/pairHUvsLAMBDA03.p", "rb" ) )

	fig = plt.figure()
	for i in range(1,10):
		
		ax = fig.add_subplot(3, 3, i)
		plt.title("Hidden Unit: %s, Lambda: %s"%(pair[i-1][0],pair[i-1][1]))

		ax.plot(plot_arr[i-1],linewidth=1.95, alpha=0.7, color='red')
		ax.set_ylabel("Loss")
		ax.set_xlabel("Epoch")

		ax.yaxis.grid(True)
		inset = inset_axes(ax, width="60%", height="60%")
		inset.plot(plot_arr[i-1],linewidth=1.95, alpha=0.7, color='red')	
		inset.set_xlim([0, 60])
		inset.yaxis.grid(True)		

	mng = plt.get_current_fig_manager()                                         
	mng.window.showMaximized()
	fig.tight_layout()
	plt.show()


HvsL()