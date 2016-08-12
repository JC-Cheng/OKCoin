import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import matplotlib.dates as md
import pytz

def rand_scatter(N):

	x = np.random.rand(N)
	y = np.random.rand(N)
	colors = np.random.rand(N)
	area = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radiuses

	plt.scatter(x, y, s=area, c=colors, alpha=0.5)
	plt.show()
	

def mDF_to_scatter(mDF, n):

	t0 = mDF['date'][0]
	P = []
	Q = []
	T = []
	
	for i in range(0, n):
		subDF = mDF.loc[mDF['date'] == t0 + timedelta(seconds=i)]
		PQ = subDF.groupby('price').sum()
		
		for q in PQ['amount']:
			Q.append(q)
			P.append(PQ.loc[PQ['amount'] == q].index[0])
			T.append(t0 + timedelta(seconds=i))
			
	orders_scatter(P, Q, T)
	
	
def orders_scatter(P, Q, T):

	colors = np.random.rand(len(P))
	T0 = T[1] - timedelta(seconds=1)
	
	t = []
	for tm in T:
		t.append(int((tm - T0).seconds))
		
	plt.scatter(t, P, s= [200 * q ** 0.5 for q in Q], c=colors, alpha=0.5, edgecolors='face')
	plt.xlim(min(t) - 1, max(t) + 1)
	
	plt.show()