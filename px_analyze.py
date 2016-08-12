#import time
from datetime import timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md
import pytz
from matplotlib import gridspec
from pylab import *

def get_summary(df):
	
	vwap = np.cumsum(df['amount'] * df['price']) / np.cumsum(df['amount'])
	
	sd = [0.0]
	dp = [0.0]
	for i in range(1, len(vwap)):
		
		roll_v = roll_w = 0.0		
		for j in range(0, i):
			roll_v = roll_v + pow(df['price'][j] - vwap[i], 2) * df['amount'][j]
			roll_w = roll_w + df['amount'][j]
	
		sd.append(np.sqrt((roll_v * i) / (roll_w * (i - 1))))
		dp.append(df['price'][i] - df['price'][i - 1])

	vwsd = pd.Series(sd, index=np.arange(len(sd)))
	difp = pd.Series(dp, index=np.arange(len(dp)))
	
	return [vwap, vwsd, difp]
	
def get_rolling(df, p=5):

	rolM = df['price'].rolling(p).mean()
	rolS = df['price'].rolling(p).std()
	
	return [rolM, rolS]

#return the alpha*100% highest traded price in the time interval [t1, t2], where t2 > t1
def pct_px(mDF, t1, t2, alpha=0.05):

	sub_df = mDF.loc[mDF['date'] >= t1].loc[mDF['date'] <= t2]
	
	if alpha == 0:
		return sub_df['price'].max()
	elif alpha == 1:
		return sub_df['price'].min()
	else:
		total_Q = sub_df['amount'].sum() * (1 - alpha)
		PQ_DF = sub_df.groupby('price').sum()
		
		agg_q = 0
		for q in PQ_DF['amount']:
	
			if (agg_q <= total_Q) & (total_Q < (agg_q + q)):
				return PQ_DF.loc[PQ_DF['amount'] == q].index[0]
	
			agg_q = agg_q + q
	
def elapsed_time_for_order(df, p, q, order_time, order_type='s'):
	
	#should i use max_p/min_p or average p?
	if order_type == 's':	
		sub_df = df.loc[df['date'] > order_time].loc[df['min_p'] > p]

	elif order_type == 'b':
		sub_df = df.loc[df['date'] > order_time].loc[df['max_p'] > p]
		
	while q > 0:
		
		if sub_df.empty: #order is not fully filled; return zero
			return np.nan
		
		q = q - sub_df.iloc[0]['amount']
		filled_time = sub_df.iloc[0]['date']
		sub_df = sub_df.iloc[1:]
		
	td = filled_time - order_time

	return int(td.seconds)

def elapsed_time_4_order(mDF, p, q, order_time, order_type='s'): #a modified edition using each single order to check if traded
	
	if order_type == 's':
		sub_df = mDF.loc[mDF['date'] > order_time].loc[mDF['price'] > p]

	elif order_type == 'b':
		sub_df = mDF.loc[mDF['date'] > order_time].loc[mDF['price'] > p]
	
	while q > 0:
		
		if sub_df.empty: #order is not fully filled; return zero
			return np.nan
		
		q = q - sub_df.iloc[0]['amount']
		filled_time = sub_df.iloc[0]['date']
		sub_df = sub_df.iloc[1:]
		
	td = filled_time - order_time

	return int(td.seconds)
	
def plot_px(df, Flag):

	z = 1.96

	[ap, sd, dp] = get_summary(df)
	[rm30, rs30] = get_rolling(df, 30)
	[rm2, rs2] = get_rolling(df, 120)
	t = []
	for d in df['date']:
		t.append(md.date2num(d))
	
	fig = plt.figure(figsize=(12, 9))
	gs = gridspec.GridSpec(3, 1, height_ratios=[1, 4, 1])
	plt.rc('axes', grid=True)
	
	ax2 = plt.subplot(gs[2])
	ax2.set_title("Volume")
	ax0 = plt.subplot(gs[0], sharex=ax2)
	ax0.set_title("Volatility Measure")
	ax1 = plt.subplot(gs[1], sharex=ax2)
	ax1.set_title("px, ma30, ma120")
	
	ax0.plot(t, dp, color = 'k')
	ax0.fill_between(t, rs30, -rs30, color='g', alpha=0.75)
	ax0.fill_between(t, rs2, -rs2, color='y', alpha=0.6)
	
	#traded price interval
	ax1.plot(t, df['price'], color = 'k')
	ax1.fill_between(t, df['max_p'], df['min_p'], color='b', alpha=0.9)
	
	ax1.plot(t, ap, color = 'r')
	ax1.plot(t, ap + z * sd, ls = 'dashdot', color = 'r')
	ax1.plot(t, ap - z * sd, ls = 'dashdot', color = 'r')
	
	ax1.plot(t, rm30, color = 'g')
	ax1.plot(t, rm30 + z * rs30, ls = 'dotted', color = 'g')
	ax1.plot(t, rm30 - z * rs30, ls = 'dotted', color = 'g')
	ax1.fill_between(t, rm30 + z * rs30, rm30 - z * rs30, color='g', alpha=0.75)
	
	ax1.plot(t, rm2, color = 'y')
	ax1.plot(t, rm2 + z * rs2, ls = 'dotted', color = 'y')
	ax1.plot(t, rm2 - z * rs2, ls = 'dotted', color = 'y')
	ax1.fill_between(t, rm2 + z * rs2, rm2 - z * rs2, color='y', alpha=0.6)
	
	hfmt = md.DateFormatter('%H:%M:%S', tz=pytz.timezone('Asia/Taipei'))
	ax2.xaxis.set_major_formatter(hfmt)
	ax2.set_xlabel(Flag + ' BTC/CNH Trade History')
	
	ax2.fill_between(t, 0, df['amount'], color='darkgoldenrod', alpha=1)
	fig.autofmt_xdate()
	
	fig.savefig(Flag + ".png")
	plt.show()
	
	
def demo_ma_strategy(df, order_quantity, rolling, mDF):

	z = 1.96	
	[rm, rs] = get_rolling(df, rolling)
	elapsedT = []
	elapsedS = []
	t = []
	
	for i in df.index:
		
		order_time = df.iloc[i]['date']
		t.append(md.date2num(order_time))
		if i >= rolling:
		
			px = rm[i] + z * rs[i]
			elapsedS.append(elapsed_time_for_order(df, px, order_quantity, order_time))
			elapsedT.append(elapsed_time_4_order(mDF, px, order_quantity, order_time))
	
	#show key stats:
	
	print " .........."
	print " Total orders:", len(elapsedT)
	print " Unfilled orders:", sum(np.isnan(elapsedT))
	print " .........."
	print " Elapsed time to filled order:"
	print " - Min T:", min(elapsedT)
	print " - Max T:", max(elapsedT)
	print " - Mean T:", np.nanmean(elapsedT), "sec"
	print " - Std T:", np.nanstd(elapsedT), "sec"
	
	#plot results
	fig = plt.figure(figsize=(12, 9))
	gs = gridspec.GridSpec(3, 1, height_ratios=[2, 2, 1])
	plt.rc('axes', grid=True)
	
	ax1 = plt.subplot(gs[1])
	ax1.set_title("Price History")
	ax0 = plt.subplot(gs[0], sharex=ax1)
	ax0.set_title("Ordered-to-filled Time")
	ax2 = plt.subplot(gs[2])
	ax2.set_title("Elapsed Time Histogram")
	
	ax0.plot(t, df['price'], color = 'k')
	ax0.fill_between(t, df['max_p'], df['min_p'], color='b', alpha=0.9)
	ax0.plot(t, rm, color = 'g')
	ax0.plot(t, rm + z * rs, ls = 'dotted', color = 'g')
	ax0.plot(t, rm - z * rs, ls = 'dotted', color = 'g')
	ax0.fill_between(t, rm + z * rs, rm - z * rs, color='g', alpha=0.75)
	
	ax1.scatter(t[rolling:], elapsedT, s=10, color = 'r', marker='+', alpha=0.5)
	ax1.plot(t[rolling:], elapsedS, color = 'g')
	ax1.plot(t[rolling:], elapsedT, color = 'b')
	
	hfmt = md.DateFormatter('%H:%M:%S', tz=pytz.timezone('Asia/Taipei'))
	ax1.xaxis.set_major_formatter(hfmt)
	
	elapsedT = np.asarray(elapsedT)
	ax2.hist(elapsedT[~np.isnan(elapsedT)], bins=20)
	
	elapsedT = np.append(np.zeros(rolling), elapsedT)
	df.loc[:,'Elapsed_T'] = pd.Series(elapsedT, index=df.index)
	
	df.loc[:,'Order_px'] = pd.Series(rm + z * rs, index=df.index)
	df.to_csv("MY_STRAT.csv")
	
	plt.show()
	
def demo_alpha_strategy(mDF, DF, bsec, order_quantity):

	dT = timedelta(seconds=1) * bsec
	rolling = 30
	z = 1.96	
	[rm, rs] = get_rolling(DF, rolling)
	
	t = []
	px00 = []
	px05 = []
	#px20 = []
	
	elapsedT1 = []
	elapsedT2 = []
	
	for i in DF.index:
		
		order_T = DF.iloc[i]['date']
		t.append(md.date2num(order_T))
		
		px = pct_px(mDF, order_T - dT, order_T, alpha=0.05)
		px05.append(px)
		
		if i >= rolling:
			elapsedT1.append(elapsed_time_4_order(mDF, rm[i] + z * rs[i], order_quantity, order_T))
		elapsedT2.append(elapsed_time_4_order(mDF, px, order_quantity, order_T))
	
	#plot results
	fig = plt.figure(figsize=(12, 9))
	gs = gridspec.GridSpec(3, 1, height_ratios=[2, 2, 1])
	plt.rc('axes', grid=True)
	
	ax2 = plt.subplot(gs[2])
	ax2.set_title("Volume")
	
	ax1 = plt.subplot(gs[1], sharex=ax2)
	ax1.set_title("Strategy Premiums")
	
	ax0 = plt.subplot(gs[0], sharex=ax2)
	ax0.set_title("Px")
	
	
	ax0.plot(t, DF['price'], color = 'k')
	ax0.fill_between(t, DF['max_p'], DF['min_p'], color='b', alpha=0.9)
	ax0.plot(t, rm, color = 'g')
	ax0.plot(t, rm + z * rs, ls = 'dotted', color = 'g')  # green - sd - t1
	ax0.plot(t, rm - z * rs, ls = 'dotted', color = 'g')
	ax0.fill_between(t, rm + z * rs, rm - z * rs, color='g', alpha=0.75)
	
	#ax0.plot(t, px00, ls = 'dotted', color = 'r')
	ax0.plot(t, px05, color = 'r') # red - local - t2
	#ax0.plot(t, px20, ls = 'dotted', color = 'r')

	
	print "- 1.96 * roSD:", np.mean(rm + z * rs - DF['price'])
	#print "- Max px     :", np.mean(px00 - DF['price'])
	print "- Top 5% px  :", np.mean(px05 - DF['price'])
	

	ax1.plot(t[rolling:], elapsedT1, color = 'g') # green - sd - t1
	ax1.plot(t, elapsedT2, color = 'r') # red - local - t2

	#ax1.scatter(t[rolling:], elapsedT, s=10, color = 'r', marker='+', alpha=0.5)	
	#ax1.scatter(t, px00 - DF['price'], s=10, color = 'r', marker='+', alpha=0.5)
	#ax1.scatter(t, px05 - DF['price'], s=10, color = 'b', marker='+', alpha=0.5)
	#ax1.scatter(t, px20 - DF['price'], s=10, color = 'g', marker='+', alpha=0.5)
	
	hfmt = md.DateFormatter('%H:%M:%S', tz=pytz.timezone('Asia/Taipei'))
	ax2.xaxis.set_major_formatter(hfmt)
	
	ax2.fill_between(t, 0, DF['amount'], color='darkgoldenrod', alpha=1)
	fig.autofmt_xdate()
	
	#elapsedT = np.asarray(elapsedT)
	#ax2.hist(elapsedT[~np.isnan(elapsedT)], bins=20)
	
	#elapsedT = np.append(np.zeros(rolling), elapsedT)
	#df.loc[:,'Elapsed_T'] = pd.Series(elapsedT, index=df.index)
	
	#df.loc[:,'Order_px'] = pd.Series(rm + z * rs, index=df.index)
	#df.to_csv("MY_STRAT.csv")
	
	plt.show()

#function for plotting scatters
def mDF_display_rolling_scatter(mDF, n=10):

	t0 = mDF['date'][0]
	N = int((mDF['date'][len(mDF.index) - 1] - t0).seconds) #N: tN-t0 in sec
	P = []
	Q = []
	T = []
	L = []
	
	#initial: 0 to n
	for i in range(0, n):
		subDF = mDF.loc[mDF['date'] == t0 + timedelta(seconds=i)]
		PQ = subDF.groupby('price').sum()
		L.append(len(PQ['amount']))
		
		for q in PQ['amount']:
			Q.append(q)
			P.append(PQ.loc[PQ['amount'] == q].index[0])
			T.append(t0 + timedelta(seconds=i))
	
	print L, ":", T[0]
	orders_scatter(P, Q, T)
	
	#remove first and append last
	for i in range(n + 1, N + 1): #range(5,10) -> 5 6 7 8 9
		Q = Q[L[0]:]
		P = P[L[0]:]
		T = T[L[0]:]	
		L = L[1:]
	
		Ti = T[len(T) - 1] + timedelta(seconds=1)
		subDF = mDF.loc[mDF['date'] == Ti]
		PQ = subDF.groupby('price').sum()
		L.append(len(PQ['amount']))
		
		for q in PQ['amount']:
			Q.append(q)
			P.append(PQ.loc[PQ['amount'] == q].index[0])
			T.append(Ti)
		
		print L, ":", T[0]
		orders_scatter(P, Q, T)

	#print t0, mDF['date'][len(mDF.index) - 1], N 

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
	
	fig = plt.figure(figsize=(12, 16))
	plt.rc('axes', grid=True)
	#gs = gridspec.GridSpec(1, 2, width_ratios=[10, 1])
	
	#ax2 = plt.subplot(gs[1])
	
	gs = gridspec.GridSpec(1, 1)
	ax1 = plt.subplot(gs[0])
	
	ax1.set_xlabel("time (s)")
	ax1.set_ylabel("price")
	
	t = []
	for tm in T:
		t.append(int((tm - T0).seconds))
	
	#option 1: random color, Q in size
	#ax1.scatter(t, P, s= [200 * q ** 0.5 for q in Q], c=colors, alpha=0.5, edgecolors='face')
	
	#option 2: same size, Q in color
	cm = plt.cm.get_cmap('Blues')
	#cm = plt.cm.get_cmap('winter')
	sc = ax1.scatter(t, P, c=log(Q), s=120, cmap=cm, alpha=0.9, edgecolors='face')
	plt.colorbar(sc)
	
	#for i in range(0, len(t)):
	#	plt.scatter(t[i], P[i], s= 100, c='r', alpha= (Q[i] - min(Q)) / (max(Q) - min(Q)), edgecolors='face')
	
	#option 3: ordered color, Q in size
	#ax1.scatter(t, P, s= [200 * q ** 0.5 for q in Q], c=t, alpha=0.5, edgecolors='face')
	
	
	#common setting
	ax1.set_xlim(min(t) - 1, max(t) + 1)
	
	'''
	ax2.scatter(1, np.mean(P), s= (200 * 1 ** 0.5), c='k', alpha=1, edgecolors='face')
	ax2.scatter(1, np.mean(P) + np.std(P), s= (200 * 0.01 ** 0.5), c='k', alpha=1, edgecolors='face')
	ax2.scatter(1, np.mean(P) + np.std(P) * 0.5, s= (200 * 0.1 ** 0.5), c='k', alpha=1, edgecolors='face')
	ax2.scatter(1, np.mean(P) - np.std(P) * 0.5, s= (200 * 10 ** 0.5), c='k', alpha=1, edgecolors='face')
	ax2.scatter(1, np.mean(P) - np.std(P), s= (200 * 100 ** 0.5), c='k', alpha=1, edgecolors='face')
	ax2.set_xlim(0, 2)
	'''
	
	plt.show()