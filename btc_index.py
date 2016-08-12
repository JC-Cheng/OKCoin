'''
$662.53	16.67%	OKCoin com
$666.39	16.67%	OKCoin cn
$664.23	16.67%	BTC China
$665.57	16.67%	Bitstamp
$666.56	16.67%	HuoBi
$664.49	16.67%	BitFinex
'''

import requests
import pandas as pd

OKCoin_CN_mkt = None
OKCoin_COM_mkt = None
HuoBi_CN_mkt = None
HuoBi_COM_mkt = None

def get_USDCNH():
	return requests.get('https://www.okcoin.com/api/v1/exchange_rate.do').json()['rate']
fx = get_USDCNH()

def get_Findex():
	return requests.get('https://www.okcoin.com/api/v1/future_index.do?symbol=btc_usd').json()['future_index']

def get_HuoBiCN():
	return get_HuoBi_CN_mkt()['ticker']['last']

def get_HuoBiCOM():
	return get_HuoBi_COM_mkt()['ticker']['last']

def get_OKcoinCN():
	return get_OKcoin_CN_mkt()['ticker']['last']
	
def get_OKcoinCOM():
	return get_OKcoin_COM_mkt()['ticker']['last']

def get_OKcoin_COM_mkt(Update=False):
	
	global OKCoin_COM_mkt
	if OKCoin_COM_mkt is None or Update == True:
		OKCoin_COM_mkt = pd.read_json('https://www.okcoin.com/api/v1/ticker.do?symbol=btc_usd')
	
	return OKCoin_COM_mkt
	
def get_OKcoin_CN_mkt(Update=False):
	
	global OKCoin_CN_mkt
	if OKCoin_CN_mkt is None or Update == True:
		OKCoin_CN_mkt = pd.read_json('https://www.okcoin.cn/api/v1/ticker.do?symbol=btc_cny')
	
	return OKCoin_CN_mkt

def get_HuoBi_COM_mkt(Update=False):

	global HuoBi_COM_mkt
	if HuoBi_COM_mkt is None or Update == True:
		HuoBi_COM_mkt = pd.read_json('http://api.huobi.com/usdmarket/ticker_btc_json.js')
	
	return HuoBi_COM_mkt

def get_HuoBi_CN_mkt(Update=False):

	global HuoBi_CN_mkt
	if HuoBi_CN_mkt is None or Update == True:
		HuoBi_CN_mkt = pd.read_json('http://api.huobi.com/staticmarket/ticker_btc_json.js')
	
	return HuoBi_CN_mkt

def arbi(crny='USD'):
	
	if crny == 'USD':
		exA = get_OKcoin_COM_mkt()
		exB = get_HuoBi_COM_mkt()
	elif crny == 'CNY':
		exA = get_OKcoin_CN_mkt()
		exB = get_HuoBi_CN_mkt()
	
	if exA['ticker']['buy'] > exB['ticker']['sell']:
		print crny, exA['ticker']['buy'] - exB['ticker']['sell'], "arbitrage opportunity presented!"
		print ' - ', 50 * (exA['ticker']['buy'] - exB['ticker']['sell'])/(exA['ticker']['buy'] + exB['ticker']['sell']), "%"
	elif exB['ticker']['buy'] > exA['ticker']['sell']:
		print crny, exB['ticker']['buy'] - exA['ticker']['sell'], "arbitrage opportunity presented!"
		print ' - ', 50 * (exB['ticker']['buy'] - exA['ticker']['sell'])/(exA['ticker']['buy'] + exB['ticker']['sell']), "%"


#taker FEE
#ETH/LSK/EXP/VOX/SC/BTM/DAO		

print "INDEX      :", get_Findex(), "\n"

#print 'This {food} is {adjective}.'.format(food='spam', adjective='absolutely horrible')
#print '{0:2d} {1:3d} {2:4d}'.format(x, x*x, x*x*x)

print '{exchg}-{dom}: {last} | {bid} | {ask}'.format(exchg='OKCoin', dom='CN', last=get_OKcoinCN(), bid=100, ask=200)
print "OKCoin CN  :", get_OKcoinCN()
print "HuoBi CN   :", get_HuoBiCN()

print "OKCoin COM :", get_OKcoinCOM()
print "HuoBi COM  :", get_HuoBiCOM()

arbi('USD')
arbi('CNY')

print "FX:", fx

#print get_OKcoin_COM_mkt()
#print get_HuoBi_COM_mkt()