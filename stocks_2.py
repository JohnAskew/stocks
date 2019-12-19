#! python3
import os

#------------------------------------#
# This section will install python 
# modules needed to run this script
#------------------------------------#
try:
    import csv
except:
    os.system("pip3 install csv")
    import csv

import datetime as dt
from datetime import timedelta
try:
    import requests
except:
    os.system('pip install requests')
    import requests
try:
    import matplotlib as mpl
except:
    os.system('pip3 install matplotlib')
    import matplotlib as mpl
try:
    import matplotlib.pyplot as plt
except:
    os.system("pip3 install matplotlib")
    import matplotlib.pyplot as plt

from  matplotlib import style
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
try:
    import mpl_finance
except:
    os.system('pip3 install mpl_finance')
    import mpl_finance
from  mpl_finance import candlestick_ohlc
try:
    import pandas as pd
except:
    os.system('pip3 install pandas')
    import pandas as pd
try:
    import numpy as np
except:
    os.system("pip3 install numpy")
    import numpy as np
try:
    import pandas_datareader.data as web
except:
    os.system("pip3 install pandas-datareader")
    import pandas_datareader.data as web
try:
    from pylab import *
except:
    os.system('pip install pylab')
    from pylab import *
try:
    import re
except:
    os.system("pip3 install re")
    import re
try:
    import getpass
except:
    os.system('pip install getpass')
    import getpass
import sys
import time

style.use('fivethirtyeight')

plt.rcParams['axes.formatter.useoffset'] = False
########################################################
# Functions (before Main Logic)
########################################################
#-------------------------------------------------------#
def calc_rsi(prices, n=14):
#-------------------------------------------------------#
    deltas = np.diff(prices)
    seed = deltas[:n + 1]
    up = seed[seed >= 0].sum()/n
    down = -seed[seed < 0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1. + rs)

    for i in range(n, len(prices)):
        delta = deltas[i - 1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = .0
            downval = -delta
        up   = (up   * (n - 1) + upval)   / n
        down = (down * (n - 1) + downval) / n

        rs = up/down
        rsi[i] = 100. - 100./(1. + rs)

    return rsi
#-------------------------------------------------------#
def moving_average(values, window):
#-------------------------------------------------------#
    weights  = np.repeat(1.0, window) / window   #Numpy repeat - repeats items in array - "window" times
    smas = np.convolve(values, weights, 'valid') #Numpy convolve - returns the discrete, linear convolution of 2 seq.
    #https://stackoverflow.com/questions/20036663/understanding-numpys-convolve
    return smas
#-------------------------------------------------------#
def calc_ema(values,window):
#-------------------------------------------------------#
    weights = np.exp(np.linspace(-1, 0., window))
    weights /= weights.sum()
    a = np.convolve(values, weights, mode = 'full')[:len(values)]
    a[:window] = a[window]
    return a
#-------------------------------------------------------#
def calc_macd(x, slow=26, fast = 12):
#-------------------------------------------------------#
    eMaSlow = calc_ema(x, slow)
    eMaFast = calc_ema(x, fast)
    return eMaSlow, eMaFast, eMaFast - eMaSlow
#-------------------------------------------------------#
def rotate_xaxis(owner):
#-------------------------------------------------------#
    for label in owner.xaxis.get_ticklabels():
        label.set_rotation(45)
        label.set_fontsize(6)
#-------------------------------------------------------#
def set_labels(owner):
#-------------------------------------------------------#
    #owner.set_xlabel('Dates', fontsize=8, fontweight =2, color = 'b')
    owner.set_ylabel('Price', fontsize=8, fontweight =5, color = 'g')
#-------------------------------------------------------#
def hide_frame(owner):
#-------------------------------------------------------#
    owner.grid(False)
    owner.xaxis.set_visible(False)
    owner.yaxis.set_visible(False)
    owner.set_xlabel(False)
#-------------------------------------------------------#
def set_spines(owner):
#-------------------------------------------------------#
    owner.spines['left'].set_color('m')
    owner.spines['left'].set_linewidth(1)
    owner.spines['right'].set_visible(False) #color('m')
    owner.spines['top'].set_color('m')
    owner.spines['top'].set_linewidth(1)
    owner.spines['bottom'].set_visible(False)
#######################################
# M A I N  L O G I C
#######################################

if __name__ == '__main__':
    
    if len(sys.argv) > 2:
        if sys.argv[1]:
            print('len(sys.argv)=', len(sys.argv))
            ax1_subject = sys.argv[1]
        else:
            ax1_subject = 'JCP'
        if len(sys.argv[2]) > 0:
            ax2_year_subject = sys.argv[2]
        else:
            ax2_year_subject = 'EBAY'
    else:
        ax1_subject = 'JCP'
        ax2_year_subject = 'EBAY'

user = getpass.getuser()
movAvg_window_days_short_term = 10                                         #Moving Average 10 days (quick)
movAvg_window_days_long_term = 30                                         #Moving Average 30 days (slow)
macd_periods_long_term = 26
macd_periods_short_term = 12
expMA_periods = 9 
#-----------------------------------#
# Variables
#-----------------------------------#
provider = 'yahoo' 
currPath = os.getcwd()              # Directory you are in NOW
savePath = 'askew'                  # We will be creating this new sub-directory
myPath = (currPath + '/' + savePath)# The full path of the new sub-dir
#-----------------------------------#
# Grab Dates
#-----------------------------------#
start = ( dt.datetime.now() - dt.timedelta(days = 365) )       # Format is year, month, day
end = dt.datetime.today()           # format of today() = [yyyy, mm, dd] - list of integers
#-----------------------------------#
# Set up place to save spreadsheet
#-----------------------------------#
if not os.path.exists(myPath):      # The directory you are in NOW
    os.makedirs(myPath)             # create a new dir below the dir your are in NOW
os.chdir(myPath)                    # move into the newly created sub-dir

for subject in ax1_subject, ax2_year_subject:
    saveFile=(subject + '.csv')    # The RESUlTS we are saving on a daily basis
    if os.path.exists(saveFile):
        st = os.stat(saveFile)
        if dt.date.fromtimestamp(st.st_mtime) != dt.date.today():
            df = web.DataReader(subject, provider, start, end)
            df.rename(columns={"Adj Close":'Adj_Close'}, inplace=True)
            df.to_csv(saveFile)
    else:
        df = web.DataReader(subject, provider, start, end)
        df.rename(columns={"Adj Close":'Adj_Close'}, inplace=True)
        df['Stock'] = subject
        header = ["High", "Low", "Open", "Close", "Volume", 'Adj_Close']
        df.to_csv(saveFile, columns = header)
             #Lose the date index so we can address it as a column

    
                 # Company providing the raw data we are after


########################################################
## Let's define our canvas, before we go after the data
## Odd numbers (ex. ax1_vol) are for stock 1. Even = stock 2.
#########################################################
plot_row = 18 + 84
plot_col = 10
fig, axs = plt.subplots(figsize=(14,8), facecolor='#FFFFFA', sharex = True, sharey = True) #Too Bad, I really liked this color, facecolor = '#FFFFFA')
plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='lower'))

ax1_year = plt.subplot2grid((plot_row,plot_col), (0,0),  rowspan = 7, colspan = 4)
ax1_ohlc = plt.subplot2grid((plot_row,plot_col), (17,0), rowspan = 7, colspan = 4, sharex = ax1_year, sharey = ax1_year)
ax1_ma   = plt.subplot2grid((plot_row,plot_col), (32,0), rowspan = 7, colspan = 4, sharex = ax1_year, sharey = ax1_year)
ax1_rsi  = plt.subplot2grid((plot_row,plot_col), (47,0), rowspan = 7, colspan = 4, sharex = ax1_year)
ax1_macd = plt.subplot2grid((plot_row,plot_col), (62,0), rowspan = 7, colspan = 4, sharex = ax1_year)
ax1_vol  = plt.subplot2grid((plot_row,plot_col), (77,0), rowspan = 4, colspan = 4, sharex = ax1_year)
ax1_tot  = plt.subplot2grid((plot_row,plot_col), (94,0), rowspan = 4, colspan = 2)

ax2_year = plt.subplot2grid((plot_row,plot_col), (0,6),  rowspan = 7, colspan = 4)
ax2_ohlc = plt.subplot2grid((plot_row,plot_col), (17,6), rowspan = 7, colspan = 4, sharex = ax2_year, sharey = ax2_year)
ax2_ma   = plt.subplot2grid((plot_row,plot_col), (32,6), rowspan = 7, colspan = 4, sharex = ax2_year, sharey = ax2_year)
ax2_rsi  = plt.subplot2grid((plot_row,plot_col), (47,6), rowspan = 7, colspan = 4, sharex = ax2_year)
ax2_macd = plt.subplot2grid((plot_row,plot_col), (62,6), rowspan = 7, colspan = 4, sharex = ax2_year)
ax2_vol  = plt.subplot2grid((plot_row,plot_col), (77,6), rowspan = 4, colspan = 4, sharex = ax2_year)
ax2_tot  = plt.subplot2grid((plot_row,plot_col), (94,6), rowspan = 4, colspan = 2)
########################################################
#      ####  #####    ###     ###  #   #      # 
#     #        #     #   #   #     #  #      ##
#       #      #     #   #   #     # #      # #
#         #    #     #   #   #     #  #       #
#         #    #     #   #   #     #   #      #
#      ###     #      ###     ###  #    #   #####
########################################################
# Populate Data
########################################################
df = pd.read_csv((ax1_subject + '.csv'), parse_dates=True, index_col =0)

df_ohlc = df['Adj_Close'].resample('10D').ohlc()

df.reset_index(inplace = True)      
########################################################
#Define DATA and attributes
########################################################
stock_entry = (df['Adj_Close'][0])               # Set marker of last years close.
movAvg_window_days_short_term = 10                                         #Moving Average 10 days (quick)
movAvg_window_days_long_term = 30                                         #Moving Average 30 days (slow)
macd_periods_long_term = 26
macd_periods_short_term = 12
expMA_periods = 9 
df.reset_index(inplace = True)                   #Lose the date index so we can address it as a column
ma1 = moving_average(df['Adj_Close'], movAvg_window_days_short_term)
ma2 = moving_average(df['Adj_Close'], movAvg_window_days_long_term)
start = len(df['Date'][movAvg_window_days_long_term - 1:])
########################################################
#Start Plotting
########################################################
ax1_year.plot_date(df['Date'], df['Adj_Close'], '-', label='ADJ Closing Price', color = 'blue', linewidth = 1)
ax1_year.plot([],[], linewidth = 2, label = 'Adj_Close yr ago' , color = 'k', alpha = 0.9)
ax1_year.axhline(df['Adj_Close'][0], color = 'k', linewidth = 2)
ax1_year.fill_between(df['Date'], df['Adj_Close'], stock_entry, where = (df['Adj_Close'] > stock_entry), facecolor='g', alpha=0.6)
ax1_year.fill_between(df['Date'], df['Adj_Close'], stock_entry, where = (df['Adj_Close'] < stock_entry), facecolor='r', alpha=0.6)
rotate_xaxis(ax1_year)
ax1_year.grid(True, color='lightgreen', linestyle = '-', linewidth=2)
set_spines(ax1_year)
ax1_year.tick_params(axis = 'x', colors = '#890b86')
ax1_year.tick_params(axis = 'y', colors = 'g', labelsize = 6)
ax1_year.set_title(ax1_subject, color = '#353335', size = 10)
set_labels(ax1_year)
ax1_year.set_color = '#890b86'
ax1_year.legend(bbox_to_anchor=(1.01, 1),fontsize = 6, fancybox = True, loc = 0, markerscale = -0.5, framealpha  = 0.5, facecolor = '#f9ffb7')

df_ohlc.reset_index(inplace=True)                #Date becomes addressable column

df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num) #Date is now in ohlc format
candlestick_ohlc(ax1_ohlc, df_ohlc.values, width = 1, colorup = 'g')
rotate_xaxis(ax1_ohlc)
set_labels(ax1_ohlc)
set_spines(ax1_ohlc)
ax1_ohlc.tick_params(axis = 'x', colors = '#890b86')
ax1_ohlc.tick_params(axis = 'y', colors = 'g', labelsize = 6)
ax1_ohlc.set_ylabel('OHLC', fontsize=8, fontweight =5, color = 'darkgreen')
ax1_ohlc.plot([],[], linewidth = 2, label = 'Up' , color = 'green', alpha = 0.9)
ax1_ohlc.plot([],[], linewidth = 2, label = 'Down' , color = 'red', alpha = 0.9)
ax1_ohlc.legend(bbox_to_anchor=(1.01, 1),fontsize = 6, fancybox = True, loc = 0, markerscale = -0.5, framealpha  = 0.5, facecolor = '#dde29a')


rotate_xaxis(ax1_ma)
ax1_ma.fill_between(df['Date'][- start:], ma1[-start:], ma2[-start:], where = (ma1[-start:] > ma2[-start:]), facecolor='g', alpha=0.6)
ax1_ma.fill_between(df['Date'][- start:], ma1[-start:], ma2[-start:], where = (ma1[-start:] < ma2[-start:]), facecolor='red', alpha=0.6)
ax1_ma.grid(True, color='lightgreen', linestyle = '-', linewidth=2)
ax1_ma.plot(df['Date'][- start:], ma1[- start:], color = 'b', linewidth = 1)     #Have to skip date ahead 10days (movAvg_window_days_short_term)
ax1_ma.plot(df['Date'][- start:], ma2[- start:], color = 'k', linewidth = 1 )      #Have to skip date ahead 30 days (movAvg_window_days_long_term)
set_spines(ax1_ma)
ax1_ma.tick_params(axis = 'x', colors = '#890b86')
ax1_ma.tick_params(axis = 'y', colors = 'g', labelsize = 6)
ax1_ma.plot([],[], linewidth = 2, label = '10d Mov. Avg.' , color = 'b', alpha = 0.9)
ax1_ma.plot([],[], linewidth = 2, label = '30d Mov. Avg.' , color = 'k', alpha = 0.9)
ax1_ma.legend(bbox_to_anchor=(1.01, 1),fontsize = 6, fancybox = True, loc = 0, markerscale = -0.5, framealpha  = 0.5, facecolor = '#dde29a')
#set_labels(ax1_ma)
ax1_ma.set_ylabel('Moving Average', fontsize=8, fontweight =5, color = 'r')
##
### ax1_rsi
##
rsi = calc_rsi(df["Close"])
rotate_xaxis(ax1_rsi)
set_spines(ax1_rsi)
ax1_rsi.tick_params(axis = 'y', colors = 'g', labelsize = 6)
ax1_rsi.set_ylabel('RSI', fontsize=8, fontweight =5, color = 'darkorange')
rsi_col_over= 'red'
rsi_col_under = 'lightgreen'
ax1_rsi.plot(df['Date'],rsi, linewidth =1, color = 'orange')
ax1_rsi.axhline(30, color=rsi_col_under, linewidth = 1)
ax1_rsi.axhline(70, color=rsi_col_over, linewidth = 1)
ax1_rsi.set_yticks([30,70])
ax1_rsi.fill_between(df['Date'], rsi, 70, where = (rsi > 70), facecolor='r', alpha=0.6)
ax1_rsi.fill_between(df['Date'], rsi, 30, where = (rsi < 30), facecolor='darkgreen', alpha=0.6)



plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='lower'))
ax1_rsi.plot([],[], linewidth = 2, label = 'rsi' , color = 'orange', alpha = 0.9)
ax1_rsi.plot([],[], linewidth = 2, label = 'OverVal' , color = 'red', alpha = 0.9)
ax1_rsi.plot([],[], linewidth = 2, label = 'UnderVal' , color = 'darkgreen', alpha = 0.9)
ax1_rsi.legend(bbox_to_anchor=(1.01, 1),fontsize = 6, fancybox = True, loc = 2, markerscale = -0.5, framealpha  = 0.5, facecolor = '#dde29a')

eMaSlow, eMaFast, macd = calc_macd(df['Close'])
ema9 = calc_ema(macd, expMA_periods)
macd_col_over = 'red'
macd_col_under = 'lightgreen'
rotate_xaxis(ax1_macd)
set_spines(ax1_macd)
ax1_macd.plot(df['Date'], macd, linewidth =2, color = 'darkred')
ax1_macd.plot(df['Date'], ema9, linewidth =1, color = 'blue')
ax1_macd.fill_between(df['Date'], macd - ema9, 0, alpha = 0.5, facecolor = 'darkgreen', where = (macd - ema9 > 0))
ax1_macd.fill_between(df['Date'], macd - ema9, 0, alpha = 0.5, facecolor = macd_col_over, where = (macd - ema9 < 0))
plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
ax1_macd.tick_params(axis = 'x', colors = '#890b86')
ax1_macd.tick_params(axis = 'y', colors = 'g', labelsize = 6)
ax1_macd.set_ylabel('MACD', fontsize=8, fontweight =5, color = 'darkred')

ax1_macd.plot([], label='macd ' + str(macd_periods_short_term)  + ',' + str(macd_periods_long_term) + ',' + str(expMA_periods), linewidth = 2, color = 'darkred')
ax1_macd.plot([], label='ema ' + str(expMA_periods),  linewidth = 2, color = 'blue')
ax1_macd.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0., fontsize = 6.0)



### ax1_vol
##
ax1_vol.tick_params(axis = 'x', colors = '#890b86')
ax1_vol.plot_date(df['Date'], df['Volume'], '-', label='Volume', color = 'blue', linewidth = 1)
ax1_vol.tick_params(axis = 'y', colors = 'k', labelsize = 6)
rotate_xaxis(ax1_vol)
set_spines(ax1_vol)
ax1_vol.set_ylim(df['Volume'].min(),df['Volume'].max())
ax1_vol.set_ylabel('Volume', fontsize=8, fontweight =5, color = 'b')
ax1_vol.fill_between(df['Date'],df['Volume'], facecolor='#00ffe8', alpha=.5)


last_rec = (len(df) -1)
#ax1_tot.text(0,-2.1,str(df.iloc[last_rec]), fontsize=9, fontweight = 20)\
last_open = df['Open'].iloc[-1]
last_high = df['High'].iloc[-1]
last_low  = df['Low'].iloc[-1]
last_vol  = df['Volume'].iloc[-1]
last_close = df['Close'].iloc[-1]
hide_frame(ax1_tot)
ax1_tot.text(1,1,'Open:' + '{:20,.2f}'.format(last_open) + '     ', verticalalignment='bottom', horizontalalignment='left',
         color='darkblue', fontsize=8)

ax1_tot.text(1,1,'Close:' + '{:20,.2f}'.format(last_close), verticalalignment='top', horizontalalignment='left',
         color='darkblue', fontsize=8)
ax1_tot.text(1,1,'High:' + '{:20,.2f}'.format(last_high) + '     ', verticalalignment='bottom', horizontalalignment='right',
         color='darkblue', fontsize=8)
ax1_tot.text(1,1,'Low:' + '{:20,.2f}'.format(last_low)+ '     ', verticalalignment='top', horizontalalignment='right',
         color='darkblue', fontsize=8)
ax1_tot.text(0.5,0.25, "Diff:" + str('{:5,.2f}'.format(last_high - last_low)), verticalalignment='bottom', horizontalalignment='left',
         color='darkblue', fontsize=8)
ax1_tot.text(0.5,0.25,"                                    Diff:" + str('{:5,.2f}'.format(last_close - last_open)), verticalalignment='bottom', horizontalalignment='left',
         color='darkblue', fontsize=8)

ax1_macd.axvline(x = df['Date'][int(len(df['Date'])/2)], linewidth = 1,  color = 'yellow')
########################################################
#      ####  #####    ###     ###  #   #      ## 
#     #        #     #   #   #     #  #     #    #
#       #      #     #   #   #     # #          # 
#         #    #     #   #   #     #  #       #
#         #    #     #   #   #     #   #    #
#      ###     #      ###     ###  #    #   #####
########################################################
# Populate Data
########################################################
df = pd.read_csv((ax2_year_subject + '.csv'), parse_dates=True, index_col =0)
df.reset_index(inplace = True) 
########################################################
#Define DATA and attributes
########################################################
stock_entry = (df['Adj_Close'][0])               # Set marker of last years close.
movAvg_window_days_short_term = 10                                         #Moving Average 10 days (quick)
movAvg_window_days_long_term = 30                                         #Moving Average 30 days (slow)
ma1 = moving_average(df['Adj_Close'], movAvg_window_days_short_term)
ma2 = moving_average(df['Adj_Close'], movAvg_window_days_long_term)
start = len(df['Date'][movAvg_window_days_long_term - 1:])

ax2_year.plot_date(df['Date'], df['Adj_Close'], '-', label='ADJ Closing Price', color = 'blue', linewidth = 1)
ax2_year.plot([],[], linewidth = 2, label = 'Adj_Close yr ago' , color = 'k', alpha = 0.9)
ax2_year.axhline(df['Adj_Close'][0], color = 'k', linewidth = 2)
ax2_year.fill_between(df['Date'], df['Adj_Close'], stock_entry, where = (df['Adj_Close'] > stock_entry), facecolor='g', alpha=0.6)
ax2_year.fill_between(df['Date'], df['Adj_Close'], stock_entry, where = (df['Adj_Close'] < stock_entry), facecolor='r', alpha=0.6)
rotate_xaxis(ax2_year)
ax2_year.grid(True, color='lightgreen', linestyle = '-', linewidth=2)
set_spines(ax2_year)
ax2_year.tick_params(axis = 'x', colors = '#890b86')
ax2_year.tick_params(axis = 'y', colors = 'g', labelsize = 6)
ax2_year.set_title(ax2_year_subject, color = '#353335', size = 10)
set_labels(ax2_year)
ax2_year.set_color = '#890b86'
ax2_year.legend(bbox_to_anchor=(1.01, 1),fontsize = 6, fancybox = True, loc = 0, markerscale = -0.5, framealpha  = 0.5, facecolor = '#f9ffb7')

#df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num) #Date is now in ohlc format
candlestick_ohlc(ax2_ohlc, df_ohlc.values, width = 1, colorup = 'g')
rotate_xaxis(ax2_ohlc)
set_labels(ax2_ohlc)
set_spines(ax2_ohlc)
ax2_ohlc.tick_params(axis = 'x', colors = '#890b86')
ax2_ohlc.tick_params(axis = 'y', colors = 'g', labelsize = 6)
ax2_ohlc.set_ylabel('OHLC', fontsize=8, fontweight =5, color = 'darkgreen')
ax2_ohlc.plot([],[], linewidth = 2, label = 'Up' , color = 'green', alpha = 0.9)
ax2_ohlc.plot([],[], linewidth = 2, label = 'Down' , color = 'red', alpha = 0.9)
ax2_ohlc.legend(bbox_to_anchor=(1.15, 1),fontsize = 6, fancybox = True, loc = 0, markerscale = -0.5, framealpha  = 0.5, facecolor = '#dde29a')

rotate_xaxis(ax2_ma)
ax2_ma.fill_between(df['Date'][- start:], ma1[-start:], ma2[-start:], where = (ma1[-start:] > ma2[-start:]), facecolor='g', alpha=0.6)
ax2_ma.fill_between(df['Date'][- start:], ma1[-start:], ma2[-start:], where = (ma1[-start:] < ma2[-start:]), facecolor='red', alpha=0.6)
ax2_ma.grid(True, color='lightgreen', linestyle = '-', linewidth=2)
ax2_ma.plot(df['Date'][- start:], ma1[- start:], color = 'b', linewidth = 1)     #Have to skip date ahead 10days (movAvg_window_days_short_term)
ax2_ma.plot(df['Date'][- start:], ma2[- start:], color = 'k', linewidth = 1 )      #Have to skip date ahead 30 days (movAvg_window_days_long_term)
set_spines(ax2_ma)
ax2_ma.tick_params(axis = 'x', colors = '#890b86')
ax2_ma.tick_params(axis = 'y', colors = 'g', labelsize = 6)
ax2_ma.plot([],[], linewidth = 2, label = '10d Mov. Avg.' , color = 'b', alpha = 0.9)
ax2_ma.plot([],[], linewidth = 2, label = '30d Mov. Avg.' , color = 'k', alpha = 0.9)
ax2_ma.legend(bbox_to_anchor=(1.01, 1),fontsize = 6, fancybox = True, loc = 0, markerscale = -0.5, framealpha  = 0.5, facecolor = '#dde29a')
#set_labels(ax2_ma)
ax2_ma.set_ylabel('Moving Average', fontsize=8, fontweight =5, color = 'r')

##
### ax2_rsi
##
rsi = calc_rsi(df["Close"])
rotate_xaxis(ax2_rsi)
set_spines(ax2_rsi)
ax2_rsi.tick_params(axis = 'y', colors = 'g', labelsize = 6)
ax2_rsi.set_ylabel('RSI', fontsize=8, fontweight =5, color = 'darkorange')
rsi_col_over= 'red'
rsi_col_under = 'lightgreen'
ax2_rsi.plot(df['Date'],rsi, linewidth =1, color = 'orange')
ax2_rsi.axhline(30, color=rsi_col_under, linewidth = 1)
ax2_rsi.axhline(70, color=rsi_col_over, linewidth = 1)
ax2_rsi.set_yticks([30,70])
ax2_rsi.fill_between(df['Date'], rsi, 70, where = (rsi > 70), facecolor='r', alpha=0.6)
ax2_rsi.fill_between(df['Date'], rsi, 30, where = (rsi < 30), facecolor='darkgreen', alpha=0.6)
ax2_rsi.plot([],[], linewidth = 2, label = 'OverVal' , color = 'red', alpha = 0.9)
ax2_rsi.plot([],[], linewidth = 2, label = 'UnderVal' , color = 'darkgreen', alpha = 0.9)
ax2_rsi.legend(bbox_to_anchor=(1.01, 1),fontsize = 6, fancybox = True, loc = 0, markerscale = -0.5, framealpha  = 0.5, facecolor = '#dde29a')


eMaSlow, eMaFast, macd = calc_macd(df['Close'])
ema9 = calc_ema(macd, expMA_periods)
macd_col_over = 'red'
macd_col_under = 'lightgreen'
rotate_xaxis(ax2_macd)
set_spines(ax2_macd)
ax2_macd.plot(df['Date'], macd, linewidth =2, color = 'darkred')
ax2_macd.plot(df['Date'], ema9, linewidth =1, color = 'blue')
ax2_macd.fill_between(df['Date'], macd - ema9, 0, alpha = 0.5, facecolor = 'darkgreen', where = (macd - ema9 > 0))
ax2_macd.fill_between(df['Date'], macd - ema9, 0, alpha = 0.5, facecolor = macd_col_over, where = (macd - ema9 < 0))
plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
ax2_macd.tick_params(axis = 'x', colors = '#890b86')
ax2_macd.tick_params(axis = 'y', colors = 'g', labelsize = 6)
ax2_macd.set_ylabel('MACD', fontsize=8, fontweight =5, color = 'darkred')

ax2_macd.plot([], label='macd ' + str(macd_periods_short_term)  + ',' + str(macd_periods_long_term) + ',' + str(expMA_periods), linewidth = 2, color = 'darkred')
ax2_macd.plot([], label='ema ' + str(expMA_periods),  linewidth = 2, color = 'blue')
ax2_macd.legend(bbox_to_anchor=(1.01, 1),fontsize = 6, fancybox = True, loc = 0, markerscale = -0.5, framealpha  = 0.5, facecolor = '#dde29a')


ax2_vol.tick_params(axis = 'x', colors = '#890b86')

ax2_vol.plot_date(df['Date'], df['Volume'], '-', label='Volume', color = 'blue', linewidth = 1)
ax2_vol.set_ylim( df['Volume'].min(),df['Volume'].max())
rotate_xaxis(ax2_vol)
set_spines(ax2_vol)
ax2_vol.set_ylabel('Volume', fontsize=8, fontweight =5, color = 'b')
ax2_vol.tick_params(axis = 'y', colors = 'k', labelsize = 6)
ax2_vol.fill_between(df['Date'],df['Volume'], facecolor='#00ffe8', alpha=.5)

last_rec = (len(df) -1)
#ax1_tot.text(0,-2.1,str(df.iloc[last_rec]), fontsize=9, fontweight = 20)\
last_open = df['Open'].iloc[-1]
last_high = df['High'].iloc[-1]
last_low  = df['Low'].iloc[-1]
last_vol  = df['Volume'].iloc[-1]
last_close = df['Close'].iloc[-1]
hide_frame(ax2_tot)
ax2_tot.text(1,1,'Open:' + '{:20,.2f}'.format(last_open) + '     ', verticalalignment='bottom', horizontalalignment='left',
         color='darkblue', fontsize=8)

ax2_tot.text(1,1,'Close:' + '{:20,.2f}'.format(last_close), verticalalignment='top', horizontalalignment='left',
         color='darkblue', fontsize=8)
ax2_tot.text(1,1,'High:' + '{:20,.2f}'.format(last_high) + '     ', verticalalignment='bottom', horizontalalignment='right',
         color='darkblue', fontsize=8)
ax2_tot.text(1,1,'Low:' + '{:20,.2f}'.format(last_low)+ '     ', verticalalignment='top', horizontalalignment='right',
         color='darkblue', fontsize=8)
ax2_tot.text(0.5,0.25, "Diff:" + str('{:5,.2f}'.format(last_high - last_low)), verticalalignment='bottom', horizontalalignment='left',
         color='darkblue', fontsize=8)
ax2_tot.text(0.5,0.25,"                                    Diff:" + str('{:5,.2f}'.format(last_close - last_open)), verticalalignment='bottom', horizontalalignment='left',
         color='darkblue', fontsize=8)


plt.rc('ytick', labelsize=6 )    # fontsize of the tick labels
plt.subplots_adjust(left = 0.10, bottom = 0.16, right = 0.920, top = 0.93, wspace = 0.2, hspace = -.1)
fig = gcf()
my_title = (user, "Stock Page")
fig.suptitle(user + " Stock Page", fontsize=14)
plt.show()
fig.savefig('Figure1.png')
