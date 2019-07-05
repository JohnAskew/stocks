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
import matplotlib.dates as mdates
try:
    import mpl_finance
except:
    os.system('pip3 install mpl_finance')
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
    import re
except:
    os.system("pip3 install re")
    import re
import sys
import time

style.use('fivethirtyeight')

plt.rcParams['axes.formatter.useoffset'] = False


if __name__ == '__main__':
    
    #print('len(sys.argv)=', len(sys.argv))
    #print('sys.argv[0]=', sys.argv[0])
    #print('sys.argv[1]=', sys.argv[1])
    #print('sys.argv[2]=', sys.argv[2])
    if len(sys.argv) > 1:
        if sys.argv[1]:
            ax1_subject = sys.argv[1]
        else:
            ax1_subject = 'GOOG'
        if len(sys.argv[2]) > 0:
            ax4_subject = sys.argv[2]
        else:
            ax4_subject = 'MANH'
    else:
        ax1_subject = 'GOOG'
        ax4_subject = 'MANH'


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

for subject in ax1_subject, ax4_subject:
    saveFile=(subject + '.csv')    # The RESUlTS we are saving on a daily basis
    if os.path.exists(saveFile):
        st = os.stat(saveFile)
        if dt.date.fromtimestamp(st.st_mtime) != dt.date.today():
            df = web.DataReader(subject, provider, start, end)
            df.to_csv(saveFile)
    else:
        df = web.DataReader(subject, provider, start, end)
        df.to_csv(saveFile)
             #Lose the date index so we can address it as a column

    
                 # Company providing the raw data we are after
########################################################
# Functions (before Main Logic)
########################################################
#-------------------------------------------------------#
def moving_average(values, window):
#-------------------------------------------------------#
    weights  = np.repeat(1.0, window) / window   #Numpy repeat - repeats items in array - "window" times
    smas = np.convolve(values, weights, 'valid') #Numpy convolve - returns the discrete, linear convolution of 2 seq.
    #https://stackoverflow.com/questions/20036663/understanding-numpys-convolve
    return smas
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

########################################################
## Let's define our canvas, before we go after the data
## Odd numbers (ex. ax3) are for stock 1. Even = stock 2.
#########################################################
plot_row = 14
plot_col = 10
fig = plt.figure(figsize=(8,6))
ax1 = plt.subplot2grid((plot_row,plot_col), (0,0), rowspan = 2, colspan = 4)
ax2 = plt.subplot2grid((plot_row,plot_col), (4,0), rowspan = 2, colspan = 4, sharex = ax1, sharey = ax1)
ax3 = plt.subplot2grid((plot_row,plot_col), (8,0), rowspan = 2, colspan = 4, sharex = ax1)
ax4 = plt.subplot2grid((plot_row,plot_col), (0,6), rowspan = 2, colspan = 4)
ax5 = plt.subplot2grid((plot_row,plot_col), (4,6), rowspan = 2, colspan = 4, sharex = ax4, sharey = ax4)
ax6 = plt.subplot2grid((plot_row,plot_col), (8,6), rowspan = 2, colspan = 4, sharex = ax4)
ax7 = plt.subplot2grid((plot_row,plot_col), (13,0), rowspan = 2, colspan = 2)
ax8 = plt.subplot2grid((plot_row,plot_col), (13,6), rowspan = 2, colspan = 2)
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
df.reset_index(inplace = True)      
########################################################
#Define DATA and attributes
########################################################
stock_entry = (df['Adj Close'][0])               # Set marker of last years close.
MovAvg_window1 = 10                                         #Moving Average 10 days (quick)
MovAvg_window2 = 30                                         #Moving Average 30 days (slow)
df.reset_index(inplace = True)                   #Lose the date index so we can address it as a column
ma1 = moving_average(df['Adj Close'], MovAvg_window1)
ma2 = moving_average(df['Adj Close'], MovAvg_window2)
start = len(df['Date'][MovAvg_window2 - 1:])
########################################################
#Start Plotting
########################################################
ax1.plot_date(df['Date'], df['Adj Close'], '-', label='ADJ Closing Price', color = 'blue', linewidth = 1)
ax1.plot([],[], linewidth = 2, label = 'ADJ Close yr ago' , color = 'k', alpha = 0.9)
ax1.axhline(df['Adj Close'][0], color = 'k', linewidth = 2)
ax1.fill_between(df['Date'], df['Adj Close'], stock_entry, where = (df['Adj Close'] > stock_entry), facecolor='g', alpha=0.6)
ax1.fill_between(df['Date'], df['Adj Close'], stock_entry, where = (df['Adj Close'] < stock_entry), facecolor='r', alpha=0.6)
rotate_xaxis(ax1)
ax1.grid(True, color='lightgreen', linestyle = '-', linewidth=2)
set_spines(ax1)
ax1.tick_params(axis = 'x', colors = '#890b86')
ax1.tick_params(axis = 'y', colors = 'g', labelsize = 6)
ax1.set_title(ax1_subject, color = '#353335', size = 10)
set_labels(ax1)
ax1.set_color = '#890b86'
ax1.legend(fontsize = 6, fancybox = True, loc = 0, markerscale = -0.5, framealpha  = 0.5, facecolor = '#f9ffb7')

rotate_xaxis(ax2)
ax2.fill_between(df['Date'][- start:], ma1[-start:], ma2[-start:], where = (ma1[-start:] > ma2[-start:]), facecolor='g', alpha=0.6)
ax2.fill_between(df['Date'][- start:], ma1[-start:], ma2[-start:], where = (ma1[-start:] < ma2[-start:]), facecolor='red', alpha=0.6)
ax2.grid(True, color='lightgreen', linestyle = '-', linewidth=2)
ax2.plot(df['Date'][- start:], ma1[- start:], color = 'b', linewidth = 1)     #Have to skip date ahead 10days (MovAvg_window1)
ax2.plot(df['Date'][- start:], ma2[- start:], color = 'k', linewidth = 1 )      #Have to skip date ahead 30 days (MovAvg_window2)
set_spines(ax2)
ax2.tick_params(axis = 'x', colors = '#890b86')
ax2.tick_params(axis = 'y', colors = 'g', labelsize = 6)
ax2.plot([],[], linewidth = 2, label = '10d mov. avg.' , color = 'b', alpha = 0.9)
ax2.plot([],[], linewidth = 2, label = '30d mov. avg.' , color = 'k', alpha = 0.9)
ax2.legend(fontsize = 6, fancybox = True, loc = 0, markerscale = -0.5, framealpha  = 0.5, facecolor = '#dde29a')
#set_labels(ax2)
ax2.set_ylabel('Moving Average', fontsize=8, fontweight =5, color = 'r')

ax3.tick_params(axis = 'x', colors = '#890b86')
ax3.plot_date(df['Date'], df['Volume'], '-', label='Volume', color = 'blue', linewidth = 1)
ax3.tick_params(axis = 'y', colors = 'k', labelsize = 6)
rotate_xaxis(ax3)
set_spines(ax3)
ax3.set_ylim(df['Volume'].min(),df['Volume'].max())
ax3.set_ylabel('Volume', fontsize=8, fontweight =5, color = 'b')


last_rec = (len(df) -1)
#ax7.text(0,-2.1,str(df.iloc[last_rec]), fontsize=9, fontweight = 20)\
last_open = df['Open'].iloc[-1]
last_high = df['High'].iloc[-1]
last_low  = df['Low'].iloc[-1]
last_vol  = df['Volume'].iloc[-1]
last_close = df['Close'].iloc[-1]
hide_frame(ax7)
ax7.text(1,1,'Open:' + '{:20,.2f}'.format(last_open) + '     ', verticalalignment='bottom', horizontalalignment='left',
         color='darkblue', fontsize=8)

ax7.text(1,1,'Close:' + '{:20,.2f}'.format(last_close), verticalalignment='top', horizontalalignment='left',
         color='darkblue', fontsize=8)
ax7.text(1,1,'High:' + '{:20,.2f}'.format(last_high) + '     ', verticalalignment='bottom', horizontalalignment='right',
         color='darkblue', fontsize=8)
ax7.text(1,1,'Low:' + '{:20,.2f}'.format(last_low)+ '     ', verticalalignment='top', horizontalalignment='right',
         color='darkblue', fontsize=8)
ax7.text(0.5,0.25,"               Diff:     " + str('{:5,.2f}'.format(last_high - last_low)), verticalalignment='bottom', horizontalalignment='right',
         color='darkblue', fontsize=8)
ax7.text(0.5,0.25,"                    Diff:     " + str('{:5,.2f}'.format(last_close - last_open)), verticalalignment='bottom', horizontalalignment='left',
         color='darkblue', fontsize=8)
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
df = pd.read_csv((ax4_subject + '.csv'), parse_dates=True, index_col =0)
df.reset_index(inplace = True) 
########################################################
#Define DATA and attributes
########################################################
stock_entry = (df['Adj Close'][0])               # Set marker of last years close.
MovAvg_window1 = 10                                         #Moving Average 10 days (quick)
MovAvg_window2 = 30                                         #Moving Average 30 days (slow)
ma1 = moving_average(df['Adj Close'], MovAvg_window1)
ma2 = moving_average(df['Adj Close'], MovAvg_window2)
start = len(df['Date'][MovAvg_window2 - 1:])

ax4.plot_date(df['Date'], df['Adj Close'], '-', label='ADJ Closing Price', color = 'blue', linewidth = 1)
ax4.plot([],[], linewidth = 2, label = 'ADJ Close yr ago' , color = 'k', alpha = 0.9)
ax4.axhline(df['Adj Close'][0], color = 'k', linewidth = 2)
ax4.fill_between(df['Date'], df['Adj Close'], stock_entry, where = (df['Adj Close'] > stock_entry), facecolor='g', alpha=0.6)
ax4.fill_between(df['Date'], df['Adj Close'], stock_entry, where = (df['Adj Close'] < stock_entry), facecolor='r', alpha=0.6)
rotate_xaxis(ax4)
ax4.grid(True, color='lightgreen', linestyle = '-', linewidth=2)
set_spines(ax4)
ax4.tick_params(axis = 'x', colors = '#890b86')
ax4.tick_params(axis = 'y', colors = 'g', labelsize = 6)
ax4.set_title(ax4_subject, color = '#353335', size = 10)
set_labels(ax4)
ax4.set_color = '#890b86'
ax4.legend(fontsize = 6, fancybox = True, loc = 0, markerscale = -0.5, framealpha  = 0.5, facecolor = '#f9ffb7')

rotate_xaxis(ax5)
ax5.fill_between(df['Date'][- start:], ma1[-start:], ma2[-start:], where = (ma1[-start:] > ma2[-start:]), facecolor='g', alpha=0.6)
ax5.fill_between(df['Date'][- start:], ma1[-start:], ma2[-start:], where = (ma1[-start:] < ma2[-start:]), facecolor='red', alpha=0.6)
ax5.grid(True, color='lightgreen', linestyle = '-', linewidth=2)
ax5.plot(df['Date'][- start:], ma1[- start:], color = 'b', linewidth = 1)     #Have to skip date ahead 10days (MovAvg_window1)
ax5.plot(df['Date'][- start:], ma2[- start:], color = 'k', linewidth = 1 )      #Have to skip date ahead 30 days (MovAvg_window2)
set_spines(ax5)
ax5.tick_params(axis = 'x', colors = '#890b86')
ax5.tick_params(axis = 'y', colors = 'g', labelsize = 6)
ax5.plot([],[], linewidth = 2, label = '10d mov. avg.' , color = 'b', alpha = 0.9)
ax5.plot([],[], linewidth = 2, label = '30d mov. avg.' , color = 'k', alpha = 0.9)
ax5.legend(fontsize = 6, fancybox = True, loc = 0, markerscale = -0.5, framealpha  = 0.5, facecolor = '#dde29a')
#set_labels(ax5)
ax5.set_ylabel('Moving Average', fontsize=8, fontweight =5, color = 'r')


ax6.tick_params(axis = 'x', colors = '#890b86')

ax6.plot_date(df['Date'], df['Volume'], '-', label='Volume', color = 'blue', linewidth = 1)
ax6.set_ylim( df['Volume'].min(),df['Volume'].max())
rotate_xaxis(ax6)
set_spines(ax6)
ax6.set_ylabel('Volume', fontsize=8, fontweight =5, color = 'b')
ax6.tick_params(axis = 'y', colors = 'k', labelsize = 6)

last_rec = (len(df) -1)
#ax7.text(0,-2.1,str(df.iloc[last_rec]), fontsize=9, fontweight = 20)\
last_open = df['Open'].iloc[-1]
last_high = df['High'].iloc[-1]
last_low  = df['Low'].iloc[-1]
last_vol  = df['Volume'].iloc[-1]
last_close = df['Close'].iloc[-1]
hide_frame(ax8)
ax8.text(1,1,'Open:' + '{:20,.2f}'.format(last_open) + '     ', verticalalignment='bottom', horizontalalignment='left',
         color='darkblue', fontsize=8)

ax8.text(1,1,'Close:' + '{:20,.2f}'.format(last_close), verticalalignment='top', horizontalalignment='left',
         color='darkblue', fontsize=8)
ax8.text(1,1,'High:' + '{:20,.2f}'.format(last_high) + '     ', verticalalignment='bottom', horizontalalignment='right',
         color='darkblue', fontsize=8)
ax8.text(1,1,'Low:' + '{:20,.2f}'.format(last_low)+ '     ', verticalalignment='top', horizontalalignment='right',
         color='darkblue', fontsize=8)
ax8.text(0.5,0.25,"                  Diff:     " + str('{:5,.2f}'.format(last_high - last_low)), verticalalignment='bottom', horizontalalignment='right',
         color='darkblue', fontsize=8)
ax8.text(0.5,0.25,"                     Diff:     " + str('{:5,.2f}'.format(last_close - last_open)), verticalalignment='bottom', horizontalalignment='left',
         color='darkblue', fontsize=8)


plt.rc('ytick', labelsize=6 )    # fontsize of the tick labels
plt.subplots_adjust(left = 0.10, bottom = 0.16, right = 0.920, top = 0.93, wspace = 0.2, hspace = -.1)
plt.show()
fig.savefig('Figure1.png')
os.chdir(currPath)
os.system('python3 send_mail.py')
