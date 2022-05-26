import matplotlib
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['hatch.color'] = 'dimgrey'
#labels = ['bibtext', 'reuters', 'delicious', 'rcv1', 'voc', 'nuswide']
#mymeans = [0.4175476584,0.5219118482,0.2197873786,0.4971334293,0.6951481795,0.4890080726]
#sotameans = [0.503702932,0.495831539,0.3986482638,0.5693257537,0.7201927495,0.5618206847]
labels = ['bibtext', 'reuters', 'delicious', 'rcv1', 'voc']
mymeans = [0.4175476584,0.5219118482,0.2197873786,0.4971334293,0.6951481795]
sotameans = [0.503702932,0.495831539,0.3986482638,0.5693257537,0.7201927495]

x = np.arange(len(labels))  # the label locations
width = 0.3  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, mymeans, width, label='Word Emb w/ Reg', hatch="x", color='lightblue')
rects2 = ax.bar(x + width/2, sotameans, width, label='No Word Emb', hatch="//", color='bisque')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Normalized Dist')
#ax.set_title('')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.show()
fig.savefig("emb_dist.pdf", bbox_inches='tight')
