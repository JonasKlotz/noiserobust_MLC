import csv
import sys


filename = sys.argv[1]
train_key = sys.argv[2]
valid_key = sys.argv[3]
outfile = sys.argv[4]
train_stats = []
valid_stats = []
epochs = []

with open(filename, 'r') as f:
    for line in f:
        if 'Epoch' in line:
            tmp = line.split(' ')
            idx = tmp.index('Epoch')
            epochs.append(int(tmp[idx+1]))
        if train_key in line:
            tmp = line.split(':')
            train_stats.append(float(tmp[1]))
        if valid_key in line:
            tmp = line.split(':')
            valid_stats.append(float(tmp[1]))

rows = zip(epochs, train_stats, valid_stats)
with open(outfile+'.csv', 'w') as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)

