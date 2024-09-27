import os
import sys
import subprocess

seq = sys.argv[1]
command = f'obabel|{seq}_noh.pdb|-osmi|-d'
subprocess.run(command.split('|'), stdout=open(f'{seq}_smi.txt', 'w'))
with open(f'{seq}_smi.txt', 'r') as f:
    line = f.readlines()[0]
    split_line = line.split()
    split_line = [s for s in split_line if 'pdb' not in s]
with open(f'{seq}_smi.txt', 'w') as f:
    f.write(' '.join(split_line))
