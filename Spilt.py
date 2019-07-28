import os
import numpy as np
import config
def writefile(filename,a):
    f=open(filename,'w')
    for x in a:
        f.writelines(str(x)+'\n')
    f.close()
N=config.Sample_num
M = int(N *0.1)
s=np.random.choice(N,M,replace=False)
n=np.ones(shape=N)
print(n)
n[s]=0
print(n)
l=[]
for x in range(0,N):
    if (n[x]!=0):
        l.append(x)
print(l)
print(len(l))
train=np.array(l)
writefile('train.txt',train)
val=s[: M//2]
test=s[M//2:]
print(val.shape)
print(test.shape)
assert val.shape[0]+test.shape[0]==M
writefile('val.txt',val)
writefile('test.txt',test)
