import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

np.random.seed(42)

N,Tmax=10,50
p=0.3
dim=3
alpha=0.05
K=150

# Connected Erdős-Rényi graph
while True:
    G=nx.erdos_renyi_graph(N,p,seed=np.random.randint(10000))
    if nx.is_connected(G):
        break

A=np.zeros((N,N))
for i in range(N):
    for j in range(N):
        if i!=j and G.has_edge(i,j):
            A[i,j]=1/(1+max(G.degree(i),G.degree(j)))
    A[i,i]=1-A[i].sum()

# Initial state and noise covariances
zbar0=np.array([0.,0.,0.])
Sigma0=np.diag([1.,1.,1.])
Sigmaw=np.diag([0.08,0.08,0.08])
Sigmav=[np.diag([0.3,0.3,0.3]) for _ in range(N)]

# Fixed drone locations
X=np.random.uniform(-5,5,(N,3))

# True intruder trajectory
z=np.zeros((Tmax+1,dim))
z[0]=np.random.multivariate_normal(zbar0,Sigma0)
for t in range(Tmax):
    z[t+1]=z[t]+np.random.multivariate_normal(np.zeros(dim),Sigmaw)

# Measurements
Y=np.zeros((Tmax+1,N,dim))
for t in range(Tmax+1):
    for i in range(N):
        v=np.random.multivariate_normal(np.zeros(dim),Sigmav[i])
        Y[t,i]=X[i]-z[t]+v

# Distributed gradient tracking
zhat=np.zeros((Tmax+1,dim))

for t in range(Tmax+1):
    zi=X-Y[t]
    s=np.zeros((N,dim))

    for i in range(N):
        Rinv=np.linalg.pinv(Sigmav[i])
        s[i]=Rinv@(zi[i]-X[i]+Y[t,i])

    for _ in range(K):
        znew=A@zi-alpha*s
        snew=np.zeros((N,dim))

        for i in range(N):
            Rinv=np.linalg.pinv(Sigmav[i])
            grad=Rinv@(znew[i]-X[i]+Y[t,i])
            oldgrad=Rinv@(zi[i]-X[i]+Y[t,i])
            snew[i]=grad-oldgrad

        s=A@s+snew
        zi=znew

    zhat[t]=zi.mean(axis=0)

error=zhat-z
error_norm=np.linalg.norm(error,axis=1)

# Communication graph
pos=nx.spring_layout(G,seed=42)
plt.figure(figsize=(6,5))
nx.draw(G,pos,with_labels=True,node_size=500)
plt.title("Erdos-Renyi Communication Graph")
plt.savefig("communication_graph.png",dpi=300,bbox_inches="tight")
plt.show()

# 3D trajectories
fig=plt.figure(figsize=(8,6))
ax=fig.add_subplot(111,projection="3d")
ax.scatter(X[:,0],X[:,1],X[:,2],s=50,label="Drones")
ax.plot(z[:,0],z[:,1],z[:,2],label="True trajectory")
ax.plot(zhat[:,0],zhat[:,1],zhat[:,2],"--",label="Estimated trajectory")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Intruder Tracking")
ax.legend()
plt.savefig("intruder_tracking.png",dpi=300,bbox_inches="tight")
plt.show()

# Estimation error
plt.figure(figsize=(8,4))
plt.plot(error_norm)
plt.xlabel("Time")
plt.ylabel(r"$\|e(t)\|$")
plt.title("Estimation Error")
plt.grid()
plt.savefig("estimation_error.png",dpi=300,bbox_inches="tight")
plt.show()

print("Final estimation error:",error_norm[-1])
print("Mean estimation error:",error_norm.mean())