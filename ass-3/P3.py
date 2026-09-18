import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pathlib import Path

np.random.seed(42)

N,M=4,8
p=0.6
rho=1.0
K=100

# Connected Erdos-Renyi graph
while True:
    G=nx.erdos_renyi_graph(N,p,seed=np.random.randint(10000))
    if nx.is_connected(G):
        break

# Capacities
b=np.array([1,2,2,3])

# Cost matrices
C1=np.random.uniform(1,10,(N,M))
C2=np.random.uniform(1,10,(N,M))

def edge_key(i,j):
    return tuple(sorted((i,j)))

def solve_local(i,C,Z,U):
    def objective(v):
        X=v.reshape(N,M)
        cost=C[i]@X[i]
        penalty=0
        for j in G.neighbors(i):
            key=edge_key(i,j)
            penalty+=np.sum((X-Z[key]+U[key][i])**2)
        return cost+rho/2*penalty

    cons=[{'type':'eq',
           'fun':lambda v:v.reshape(N,M)[i].sum()-b[i]}]

    for j in range(M):
        cons.append({
            'type':'eq',
            'fun':lambda v,j=j:v.reshape(N,M)[:,j].sum()-1
        })

    x0=np.ones((N,M))/N

    result=minimize(
        objective,x0.ravel(),
        bounds=[(0,1)]*(N*M),
        constraints=cons,
        method='SLSQP',
        options={'maxiter':300,'ftol':1e-8}
    )

    if not result.success:
        print(f"Warning: local optimization failed for agent {i}: {result.message}")

    return result.x.reshape(N,M)

def admm(C):
    X=[np.ones((N,M))/N for _ in range(N)]
    Z={}
    U={}

    for i,j in G.edges:
        key=edge_key(i,j)
        Z[key]=np.ones((N,M))/N
        U[key]={i:np.zeros((N,M)),j:np.zeros((N,M))}

    for _ in range(K):
        Xnew=[]

        for i in range(N):
            Xnew.append(solve_local(i,C,Z,U))

        X=Xnew

        for i,j in G.edges:
            key=edge_key(i,j)
            Z[key]=0.5*(X[i]+U[key][i]+X[j]+U[key][j])

        for i,j in G.edges:
            key=edge_key(i,j)
            U[key][i]+=X[i]-Z[key]
            U[key][j]+=X[j]-Z[key]

    Xavg=np.mean(X,axis=0)
    return Xavg

def get_allocation(X):
    A=np.zeros((N,M),dtype=int)
    remaining=b.copy()

    for j in np.argsort(np.min(X,axis=0)):
        agents=np.argsort(-X[:,j])
        for i in agents:
            if remaining[i]>0:
                A[i,j]=1
                remaining[i]-=1
                break

    return A

# ADMM
X1=admm(C1)
X2=admm(C2)

A1=get_allocation(X1)
A2=get_allocation(X2)

# Graph
pos=nx.spring_layout(G,seed=42)
plt.figure(figsize=(6,5))
nx.draw(G,pos,with_labels=True,node_size=700)
plt.title("Erdos-Renyi Communication Graph")
plt.savefig("problem3_graph.png",dpi=300,bbox_inches="tight")
plt.show()

# Allocation 1
fig,ax=plt.subplots(figsize=(8,4))
ax.imshow(A1,cmap="Greys",vmin=0,vmax=1)
ax.set_xlabel("Tasks")
ax.set_ylabel("Agents")
ax.set_title("Task Allocation - Cost Matrix 1")
ax.set_xticks(range(M))
ax.set_yticks(range(N))
plt.savefig("allocation_cost1.png",dpi=300,bbox_inches="tight")
plt.show()

# Allocation 2
fig,ax=plt.subplots(figsize=(8,4))
ax.imshow(A2,cmap="Greys",vmin=0,vmax=1)
ax.set_xlabel("Tasks")
ax.set_ylabel("Agents")
ax.set_title("Task Allocation - Cost Matrix 2")
ax.set_xticks(range(M))
ax.set_yticks(range(N))
plt.savefig("allocation_cost2.png",dpi=300,bbox_inches="tight")
plt.show()

print("Capacities:",b)
print("\nCost Matrix 1:\n",np.round(C1,2))
print("\nAllocation 1:\n",A1)
print("\nCost Matrix 2:\n",np.round(C2,2))
print("\nAllocation 2:\n",A2)
print("\nTotal cost 1:",np.sum(C1*A1))
print("Total cost 2:",np.sum(C2*A2))