import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

np.random.seed(18)

N,P,NAME=20,0.4,"PUSHKAR"
DT,T_LETTER=0.02,4.0
STEPS=int(T_LETTER/DT)
K_GAIN,SCALE=2.0,6.0
FPS,STRIDE=30,4

def generate_graph(n,p,seed=1):
    while True:
        G=nx.erdos_renyi_graph(n,p,seed=seed)
        if nx.is_connected(G): return G
        seed+=1

G=generate_graph(N,P)
Adj=nx.to_numpy_array(G)
print(f"Graph: N={N}, p={P}, edges={G.number_of_edges()}, connected={nx.is_connected(G)}")

x0=np.random.uniform(-SCALE,SCALE,(N,2))

LETTER_SEGMENTS={
"P":[((-.6,-1),(-.6,1)),((-.6,1),(.3,1)),((.3,1),(.6,.7)),((.6,.7),(.6,.3)),((.6,.3),(.3,0)),((.3,0),(-.6,0))],
"U":[((-.6,1),(-.6,-.5)),((-.6,-.5),(-.3,-1)),((-.3,-1),(.3,-1)),((.3,-1),(.6,-.5)),((.6,-.5),(.6,1))],
"S":[((.6,.8),(.3,1)),((.3,1),(-.4,1)),((-.4,1),(-.6,.7)),((-.6,.7),(-.6,.2)),((-.6,.2),(.4,-.2)),((.4,-.2),(.6,-.5)),((.6,-.5),(.4,-1)),((.4,-1),(-.4,-1)),((-.4,-1),(-.6,-.8))],
"H":[((-.6,-1),(-.6,1)),((.6,-1),(.6,1)),((-.6,0),(.6,0))],
"K":[((-.6,-1),(-.6,1)),((-.6,0),(.6,1)),((-.6,0),(.6,-1))],
"A":[((-.7,-1),(0,1)),((0,1),(.7,-1)),((-.4,0),(.4,0))],
"R":[((-.6,-1),(-.6,1)),((-.6,1),(.3,1)),((.3,1),(.6,.7)),((.6,.7),(.6,.3)),((.6,.3),(.3,0)),((.3,0),(-.6,0)),((0,0),(.7,-1))]
}

def sample_segment(a,b,n):
    a,b=np.array(a),np.array(b)
    t=np.linspace(0,1,n)
    return np.outer(1-t,a)+np.outer(t,b)

def letter_points(letter,n=N):
    seg=LETTER_SEGMENTS[letter]
    lengths=[np.linalg.norm(np.array(b)-a) for a,b in seg]
    counts=[max(1,round(n*l/sum(lengths))) for l in lengths]

    while sum(counts)>n:
        i=np.argmax(counts)
        if counts[i]>1: counts[i]-=1
    while sum(counts)<n: counts[np.argmax(lengths)]+=1

    return np.vstack([sample_segment(a,b,c) for (a,b),c in zip(seg,counts)])*SCALE

targets=[letter_points(c) for c in NAME]

def assign_targets(x,target):
    remaining=list(range(N))
    assigned=np.zeros_like(target)
    for i in range(N):
        d=np.linalg.norm(target[remaining]-x[i],axis=1)
        assigned[i]=target[remaining.pop(np.argmin(d))]
    return assigned

def formation_control(x,target):
    dx=np.zeros_like(x)
    for i in range(N):
        nbr=np.where(Adj[i]>0)[0]
        if len(nbr):
            dx[i]=-K_GAIN*(((x[i]-x[nbr])-(target[i]-target[nbr])).sum(axis=0))
    return dx

trajectory=[x0.copy()]
x=x0.copy()

for letter,target in zip(NAME,targets):
    print(f"Forming letter: {letter}")
    target=assign_targets(x,target)
    for _ in range(STEPS):
        x+=DT*formation_control(x,target)
        trajectory.append(x.copy())

trajectory=np.array(trajectory)
print(f"Simulation finished: {len(trajectory)} states")

# Animation
fig,ax=plt.subplots(figsize=(8,8))
margin=2
ax.set_xlim(trajectory[:,:,0].min()-margin,trajectory[:,:,0].max()+margin)
ax.set_ylim(trajectory[:,:,1].min()-margin,trajectory[:,:,1].max()+margin)
ax.set_aspect("equal")
ax.set_title(f"Formation Control: {NAME} ({N} agents)")

scat=ax.scatter([],[],s=70,c="crimson",edgecolors="black",linewidths=.5,zorder=2)
edges=[ax.plot([],[],color="gray",lw=.5,alpha=.5)[0] for _ in G.edges()]
label=ax.text(.02,.96,"",transform=ax.transAxes,fontsize=16,fontweight="bold",va="top")

frames=np.arange(0,len(trajectory),STRIDE)

def current_letter(step):
    return NAME[min(step//STEPS,len(NAME)-1)]

def init():
    scat.set_offsets(np.zeros((N,2)))
    for e in edges: e.set_data([],[])
    label.set_text("")
    return [scat,label]+edges

def update(f):
    step=frames[f]
    pos=trajectory[step]
    scat.set_offsets(pos)
    for e,(i,j) in zip(edges,G.edges()):
        e.set_data(pos[[i,j],0],pos[[i,j],1])
    label.set_text(f"Target letter: {current_letter(step)}")
    return [scat,label]+edges

anim=animation.FuncAnimation(fig,update,frames=len(frames),init_func=init,
                             blit=True,interval=1000/FPS)

try:
    anim.save("pushkar_formation_control.mp4",
              writer=animation.FFMpegWriter(fps=FPS,bitrate=1800))
    print("Saved: pushkar_formation_control.mp4")
except Exception as e:
    print(f"FFmpeg unavailable: {e}")
    anim.save("pushkar_formation_control.gif",
              writer=animation.PillowWriter(fps=FPS))
    print("Saved: pushkar_formation_control.gif")
plt.close(fig)

# Graph topology
fig,ax=plt.subplots(figsize=(6,6))
pos=nx.spring_layout(G,seed=1)
nx.draw(G,pos,ax=ax,with_labels=True,node_color="crimson",
        edge_color="gray",font_color="white",font_size=8)
ax.set_title(f"Erdos-Renyi Graph (N={N}, p={P})")
fig.savefig("erdos_renyi_graph.png",dpi=150,bbox_inches="tight")
plt.close(fig)
print("Saved: erdos_renyi_graph.png")