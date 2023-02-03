import numpy as np
import matplotlib.pyplot as plt


# plotting 1d potentials
r = np.linspace(0,3,256)

fig,ax = plt.subplots(2,1, sharex=True, figsize=(3,5))
ax[0].plot(r,1/r)
ax[1].plot(r,-1/r,color='#ff9300')
ax[1].set_xlabel("r")
ax[0].set_ylabel("V (q>0)")
ax[1].set_ylabel("V (q<0)")
ax[1].set_yticklabels([])
ax[1].set_xticklabels([])
ax[0].set_yticklabels([])
plt.tight_layout()
plt.savefig("../figs/coulombs.pdf")
plt.close()

# plotting field+potential construction
frames = 24

fig =plt.figure(figsize=(3,3))
# plot radial field lines
x0,y0 = 0,0
R = 1.5
RR = 2
dangle = np.pi/256
narc = 16
alpha = 0.4

for arrows in range(1,25):
    # draw outer circle
    fig =plt.figure(figsize=(3,3), dpi=320)
    outer = np.array([2.5*np.array([np.cos(np.pi/32*i),np.sin(np.pi/32*i)]) for i in range(64)])
    plt.plot(outer[:,0],outer[:,1],lw=0)
    for f in range(arrows):
        angle = -np.pi/12*f+np.pi/3
        dx,dy = np.cos(angle),np.sin(angle)
        x,y = RR*np.cos(angle),RR*np.sin(angle)
        # draw field
        # color = plt.cm.rainbow(f/(frames-1))
        color='k'
        a = plt.arrow(x0,y0,dx,dy,head_width=0.1,edgecolor=color, facecolor=color)
        
        plt.plot([x0,x],[y0,y], '-', color=color)
        
        # draw circular arc
        if arrows==24:
            color = 'green'
        elif f==arrows-1:
            color='red'
        else:
            color='green'
        
    
        arc = np.array([(R*np.cos(angle+int(narc/2)*dangle-p*dangle),R*np.sin(angle+int(narc/2)*dangle-p*dangle)) for p in range(narc)])
        plt.plot(arc[:,0], arc[:,1], color=color)
    
    circle = plt.Circle((0,0),0.2, facecolor='r', edgecolor='k', zorder=10)
    plt.gca().add_patch(circle)
    # label = plt.gca().annotate("+", xy=(0, 0), fontsize=30, color='w',zorder=11)
    plt.text(-0.13,-0.09,"+", color='w',zorder=13)
    plt.axis('off')
    plt.axis('equal')
    plt.xlim(-2.5,2.5)
    plt.ylim(-2.5,2.5)    
    plt.savefig(f"../figs/seq/frame{arrows}.png")
    plt.close()
plt.close()
plt.show()