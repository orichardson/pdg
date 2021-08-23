
# TODO history


import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.cm import get_cmap
from tracing import TraceStore

cmaps = [get_cmap(cmn) for cmn in ["Blues", "Reds", "Greens", "Greys", "Oranges", "Purples"]]

from utils import nparray_of

def pca_view(*unnamedtraces,
        ax=None, arrows=True, transform=None,
        **kwtraces):
    
    nametrace = [("%d"%i, t) for i,t in enumerate(unnamedtraces)] + [(k,t) for k,t in kwtraces.items()]
    alltraces = np.vstack([nparray_of(trace).reshape(len(trace),-1) for n,trace in nametrace])

    if transform == None:
        pca = PCA(n_components=2)
        pca.fit(alltraces)
        expl = pca.explained_variance_ratio_
        print("Explained_variance:", expl, "\t(total: %f)"%sum(expl))
        
        transform = pca.transform

    #pca.transform(ddata).shape
    if ax == None:
        fig, ax = plt.subplots()
    
    artistname = []
    for i,(n,t) in enumerate(nametrace):
        ddata = nparray_of(t).reshape(len(t),-1)
        X,Y = transform(ddata)[:,:].T
        U,V = np.diff(X), np.diff(Y) 
        # norm = np.sqrt(U**2 + V**2)
    #     norm = np.where(norm==0, 1, norm)
    
        cmap=cmaps[i%len(cmaps)]

        

        # print(len(X))
        if len(X) == 1:
            art = ax.scatter(X,Y, s=150, c = cmap([0.7]), alpha=0.8)
        
        elif len(X) > 1:
            ax.plot(X,Y,'-', alpha=0.1, color=cmap(0.9))
            if arrows:
                # maxuv = np.abs(U + V*1j).max()
                ax.quiver(X[:-1]+U/2, Y[:-1]+V/2, U*0.9, V*0.9, np.linspace(0.15,0.95,len(X)),
                    pivot="mid", angles="xy",headwidth=5, headaxislength=3.5,
                    width=0.005, scale_units='xy', scale=1,
                    alpha=1, zorder=4, cmap=cmap, norm=colors.Normalize(0,1)
                )
                # ax.barbs(X[:-1]+U/2, Y[:-1]+V/2, U/2, V/2, np.linspace(0,1,len(X)-1),
                #       zorder=4, alpha=0.3, cmap=cmap)

            art = ax.scatter(X,Y, s=100, c =cmap( np.linspace(0.15,0.95,len(X))  ),
                zorder=5,
                 # linewidths=1, edgecolors='k', 
                alpha=0.5)

                  
        artistname.append((art,n))

    ax.legend(*zip(*artistname))



from matplotlib.animation import FuncAnimation
from collections import defaultdict
import re

def pca_anim(ts : TraceStore, regexstr="", arrows=True):
    pca = PCA(n_components=2)
    # 
    nametrace =  [(k,t) for k,t in ts.matching(regexstr).items()]
    alltraces = np.vstack([np.array(trace).reshape(len(trace),-1) for n,trace in nametrace])
    # 
    pca.fit(alltraces)
    # expl = pca.explained_variance_ratio_
    # print("Explained_variance:", expl, "\t(total: %f)"%sum(expl))
    # 
    pattern = re.compile(regexstr)
    # 
    # artists = defaultdict(defaultdict(list))
    # 
    # #pca.transform(ddata).shape
    # # if ax == None:
    fig, ax = plt.subplots()
    # 
    # artistname = []
    # for i,(n,t) in enumerate(nametrace):
    #     ddata = np.array(t).reshape(len(t),-1)
    #     X,Y = pca.transform(ddata)[:,:].T
    #     U,V = np.diff(X), np.diff(Y) 
    #     # norm = np.sqrt(U**2 + V**2)
    # #     norm = np.where(norm==0, 1, norm)
    # 
    #     cmap=cmaps[i%len(cmaps)]
    # 
    # 
    # 
    #     # print(len(X))
    #     if len(X) == 1:
    #         art = ax.scatter(X,Y, s=150, c = cmap([0.7]), alpha=0.8)
    # 
    #     elif len(X) > 1:
    #         artists[n]['plot'] = ax.plot(X,Y,'-', alpha=0.1, color=cmap(0.9))
    #         if arrows:
    #             # maxuv = np.abs(U + V*1j).max()
    #             artists[n]['quiver'] = ax.quiver(X[:-1]+U/2, Y[:-1]+V/2, U*0.9, V*0.9, np.linspace(0.15,0.95,len(X)),
    #                 pivot="mid", angles="xy",headwidth=5, headaxislength=3.5,
    #                 width=0.005, scale_units='xy', scale=1,
    #                 alpha=1, zorder=4, cmap=cmap, norm=colors.Normalize(0,1)
    #             )
    #             # ax.barbs(X[:-1]+U/2, Y[:-1]+V/2, U/2, V/2, np.linspace(0,1,len(X)-1),
    #             #       zorder=4, alpha=0.3, cmap=cmap)
    # 
    #         art = ax.scatter(X,Y, s=100, c =cmap( np.linspace(0.15,0.95,len(X))  ),
    #             zorder=5,
    #              # linewidths=1, edgecolors='k', 
    #             alpha=0.5)
    #         artists[n]['scatter'] = art
    # 
    #     artistname.append((art,n))
    # 
    # ax.legend(*zip(*artistname))
    # 
    
    #TODO transform once.
        
    def update(frame):
        print("update @ %d"%frame)
        todraw = ts.firstN(frame+1, pattern)
        
        fig.clear()
        pca_view(ax=ax,transform=pca.transform, **todraw)
            
        
    ani = FuncAnimation(fig, update, interval=200, blit=False)
    # ani.save("/tmp/test.mp4", fps=1)
    pca_anim._last = ani
    return ani
