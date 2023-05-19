from __future__ import division
import numpy
import pickle
import random


from matplotlib import pyplot
pyplot.rcParams.update({'font.size': 20})
#from matplotlib.mlab import griddata
#from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.image as mpimg

xs = numpy.linspace(0, 1, 1000)
ys = numpy.linspace(0, 1, 1000)
#ks = [3, 4, 5, 6]
ks = [6]
pers = 0.7

fig_vec = numpy.zeros([len(xs),len(ys),3])
sumAmps = 0

rgb1 = numpy.array([1,0,0])
rgb2 = numpy.array([0,1,0])
rgb3 = numpy.array([0,0,1])

for f in ks:
    sumAmps = sumAmps + pers**f

for f in ks:
    freq = 2**f
    #print(f)
    G_vec = numpy.zeros([freq+1, freq+1,2])

    for jGcounter in range(0, freq+1):
        for kGcounter in range(0, freq+1):
            #randG = numpy.array([numpy.random.random()*2 - 1, numpy.random.random()*2 - 1])
            #G_vec[jGcounter][kGcounter] = randG
            randG = numpy.array([[1,1],[1,-1],[-1,1],[-1,-1]])
            #randG = randG/numpy.sqrt((numpy.sum(randG**2)))
            #randG = randG/numpy.sqrt(2)
            G_vec[jGcounter][kGcounter] = random.choice(randG)

    for jcounter,j in enumerate(xs):
        print(freq, jcounter)
        for kcounter,k in enumerate(ys):
            Gbox = numpy.array([jcounter//(len(xs)/freq), kcounter//(len(ys)/freq)]).astype(int) #box youre in
            D_TL = numpy.array([(jcounter - Gbox[0]*len(xs)/freq)/(len(xs)/freq-1), (kcounter - Gbox[1]*len(ys)/freq)/(len(ys)/freq-1)])
            D_TR = numpy.array([(jcounter + 1 - (Gbox[0]+1)*len(xs)/freq)/(len(xs)/freq-1), (kcounter - Gbox[1]*len(ys)/freq)/(len(ys)/freq-1)])
            D_BL = numpy.array([(jcounter - (Gbox[0])*len(xs)/freq)/(len(xs)/freq-1), (kcounter + 1 - (Gbox[1]+1)*len(ys)/freq)/(len(ys)/freq-1)])
            D_BR = numpy.array([(jcounter + 1 - (Gbox[0]+1)*len(xs)/freq)/(len(xs)/freq-1), (kcounter + 1 - (Gbox[1]+1)*len(ys)/freq)/(len(ys)/freq-1)])
            #print(jcounter, kcounter, Gbox, D_TL, D_TR, D_BL, D_BR)
            #print(G_vec[Gbox[0]][Gbox[1]])
            dot_TL = numpy.dot(G_vec[Gbox[0]][Gbox[1]], D_TL)
            dot_TL = (dot_TL/2 + 1)/2
            dot_TR = numpy.dot(G_vec[Gbox[0]+1][Gbox[1]], D_TR)
            dot_TR = (dot_TR/2 + 1)/2
            dot_BL = numpy.dot(G_vec[Gbox[0]][Gbox[1]+1], D_BL)
            dot_BL = (dot_BL/2 + 1)/2
            dot_BR = numpy.dot(G_vec[Gbox[0]+1][Gbox[1]+1], D_BR)
            dot_BR = (dot_BR/2 + 1)/2
            D_TL = 6*abs(D_TL)**5 - 15*abs(D_TL)**4 + 10*abs(D_TL)**3
            D_TR = 6*abs(D_TR)**5 - 15*abs(D_TR)**4 + 10*abs(D_TR)**3
            D_BL = 6*abs(D_BL)**5 - 15*abs(D_BL)**4 + 10*abs(D_BL)**3
            D_BR = 6*abs(D_BR)**5 - 15*abs(D_BR)**4 + 10*abs(D_BR)**3
            top_lint = dot_TL + D_TL[0]*(dot_TR - dot_TL)
            #top_lint = 6*abs(top_lint)**5 - 15*abs(top_lint)**4 + 10*abs(top_lint)**3
            bot_lint = dot_BL + D_BL[0]*(dot_BR - dot_BL)
            #bot_lint = 6*abs(bot_lint)**5 - 15*abs(bot_lint)**4 + 10*abs(bot_lint)**3
            fig_vec[jcounter][kcounter][0] = 255/255
            fig_vec[jcounter][kcounter][1] = fig_vec[jcounter][kcounter][1] + (top_lint + D_TL[1]*(bot_lint - top_lint))*pers**f/(sumAmps/1.55)
            if fig_vec[jcounter][kcounter][1] > 1:
                fig_vec[jcounter][kcounter][1] = 1
            fig_vec[jcounter][kcounter][2] = 0
            #fig_vec[jcounter][kcounter] = fig_vec[jcounter][kcounter] + (top_lint + D_TL[1]*(bot_lint - top_lint))*pers**f/(sumAmps)
            #print(jcounter, kcounter, Gbox, G_vec[Gbox[0]][Gbox[1]], G_vec[Gbox[0]+1][Gbox[1]], G_vec[Gbox[0]][Gbox[1]+1], G_vec[Gbox[0]+1][Gbox[1]+1], D_TL, D_TR, D_BL, D_BR, dot_TL, dot_TR, dot_BL, dot_BR, fig_vec[jcounter][kcounter])
            a = numpy.random.random()
            b = numpy.random.random()
            c = numpy.random.random()
            d = numpy.random.random()
            w_TL = 2*abs(D_TL)**3 - 3*abs(D_TL)**2 + 1
            w_TR = 2*abs(D_TL)**3 - 3*abs(D_TL)**2 + 1
            w_BL = 2*abs(D_TL)**3 - 3*abs(D_TL)**2 + 1
            w_BR = 2*abs(D_TL)**3 - 3*abs(D_TL)**2 + 1
            #fig_vec[len(ys) - 1 - kcounter][jcounter] = a*rgb1 + b*rgb2 + c*rgb3 + d*rgb4
            #fig_vec[jcounter][kcounter] = numpy.array([jcounter ,-0.5,-0.5])
            #fig_vec[jcounter][kcounter] = w_TL[0]*w_TL[1]*dot_TL + w_TR[0]*w_TR[1]*dot_TR + w_BL[0]*w_BL[1]*dot_BL + w_BR[0]*w_BR[1]*dot_BR


#####################################
##          plot coverage          ##
#####################################

fig = pyplot.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
ax.imshow(fig_vec, extent=[0, 1, 0, 1], aspect='auto', interpolation='none')


#ax.text(1.1,1.9,r'$U_{applied}$ $=$ $%3.2f V_{RHE}$'%i, verticalalignment='top', horizontalalignment='left', fontsize=20)
#ax.set_ylabel(r'$\Delta$$G_{OH},$ $eV$', fontsize=25, )
#ax.set_xlabel(r'$\Delta$$G_{O}$ $-$ $\Delta$$G_{OH},$ $eV$', fontsize=25)
    

ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_xticks([])
ax.set_yticks([])

fig.tight_layout()
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.set_frame_on(False)
ax.set_xticks([])
fig.savefig('figname.png', bbox_inches='tight', pad_inches=0)
pyplot.close()
