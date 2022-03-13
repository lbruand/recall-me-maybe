import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix
from pywaffle import Waffle
import numpy as np
import math
import itertools

def cascade_rounding(vector:  np.array, size) -> np.array:
    rows, cols = size
    total_boxes = rows * cols
    ratio_vector = vector * total_boxes/ sum(vector)
    floor_vector = np.floor(ratio_vector).astype(int)
    roundoff_error = ratio_vector - floor_vector
    argsort = np.argsort(roundoff_error)
    carry = np.zeros(vector.shape, dtype=int)
    for i in range(total_boxes - np.sum(floor_vector)):
        ix = argsort[i]
        carry[ix] = 1
    return floor_vector + carry



def build_waffle_matrix(size, 
                        cm):
    rows, cols = size
    hmap = np.ones( size, dtype=int)
    tn, fp, fn, tp = cm.ravel()
    
    vector = np.array([fn, tp, fp, tn])
    fn_boxes, tp_boxes, fp_boxes, tn_boxes = cascade_rounding(vector, size)

    fn_tp_boxes = fn_boxes + tp_boxes

    for n in range(fn_tp_boxes, rows * cols):
        ix = n % rows
        iy = n // rows
        hmap[ix, iy] = 4
    normalize =  rows * cols / sum(cm.ravel())

    col_part = fn_tp_boxes // rows
    h = int(min(
        max( math.ceil(tp/col_part * normalize), 
            math.ceil(fp / (cols - col_part) * normalize ))+1,
        rows))
    
    centerx = int(round(rows / 2))
    centery = col_part

    midh = int(math.ceil(h/2))


    def boxes_generator(direction=-1, expected_value=1, recursionguard=500):
        for n in range(recursionguard):
            ix = min(centerx - midh + (n % (h)), rows - 1)
            iy = min(centery + direction * (n// (h) - 1), cols - 1)
            n += 1
            if hmap[ix, iy] == expected_value:
                yield ix, iy
        return # TODO : breaks one test : assert False, "This should not happen - recursionguard"


    def update_boxes(hmap, gene, nb_boxes: int, from_cat: int, to_cat: int):
        toUpdate = itertools.islice(gene, nb_boxes)
        for xy in toUpdate:
            ix, iy = xy
            assert(hmap[ix, iy] == from_cat)
            hmap[ix, iy] = to_cat

    if tp_boxes > 0:
        tp_boxes_gen = boxes_generator(direction=-1, expected_value=1)
        update_boxes(hmap, tp_boxes_gen, tp_boxes, from_cat=1, to_cat=2)

    if fp_boxes > 0:
        fp_boxes_gen = boxes_generator(direction=1, expected_value=4)
        update_boxes(hmap, fp_boxes_gen, fp_boxes, from_cat=4, to_cat=3)

        
    return hmap

def subplot_waffle_matrix(ax, 
                          hmap,
                          cmap,
                          linewidth=5,
                          linecolor="white"   
                      ):
    im = ax.imshow(hmap, cmap=cmap)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    unitmove = 1.0
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks(np.arange(hmap.shape[1]+unitmove)-unitmove/2, minor=True)
    ax.set_yticks(np.arange(hmap.shape[0]+unitmove)-unitmove/2, minor=True)
    
    ax.grid(which="minor", color=linecolor, linestyle='-', linewidth=linewidth)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

def add_fraction_bar(ax):
    p = plt.Rectangle((-0.1, 1.1), 1.2, 0.01, fill=False)
    p.set_transform(ax.transAxes)
    p.set_clip_on(False)
    ax.add_patch(p)

def add_value(ax, value, desc):
    ax.text (1.2, 1.05, 
             f' = { 100.0 * value :.1f}% = {desc}', 
             transform=ax.transAxes,
             fontsize=15)    
    
def plot_waffle_matrix(hmap, 
                       cm,
                       cmap = mpl.colors.ListedColormap(["orange", "darkgreen", "red", "lightgrey"]), 
                       linewidth=5
                      ):
    ymarg = 3.0
    tn, fp, fn, tp = cm.ravel()
    fig = plt.figure(constrained_layout=False, facecolor="white")
    gs = GridSpec(4, 4, figure=fig)
       
    axbig = fig.add_subplot(gs[0:4, 0:3])
    axbig.margins(y=ymarg)
    subplot_waffle_matrix(axbig, hmap, cmap, linewidth)
    
    ax1 = fig.add_subplot(gs[0, 3])
    
    subplot_waffle_matrix(ax1, hmap, 
                          mpl.colors.ListedColormap(["white", "darkgreen", "white", "white"]), 
                          linewidth=2)
    ax1.margins(y=ymarg)
    ax2 = fig.add_subplot(gs[1, 3])

    add_fraction_bar(ax2)
    

    add_value(ax=ax2, value=tp/(tp+fp), desc="prec" )
    
    subplot_waffle_matrix(ax2, hmap, 
                          mpl.colors.ListedColormap(["white", "darkgreen", "red", "white"]), 
                          linewidth=2,
                          linecolor="white")
    ax2.margins(y=ymarg)
    ax3 = fig.add_subplot(gs[2, 3])
    
    subplot_waffle_matrix(ax3, hmap, 
                          mpl.colors.ListedColormap(["white", "darkgreen", "white", "white"]), 
                          linewidth=2)

    ax3.margins(y=ymarg)
    ax4 = fig.add_subplot(gs[3, 3])
    add_fraction_bar(ax4)
    add_value(ax=ax4, value=tp/(fn+tp), desc="recall" )
    subplot_waffle_matrix(ax4, hmap, 
                          mpl.colors.ListedColormap(["orange", "darkgreen", "white", "white"]), 
                          linewidth=2,
                          linecolor="white")    
    ax4.margins(y=ymarg)
    fig.legend(
                handles=[Patch(color="orange"), 
                         Patch(color="darkgreen"),
                         Patch(color="white"),
                         Patch(color="lightgrey"),
                         Patch(color="red"),
                         Patch(color="white"),
                         ],
                labels=["false negative", "true positive", "relevant", "true negative", "false positive",  "irrelevant"],
                loc='lower left',
                bbox_to_anchor= (0.07, -0.03),
                edgecolor='k',
                ncol=2,
                frameon=False
              )
    return fig

