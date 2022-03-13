import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.gridspec import GridSpec
import numpy as np
import math
import itertools

def cascade_rounding(vector:  np.array, size) -> np.array:
    rows, cols = size
    total_boxes = rows * cols
    ratio_vector = vector * total_boxes/ sum(vector)
    floor_vector = np.floor(ratio_vector).astype(int)
    argsort = np.argsort(ratio_vector - floor_vector)
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
        return # TODO : breaks one test (iter=8): assert False, "This should not happen - recursionguard"


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
                          colormap,
                          interval_ratio_x=0.3,
                          interval_ratio_y=0.3,
                          block_aspect_ratio=1.0,
                          facecolor=(1., 1., 1., 0.0)
                      ):
    rows, cols = hmap.shape
    figure_height = 1
    block_y_length = figure_height / (
        rows + rows * interval_ratio_y - interval_ratio_y
    )
    block_x_length = block_aspect_ratio * block_y_length
    x_full = (1 + interval_ratio_x) * block_x_length
    y_full = (1 + interval_ratio_y) * block_y_length

    ax.axis(
            xmin=0,
            xmax=(
                cols + cols * interval_ratio_x - interval_ratio_x
            )
            * block_x_length,
            ymin=0,
            ymax=figure_height,
        )
    for ix in range(rows):
        for iy in range(cols):
            x = x_full * iy
            y = y_full * ix
            category = hmap[ix, iy]
            color = colormap.get(category, None)
            if color:
                ax.add_artist(Rectangle(xy=(x, y), width=block_x_length, height=block_y_length, color=color))

    ax.set_xlabel(None)
    ax.set_ylabel(None)

    ax.set_xticks([])
    ax.set_yticks([])
    if facecolor:
        ax.set_facecolor( facecolor )
    
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
                       cm=None,
                       do_plot_prec_recall=True,
                       colormap=dict(enumerate([None, "orange", "darkgreen", "red", "lightgrey"])),
                       ):
    fig = plt.figure(constrained_layout=False, )
    ymarg = 3.0
    if do_plot_prec_recall:
        gs = GridSpec(4, 4, figure=fig)
        if cm is None:
            _, fn, tp, fp, tn = [np.count_nonzero(hmap == x) for x in range(len(colormap))]
        else:
            tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp)
        recall = tp / (fn + tp)

        build_right_figure(hmap, fig, colormap,
                           "prec",
                           gs,
                           {1: None, 4: None}, 0, precision, ymarg)

        build_right_figure(hmap, fig, colormap,
                           "recall",
                           gs,
                           {3: None, 4: None}, 2, recall, ymarg)
        params = [gs[0:4, 0:3]]
    else:
        params = []
    axbig = fig.add_subplot(*params)
    axbig.margins(y=ymarg)
    subplot_waffle_matrix(axbig, hmap, colormap=colormap, )

    transparent_color = (1., 1., 1., 0.)
    fig.legend(
                handles=[Patch(color=colormap[1]),
                         Patch(color=colormap[2]),
                         Patch(color=transparent_color),
                         Patch(color=colormap[4]),
                         Patch(color=colormap[3]),
                         Patch(color=transparent_color),
                         ],
                labels=["false negative", "true positive", "relevant", "true negative", "false positive",  "irrelevant"],
                loc='lower left',
                bbox_to_anchor= (0.07, -0.03),
                edgecolor='k',
                ncol=2,
                frameon=False
              )
    return fig


def build_right_figure(hmap, fig, colormap, desc, gs, override_dict, pos, value, ymarg, small_interval=1.0):
    ax1 = fig.add_subplot(gs[0 + pos, 3])
    subplot_waffle_matrix(ax1, hmap,
                          colormap={**colormap, 1: None, 3: None, 4: None},
                          interval_ratio_x=small_interval,
                          interval_ratio_y=small_interval, )
    ax1.margins(y=ymarg)
    ax2 = fig.add_subplot(gs[1 + pos, 3])
    add_fraction_bar(ax2)
    add_value(ax=ax2, value=value, desc=desc)
    subplot_waffle_matrix(ax2, hmap,
                          colormap={**colormap, **override_dict},
                          interval_ratio_x=small_interval,
                          interval_ratio_y=small_interval, )
    ax2.margins(y=ymarg)

