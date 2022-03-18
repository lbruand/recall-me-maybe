from typing import Dict, Tuple, NoReturn, Generator, Optional, Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle
from matplotlib.gridspec import GridSpec
import numpy as np
import math
import itertools


def cascade_rounding(vector:  np.array, size: Tuple[int, int]) -> np.array:
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


def update_boxes(heatmap: np.array,
                 gene: Generator[Tuple[int, int], None, None],
                 nb_boxes: int,
                 from_cat: int,
                 to_cat: int) -> NoReturn:
    to_update = itertools.islice(gene, nb_boxes)
    for xy in to_update:
        ix, iy = xy
        assert(heatmap[ix, iy] == from_cat)
        heatmap[ix, iy] = to_cat


def distance_norm_l2(x: np.array, y: np.array) -> np.array:
    return np.square(x) + np.square(y)


def distance_norm_l1(x: np.array, y: np.array) -> np.array:
    return np.abs(x) + np.abs(y)


def distance_norm_linf(x: np.array, y: np.array) -> np.array:
    return np.maximum(np.abs(x), np.abs(y))


def build_distance_field(size: Tuple[int, int], center: Tuple[int, int], norm: Callable[[np.array, np.array], np.array] = distance_norm_l2):
    x, y = size
    centerx, centery = center
    return norm(build_arange_field(x, y, centerx),  np.transpose(build_arange_field(y, x, centery)))


def build_arange_field(x, y, center):
    return np.repeat(np.arange(x), y).reshape((x, y)) - np.ones( (x, y)) * center


def build_snake_matrix_from_value_counts(size: Tuple[int, int],
                                         value_counts: Dict[int, int],
                                         order='C'
                                         ) -> np.array:
    rows, cols = size
    total = sum(value_counts.values())
    total_size = rows * cols
    assert total_size >= total
    arrays = [np.full((count), fill_value=ix, dtype=int) for ix, count in value_counts.items()]
    missing = total_size - total
    if missing > 0:
        arrays += [np.full(missing, 0, dtype=int)]
    linarray = np.concatenate(arrays)
    return linarray.reshape(size, order=order)


def build_waffle_matrix_from_confusion_matrix(size: Tuple[int, int],
                                              cm: np.array,
                                              norm="Linf") -> np.array:
    assert cm.shape == (2, 2)
    rows, cols = size
    tn, fp, fn, tp = cm.ravel()

    fn_boxes, tp_boxes, fp_boxes, tn_boxes = cascade_rounding(np.array([fn, tp, fp, tn]), size)

    hmap = np.where(np.arange(rows * cols) < fn_boxes + tp_boxes, 1, 4).reshape(size, order='F')

    center = ((int(round(rows / 2))),
              ((fn_boxes + tp_boxes) // rows))
    normfunc_dict = {
        "Linf": distance_norm_linf,
        "L2": distance_norm_l2,
        "L1": distance_norm_l1,
    }
    normfunc = normfunc_dict.get(norm, distance_norm_linf)
    if tp_boxes > 0:
        result = update_boxes_using_distance_from_center(hmap, 1, 2, center, tp_boxes, norm=normfunc)
    else:
        result = hmap
    if fp_boxes > 0:
        result2 = update_boxes_using_distance_from_center(result, 4, 3, center, fp_boxes, norm=normfunc)
    else:
        result2 = result
    return result2


def subplot_waffle_matrix(ax: Axes,
                          hmap: np.array,
                          colormap: Dict[int, Optional[str]],
                          interval_ratio_x=0.3,
                          interval_ratio_y=0.3,
                          block_aspect_ratio=1.0,
                          facecolor=(1., 1., 1., 0.0),
                          adjust_height=1.0,
                          ) -> Axes:
    rows, cols = hmap.shape
    figure_height = 1.0
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
            ymax=adjust_height * figure_height,
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

    return ax


def add_fraction_bar(ax: Axes):
    p = plt.Rectangle((-0.1, 1.1), 1.2, 0.01, fill=False)
    p.set_transform(ax.transAxes)
    p.set_clip_on(False)
    ax.add_patch(p)


def add_value(ax: Axes, value: float, desc: str):
    ax.text (1.2, 1.05, 
             f' = { 100.0 * value :.1f}% = {desc}', 
             transform=ax.transAxes,
             fontsize=15)    


def plot_waffle_matrix(hmap: np.array,
                       cm: Optional[np.array] = None,
                       do_plot_prec_recall=True,
                       colormap: Dict[int, Optional[str]] = None,
                       ) -> Figure:
    if colormap is None:
        colormap = dict(enumerate([None, "orange", "darkgreen", "red", "lightgrey"]))
    fig = plt.figure(constrained_layout=False, )
    ymarg = 3.0
    if do_plot_prec_recall:
        gs = GridSpec(4, 10, figure=fig)
        if cm is None:
            _, fn, tp, fp, tn = [np.count_nonzero(hmap == x) for x in range(len(colormap))]
        else:
            tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp)
        recall = tp / (fn + tp)

        build_right_figure(hmap, fig, colormap, "prec", precision, gs, 1, 0, ymarg)

        build_right_figure(hmap, fig, colormap, "recall", recall, gs, 3, 2, ymarg)
        params = [gs[0:4, 0:6]]
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
                bbox_to_anchor= (0.07, -0.05),
                edgecolor='k',
                ncol=2,
                frameon=False
              )
    return fig


def build_right_figure(hmap, fig, colormap, desc, value, gs, override_value, pos, ymarg, small_interval=1.0):
    override_dict = {override_value: None, 4: None}
    count_value = 1 if override_value == 3 else 3
    recap_frac_ax = fig.add_subplot(gs[0 + pos: 2 + pos, 6])
    value_counts = { 2: np.sum(hmap == 2), count_value: np.sum(hmap == count_value),   }
    total = sum(value_counts.values())
    nb_cols = min(hmap.shape) // 2
    snake_map = build_snake_matrix_from_value_counts( (int(math.ceil(total / nb_cols)), nb_cols, ), value_counts=value_counts)
    color_map = {**colormap, **override_dict}
    subplot_waffle_matrix(recap_frac_ax, snake_map,
                          colormap=color_map,
                          interval_ratio_x=small_interval,
                          interval_ratio_y=small_interval,
                          adjust_height=1.4)
    recap_frac_ax.set_title(f"{desc}\n{ 100.0 * value :.1f}%", y=0.7)
    recap_frac_ax.margins(x=1., y=ymarg)

    top_frac_ax = fig.add_subplot(gs[0 + pos, 7:9])
    subplot_waffle_matrix(top_frac_ax, hmap,
                          colormap={**colormap, 1: None, 3: None, 4: None},
                          interval_ratio_x=small_interval,
                          interval_ratio_y=small_interval, )
    top_frac_ax.margins(y=ymarg)

    bottom_frac_ax = fig.add_subplot(gs[1 + pos, 7:9])
    add_fraction_bar(bottom_frac_ax)
    #add_value(ax=bottom_frac_ax, value=value, desc=desc)
    subplot_waffle_matrix(bottom_frac_ax, hmap,
                          colormap={**colormap, **override_dict},
                          interval_ratio_x=small_interval,
                          interval_ratio_y=small_interval, )
    bottom_frac_ax.margins(y=ymarg)


def update_boxes_using_distance_from_center(hmap: np.array,
                                            fromcat: int,
                                            tocat: int,
                                            center: Tuple[int, int],
                                            nb_boxes: int,
                                            norm: Callable[[np.array, np.array], np.array] = distance_norm_linf):
    size = np.shape(hmap)
    distance_field = build_distance_field(size, center, norm=norm)
    d = np.where(hmap == fromcat, distance_field, np.nan * np.ones(size))
    ravel = d.ravel()
    aravel = hmap.ravel()
    sorted = np.argsort(ravel)
    for i in range(nb_boxes):
        aravel[sorted[i]] = tocat
    return aravel.reshape(size)