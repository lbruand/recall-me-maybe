#!/usr/bin/python
# -*-coding: utf-8 -*-

import unittest
import recallme.recallme
import matplotlib.pyplot as plt
import random
import numpy as np
from pathlib import Path

from recallme.recallme import update_boxes_using_distance_from_center


def test_build_waffle_matrix(cols, rows, cm, test_cm=True):
    hmap = recallme.recallme.build_waffle_matrix_from_confusion_matrix((rows, cols), cm)
    fig = recallme.recallme.plot_waffle_matrix(hmap, cm=cm if test_cm else None)
    tn, fp, fn, tp = cm.ravel()
    values, counts = np.unique(hmap, return_counts=True)
    result_counts = np.zeros(len(cm.ravel()), dtype=int)
    for v, c in zip(values, counts):        
        result_counts[int(v)-1] = c
        
    ordered_vector = np.array([fn, tp, fp, tn])
    normalized_vect = ordered_vector / sum(ordered_vector) * cols * rows

    print(normalized_vect)
    print(result_counts)
    print(f"counts sum = {sum(result_counts)} - cols x rows = { cols * rows} ")
    max_error = np.max(np.abs(normalized_vect - result_counts))
    assert max_error <= 1.01, f" max_error = {max_error}"
    return fig


class TestRecallMeMaybe(unittest.TestCase):

    def test_distance_field(self):
        res = recallme.recallme.build_distance_field( (5, 7), (2, 3))
        print(res)
        self.assertEqual(0.0, res[ 2, 3])

    def test_distance_field_with_cat(self):

        size = (5, 7)
        catstart = 1
        catend = 4
        hmap = (catend - 1) * (np.repeat(np.arange(size[0]), size[1]).reshape(size) < 3) + catstart
        nb_boxes = 7
        center = (2, 3)

        result = update_boxes_using_distance_from_center(hmap, catend, 3, center, 7)
        self.assertEqual(nb_boxes, np.sum(result == 3))
        result2 = update_boxes_using_distance_from_center(result, catstart, 2, center, 9)
        self.assertEqual(7, np.sum(result2 == 3))
        self.assertEqual(9, np.sum(result2 == 2))
        print(result)

    def test_cascade_rounding(self):
        rng = np.random.default_rng(seed=42)
        random.seed(3)
        for i in range(60):
            input = rng.integers(low=0, high=100, size=(4))
            size = (random.randint(5, 5), random.randint(5, 25))

            res = recallme.recallme.cascade_rounding(input, size)
            total_boxes = size[0] * size[1]
            self.assertEqual(total_boxes, np.sum(res))
            self.assertTrue( (max(np.abs((total_boxes / sum(input)) * input -res))) <= 1.0 )

    def test_recall(self):
        Path("output").mkdir(parents=True, exist_ok=True)
        random.seed(3)
        rng = np.random.default_rng(seed=42)
        test_cm = random.choice([True, False])
        for i in range(60):
            print(f"iter = {i}")
            cols = random.randint(5, 15)
            rows = random.randint(5, 15)
            print( (rows, cols))
            cm = rng.integers(low=0, high=15, size=(2, 2))
            print(cm)
            fig = test_build_waffle_matrix(cols, rows, cm, test_cm=test_cm)
            plt.savefig(f"output/output_{i:001}.svg", bbox_inches='tight')


if __name__ == "__main__":
    unittest.main()
