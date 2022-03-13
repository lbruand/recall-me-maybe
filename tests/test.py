#!/usr/bin/python
# -*-coding: utf-8 -*-

import os
import unittest
import recallme.recallme
import matplotlib.pyplot as plt
import random
import numpy as np
from pathlib import Path

def test_build_waffle_matrix(cols, rows, cm):
    hmap = recallme.recallme.build_waffle_matrix( (rows, cols), cm)
    fig = recallme.recallme.plot_waffle_matrix(hmap, cm)
    tn, fp, fn, tp = cm.ravel()
    values, counts = np.unique(hmap, return_counts=True)
    result_counts = [0] * 4
    for v, c in zip(values, counts):        
        result_counts[int(v)-1] = c
        
    ordered_vector = [fn, tp, fp, tn]
    normalized_vect =  ordered_vector / sum(ordered_vector) * cols * rows

    print(normalized_vect)
    print(result_counts)
    print(f"counts sum = {sum(result_counts)} - cols x rows = { cols * rows} ")
    max_error = np.max(np.abs(normalized_vect - result_counts))
    assert max_error <= 4.0, f" max_error = {max_error}"
    return fig

class TestRecallMeMaybe(unittest.TestCase):
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
		for i in range(60):
			print(f"iter = {i}")
			cols = random.randint(5, 15)
			rows = random.randint(5, 15)
			print( (rows, cols))
			cm = rng.integers(low=0, high=15, size=(2, 2))
			print(cm)
			fig = test_build_waffle_matrix(cols, rows, cm)
			plt.savefig(f"output/output_{i:001}.svg", bbox_inches='tight')


if __name__ == "__main__":
	unittest.main()