import os
import sys
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from CharacterData.mh_skeleton import Skeleton
from CharacterData.obj_loader import load_obj


class SkeletonWeightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        makehuman = os.path.join(ROOT, "CharacterData", "makehuman", "makehuman")
        cls.mesh = load_obj(os.path.join(makehuman, "data", "3dobjs", "base.obj"))
        cls.skeleton = Skeleton()
        cls.skeleton.fromFile(os.path.join(makehuman, "data", "rigs", "game_engine.mhskel"), cls.mesh)

    def test_weights_are_unique_normalized_and_limited(self):
        vertex_count = len(self.mesh.vertices)
        weight_sum = np.zeros(vertex_count, dtype=np.float64)
        influence_count = np.zeros(vertex_count, dtype=np.uint32)

        for indices, weights in self.skeleton.vertexWeights.data.values():
            self.assertEqual(len(indices), len(np.unique(indices)))
            weight_sum[indices] += weights
            influence_count[indices] += 1

        np.testing.assert_allclose(weight_sum, 1.0, rtol=0.0, atol=1e-6)
        self.assertGreaterEqual(int(influence_count.min()), 1)
        self.assertLessEqual(int(influence_count.max()), 4)
        np.testing.assert_array_equal(influence_count, self.skeleton.vertexWeights._wCounts)

    def test_weight_canonicalization_is_idempotent(self):
        before = [
            (bone_name, indices.copy(), weights.copy())
            for bone_name, (indices, weights) in self.skeleton.vertexWeights.data.items()
        ]

        self.skeleton._canonicalize_weights(self.mesh)

        after = list(self.skeleton.vertexWeights.data.items())
        self.assertEqual(len(before), len(after))
        for (before_name, before_indices, before_weights), (after_name, (after_indices, after_weights)) in zip(before, after):
            self.assertEqual(before_name, after_name)
            np.testing.assert_array_equal(before_indices, after_indices)
            np.testing.assert_array_equal(before_weights, after_weights)


if __name__ == "__main__":
    unittest.main()
