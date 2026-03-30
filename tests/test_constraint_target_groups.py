from types import SimpleNamespace

import numpy as np

from VMuscle.constraints import (
    ConstraintBuilderMixin,
    _DEPRECATED_CONSTRAINT_TYPE_WARNINGS,
    constraint_alias,
    warn_deprecated_constraint_type,
)


class _DummyBuilder(ConstraintBuilderMixin):
    def __init__(self):
        self.pos0_np = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        self.tet_np = np.array([[0, 1, 2, 3]], dtype=np.int32)
        self.v_fiber_np = None
        self.geo = SimpleNamespace(mask=[1.0, 1.0, 1.0, 1.0])
        self.bone_pos = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.2, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.2, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        self.bone_muscle_ids = {
            "proximal": np.array([0, 1], dtype=np.int32),
            "mid": np.array([2, 3], dtype=np.int32),
            "distal": np.array([4, 5], dtype=np.int32),
        }
        self.bone_geo = None

    def load_bone_geo(self, target_path):
        return self.bone_geo, self.bone_pos


def test_attach_constraints_can_filter_sources_and_targets_by_group():
    builder = _DummyBuilder()

    constraints = builder.create_attach_constraints(
        {
            "mask_name": "mask",
            "mask_threshold": 0.5,
            "target_path": "unused",
            "source_nearest_group": ["proximal", "mid"],
            "target_group": "mid",
            "stiffness": 1.0,
        }
    )

    assert [constraint["pts"][0] for constraint in constraints] == [0, 1, 2]
    assert all(constraint["pts"][2] in (2, 3) for constraint in constraints)


def test_attachnormal_warns_once_and_maps_to_distanceline(capsys):
    _DEPRECATED_CONSTRAINT_TYPE_WARNINGS.clear()

    warn_deprecated_constraint_type("attachnormal")
    warn_deprecated_constraint_type("attachnormal")
    captured = capsys.readouterr()

    assert captured.out.count("deprecated") == 1
    assert constraint_alias("attachnormal") == "distanceline"
