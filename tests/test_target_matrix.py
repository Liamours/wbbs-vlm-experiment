import json

from analysis.r3_supplementary import TARGET_COLUMNS, target_matrix


def _row(task_id: str, patient: str, bbox: list[int]) -> dict:
    return {"task_id": task_id, "task": "grounding", "region_level": "metastasis", "patient_id": patient, "targets": [{"view": "ANT", "bbox": bbox}]}


def _prediction(bbox: list[int]) -> str:
    return json.dumps({"task": "grounding", "region": "head", "boxes": [{"view": "ANT", "bbox": bbox}]})


def test_target_matrix_scores_a_hit_a_near_miss_and_a_missing_box():
    manifest = [_row("a", "p1", [10, 10, 20, 20]), _row("b", "p2", [10, 10, 20, 20]), _row("c", "p2", [10, 10, 20, 20])]
    prediction = {"a": _prediction([10, 10, 20, 20]), "b": _prediction([25, 10, 35, 20]), "c": "not json"}
    matrix = target_matrix(manifest, prediction, "grounding", "metastasis", {"p1": 0, "p2": 1})
    column = {name: matrix[:, i].sum() for i, name in enumerate(TARGET_COLUMNS)}
    assert column["targets"] == 3
    assert column["iou_at_0_5"] == 1 and column["center_inside"] == 1
    assert target_matrix(manifest, prediction, "grounding", "bone_region", {"p1": 0, "p2": 1}).sum() == 0
