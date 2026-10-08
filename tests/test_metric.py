from metric.grounding import iou
from metric.evaluate import evaluate_predictions
from metric.vqa import exact_match
from postprocess.paligemma import export_paligemma_with_rejections, normalize_box_1024


def test_metrics_and_export_coordinates() -> None:
    assert exact_match("Yes.", ["no", "yes"]) == 1.0
    assert iou([0, 0, 10, 10], [5, 5, 15, 15]) == 25 / 175
    assert normalize_box_1024([0, 0, 100, 200], [100, 200]) == [0, 0, 1023, 1023]


def test_paired_export_and_evaluation() -> None:
    row = {
        "task_id": "g1",
        "split": "test",
        "images": [
            {"view": "ANT", "image": "ant.jpg", "image_size": [100, 100]},
            {"view": "POST", "image": "post.jpg", "image_size": [100, 100]},
        ],
        "query": "Locate the left chest.",
        "label": "left chest",
        "targets": [{"view": "POST", "bbox": [10, 10, 20, 20]}],
    }
    exported, rejected = export_paligemma_with_rejections([row], "grounding", split="test")
    assert not rejected
    assert exported[0]["box_1024"] == normalize_box_1024([110, 10, 120, 20], [200, 100])
    prediction = "<loc0102><loc0563><loc0205><loc0614> left chest"
    report = evaluate_predictions(exported, [{"task_id": "g1", "prediction": prediction}], "grounding")
    assert report["valid_boxes"] == 1
    assert report["mean_iou"] == 1.0


def test_paired_multi_target_export_is_explicitly_rejected() -> None:
    row = {
        "task_id": "g2",
        "images": [{"view": "ANT", "image": "ant.jpg", "image_size": [10, 10]}, {"view": "POST", "image": "post.jpg", "image_size": [10, 10]}],
        "query": "Locate the chest.",
        "label": "chest",
        "targets": [{"view": "ANT", "bbox": [1, 1, 2, 2]}, {"view": "POST", "bbox": [1, 1, 2, 2]}],
    }
    exported, rejected = export_paligemma_with_rejections([row], "grounding")
    assert not exported
    assert rejected == [{"task_id": "g2", "reason": "multi_target_not_supported_by_paligemma_baseline", "split": None}]
