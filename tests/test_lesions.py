from preprocess.lesions import build_lesion_task_manifests


def test_lesion_tasks_use_only_unique_paired_hotspots() -> None:
    canonical = [{"record_id": "region-ant", "patient_id": "1", "view": "ANT", "target": {"name": "left chest"}}]
    pairs = [
        {
            "evidence_id": "wbbs:pair:1:left_chest",
            "patient_id": "1",
            "region": "left chest",
            "split": "test",
            "images": [
                {"view": "ANT", "image": "ant.jpg", "image_size": [100, 100], "bbox": [1, 1, 20, 20]},
                {"view": "POST", "image": "post.jpg", "image_size": [100, 100], "bbox": [1, 1, 20, 20]},
            ],
            "source_labels": {"ANT": "1", "POST": "0"},
        }
    ]
    metastases = [
        {"metastasis_id": "m1", "assignment": "unique_region", "candidate_record_ids": ["region-ant"], "view": "ANT", "bbox": [3, 3, 8, 8], "label": "Abnormal", "source_id": "1:0"},
        {"metastasis_id": "m2", "assignment": "ambiguous_regions", "candidate_record_ids": ["region-ant"], "view": "ANT", "bbox": [3, 3, 8, 8], "label": "Abnormal"},
    ]

    manifests, rejected, summary = build_lesion_task_manifests(canonical, pairs, metastases)

    assert summary["eligible_lesions"] == 1
    assert rejected == [{"metastasis_id": "m2", "reason": "ambiguous_regions"}]
    assert manifests["lesion_vqa"][0]["answer_label"] == "yes"
    assert manifests["lesion_grounding"][0]["targets"] == [{"view": "ANT", "bbox": [3, 3, 8, 8]}]
    assert manifests["lesion_grounded_vqa"][0]["target"] == "<REG>left_chest</REG><CLS>abnormal</CLS><ANT><BBX>3,3,8,8</BBX></ANT>"
