from preprocess.tokens import format_bbx, format_grounded_target, format_view_bbx


def test_bbx_target_keeps_original_pixel_coordinates() -> None:
    assert format_bbx([1, 2.5, 30, 40]) == "<BBX>1,2.5,30,40</BBX>"


def test_grounded_target_is_structured() -> None:
    assert format_grounded_target("right chest", "abnormal", [1, 2, 30, 40]) == "<REG>right_chest</REG><CLS>abnormal</CLS><BBX>1,2,30,40</BBX>"


def test_paired_boxes_keep_view_identity() -> None:
    targets = [{"view": "ANT", "bbox": [1, 2, 30, 40]}, {"view": "POST", "bbox": [2, 3, 31, 41]}]
    assert format_view_bbx(targets) == "<ANT><BBX>1,2,30,40</BBX></ANT><PST><BBX>2,3,31,41</BBX></PST>"
    assert format_grounded_target("right chest", "abnormal", targets) == "<REG>right_chest</REG><CLS>abnormal</CLS><ANT><BBX>1,2,30,40</BBX></ANT><PST><BBX>2,3,31,41</BBX></PST>"
