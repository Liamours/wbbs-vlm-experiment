from training.train import _steps_per_epoch


def test_steps_per_epoch_matches_qwen3vl_r3_multitask_run():
    # R3 multitask export: 30,862 training rows. Qwen3-VL's completed baseline
    # (runs/r3-unified-qwen3vl-2b-2epoch-5070) used effective batch 8 and
    # measured 3,858 optimizer steps per epoch, 7,716 total for 2 epochs
    # (training_progress.json). PaliGemma's epoch-boundary checkpointing must
    # land on the same step numbers for the two runs to be a fair comparison.
    steps_per_epoch = _steps_per_epoch(example_count=30862, gradient_accumulation=8)
    assert steps_per_epoch == 3858
    assert steps_per_epoch * 2 == 7716


def test_steps_per_epoch_rounds_up_partial_batch():
    assert _steps_per_epoch(example_count=17, gradient_accumulation=8) == 3


def test_steps_per_epoch_exact_division():
    assert _steps_per_epoch(example_count=16, gradient_accumulation=8) == 2
