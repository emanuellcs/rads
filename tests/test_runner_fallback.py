from scripts.run_arc_agi_2_ttt import _fallback_attempts_for_task


def test_fallback_attempts_emit_valid_two_attempts_per_test_item():
    task = {
        "test": [
            {"input": [[1, 2], [3, 4]]},
            {"input": [[0]]},
        ]
    }

    fallback = _fallback_attempts_for_task(task)

    assert fallback == [
        {"attempt_1": [[1, 2], [3, 4]], "attempt_2": [[1, 2], [3, 4]]},
        {"attempt_1": [[0]], "attempt_2": [[0]]},
    ]
