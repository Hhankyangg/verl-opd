def dummy_compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    return 0.0


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    return dummy_compute_score(
        data_source,
        solution_str,
        ground_truth,
        extra_info=extra_info,
        **kwargs,
    )
