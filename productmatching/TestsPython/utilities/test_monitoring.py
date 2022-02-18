import sys
import asyncio
import json
from datetime import datetime

from unittest.mock import patch
import pytest

sys.path.append("/app/SourcesPython")
sys.path.append("/app/resources/candy/src")

from auto_monitoring.main import Evaluator, calculate_coverage, alert_thresholds


def _get_matching_results_unknown():
    candidates_unknown = [
        {
            "id": 100,
            "name": "Moje mama",
            "category_id": 1962
        },
        {
            "id": 21,
            "name": "Tvoje mama",
            "category_id": 1962
        },
        {
            "id": 101,
            "name": "Jeho mama",
            "category_id": 1963
        },
    ]
    mess1 = {
        "item": {
            "id": 10,
            "match_name": "Moje baba"
        },
        "candidates": candidates_unknown,
        "model_info": {
            "name": "model",
            "version": "666"
        },
        "comparisons": [
            {
                "id": "100",
                "details": "prediction=0.71,...",
                "decision": "yes"
            },
            {
                "id": "21",
                "details": "prediction=0.89,...",
                "decision": "yes"
            },
            {
                "id": "101",
                "details": "prediction=0.36,...",
                "decision": "no"
            },
        ],
        "final_decision": "unknown",
        "final_candidate": "",
        "possible_categories": "1962"
    }
    mess2 = {
        "item": {
            "id": 11,
            "match_name": "Tvoje baba"
        },
        "candidates": candidates_unknown,
        "model_info": {
            "name": "model",
            "version": "666"
        },
        "comparisons": [
            {
                "id": "100",
                "details": "prediction=0.89,...",
                "decision": "yes"
            },
            {
                "id": "21",
                "details": "Name match",
                "decision": "yes"
            },
            {
                "id": "101",
                "details": "prediction=0.31,...",
                "decision": "no"
            },
        ],
        "final_decision": "unknown",
        "final_candidate": "",
        "possible_categories": "1962"
    }
    mess3 = {
        "item": {
            "id": 12,
            "match_name": "Jeho baba"
        },
        "candidates": candidates_unknown,
        "model_info": {
            "name": "model",
            "version": "666"
        },
        "comparisons": [
            {
                "id": "100",
                "details": "prediction=0.11,...",
                "decision": "no"
            },
            {
                "id": "21",
                "details": "prediction=0.52,...",
                "decision": "unknown"
            },
            {
                "id": "101",
                "details": "prediction=0.49,...",
                "decision": "unknown"
            },
        ],
        "final_decision": "unknown",
        "final_candidate": "",
        "possible_categories": "1962"
    }
    mess4 = {
        "item": {
            "id": 13,
            "match_name": "Cizi baba"
        },
        "candidates": [{"id": 103, "name": "Mama baba", "category_id": 42}],
        "model_info": {
            "name": "model",
            "version": "666"
        },
        "comparisons": [{"id": "103", "details": "prediction=0.50,...", "decision": "unknown"}],
        "final_decision": "unknown",
        "final_candidate": "",
        "possible_categories": "1962"
    }
    times = [datetime(666, 6, 1), datetime(666, 6, 2), datetime(666, 6, 3), datetime(666, 6, 5)]
    return (
        (10, json.dumps(mess1), times[0]),
        (11, json.dumps(mess2), times[1]),
        (12, json.dumps(mess3), times[2]),
        (13, json.dumps(mess4), times[3]),
    )


def _get_matching_results_matched():
    candidates_matched = [
        {
            "id": 104,
            "name": "Muj deda",
            "category_id": 2337
        },
        {
            "id": 105,
            "name": "Tvuj deda",
            "category_id": 1962
        },
        {
            "id": 106,
            "name": "Jeho deda",
            "category_id": 2337
        },
    ]
    mess1 = {
        "item": {
            "id": 15,
            "match_name": "Muj tata"
        },
        "candidates": candidates_matched,
        "model_info": {
            "name": "model",
            "version": "666"
        },
        "comparisons": [
            {
                "id": "104",
                "details": "prediction=0.51,...",
                "decision": "unknown"
            },
            {
                "id": "105",
                "details": "prediction=0.89,...",
                "decision": "yes"
            },
            {
                "id": "106",
                "details": "prediction=0.36,...",
                "decision": "no"
            },
        ],
        "final_decision": "yes",
        "final_candidate": 105,
        "possible_categories": "2337"
    }
    mess2 = {
        "item": {
            "id": 16,
            "match_name": "Tvuj tata"
        },
        "candidates": candidates_matched,
        "model_info": {
            "name": "model",
            "version": "666"
        },
        "comparisons": [
            {
                "id": "104",
                "details": "prediction=0.19,...",
                "decision": "no"
            },
            {
                "id": "105",
                "details": "prediction=0.31,...",
                "decision": "no"
            },
            {
                "id": "106",
                "details": "Ean match",
                "decision": "yes"
            },
        ],
        "final_decision": "yes",
        "final_candidate": 106,
        "possible_categories": "2337"
    }
    mess3 = {
        "item": {
            "id": 17,
            "match_name": "Jeho tata"
        },
        "candidates": candidates_matched,
        "model_info": {
            "name": "model",
            "version": "666"
        },
        "comparisons": [
            {
                "id": "104",
                "details": "prediction=0.99,...",
                "decision": "yes"
            },
            {
                "id": "105",
                "details": "prediction=0.52,...",
                "decision": "unknown"
            },
            {
                "id": "106",
                "details": "prediction=0.49,...",
                "decision": "unknown"
            },
        ],
        "final_decision": "yes",
        "final_candidate": 104,
        "possible_categories": "2337"
    }
    mess4 = {
        "item": {
            "id": 18,
            "match_name": "Cizi tata"
        },
        "candidates": candidates_matched,
        "model_info": {
            "name": "model",
            "version": "666"
        },
        "comparisons": [
            {
                "id": "104",
                "details": "Name match",
                "decision": "yes"
            },
            {
                "id": "105",
                "details": "prediction=0.25,...",
                "decision": "no"
            },
            {
                "id": "106",
                "details": "prediction=0.44,...",
                "decision": "unknown"
            },
        ],
        "final_decision": "yes",
        "final_candidate": 104,
        "possible_categories": "2337"
    }
    times = [datetime(666, 6, 1), datetime(666, 6, 2), datetime(666, 6, 3), datetime(666, 6, 5)]
    return (
        (15, json.dumps(mess1), times[0]),
        (16, json.dumps(mess2), times[1]),
        (17, json.dumps(mess3), times[2]),
        (18, json.dumps(mess4), times[3]),
    )


def _get_matching_results_new_candidate():
    candidates_np = [
        {
            "id": 108,
            "name": "Tvuj strejda",
            "category_id": 1962
        },
        {
            "id": 109,
            "name": "Jeho strejda",
            "category_id": 2337
        },
        {
            "id": 110,
            "name": "Muj strejda",
            "category_id": 2337
        },
    ]
    mess1 = {
        "item": {
            "id": 20,
            "match_name": "Moje teta"
        },
        "candidates": candidates_np,
        "model_info": {
            "name": "model",
            "version": "666"
        },
        "comparisons": [
            {
                "id": "108",
                "details": "prediction=0.01,...",
                "decision": "no"
            },
            {
                "id": "109",
                "details": "prediction=0.20,...",
                "decision": "no"
            },
            {
                "id": "110",
                "details": "prediction=0.36,...",
                "decision": "no"
            },
        ],
        "final_decision": "no",
        "final_candidate": "",
        "possible_categories": "2337"
    }
    mess2 = {
        "item": {
            "id": 21,
            "match_name": "Tvoje teta"
        },
        "candidates": candidates_np,
        "model_info": {
            "name": "model",
            "version": "666"
        },
        "comparisons": [
            {
                "id": "108",
                "details": "prediction=0.09,...",
                "decision": "no"
            },
            {
                "id": "109",
                "details": "Name match",
                "decision": "no"
            },
            {
                "id": "110",
                "details": "prediction=0.31,...",
                "decision": "no"
            },
        ],
        "final_decision": "no",
        "final_candidate": "",
        "possible_categories": "2337"
    }
    mess3 = {
        "item": {
            "id": 12,
            "match_name": "Jeho baba"
        },
        "candidates": candidates_np,
        "model_info": {
            "name": "model",
            "version": "666"
        },
        "comparisons": [
            {
                "id": "108",
                "details": "prediction=0.09,...",
                "decision": "no"
            },
            {
                "id": "109",
                "details": "Name match",
                "decision": "no"
            },
            {
                "id": "110",
                "details": "prediction=0.31,...",
                "decision": "no"
            },
        ],
        "final_decision": "no",
        "final_candidate": "",
        "possible_categories": "2337"
    }
    times = [datetime(666, 6, 1), datetime(666, 6, 3), datetime(666, 6, 5)]
    return (
        (20, json.dumps(mess1), times[0]),
        (21, json.dumps(mess2), times[1]),
        (22, json.dumps(mess3), times[2]),
    )


@pytest.mark.asyncio
async def test_eval_offers_unknown():
    with patch('auto_monitoring.main.Evaluator.offer_download') as mocked_offers, \
        patch('auto_monitoring.main.Evaluator.get_product_url') as mocked_get_product_url, \
        patch('auto_monitoring.main.Evaluator.product_download') as mocked_products:

        mocked_offers.side_effect = [*_get_offers_data([10, 11, 12, 13])]
        # 00 stands for candidate_data used only for candidate_url creation
        mocked_products.side_effect = [*_get_products_data([100, 00, 00, 102, 00, 00, 101, 00])]
        mocked_get_product_url.side_effect = [""] * 6

        evaluator = Evaluator("")

        evaluated_unknown_rows = await asyncio.gather(*[
            evaluator.eval_offer_unknown(row) for row in _get_matching_results_unknown()
        ])

        assert_unknown_eval_results(evaluated_unknown_rows)

        _ = evaluator.calc_log_metrics_unknown(evaluated_unknown_rows)


@pytest.mark.asyncio
async def test_eval_offers_matched():
    with patch('auto_monitoring.main.Evaluator.offer_download') as mocked_offers, \
        patch('auto_monitoring.main.Evaluator.get_full_product_merge_history') as mocked_get_merge_history, \
        patch('auto_monitoring.main.Evaluator.get_product_url') as mocked_get_product_url, \
        patch('auto_monitoring.main.Evaluator.product_download') as mocked_products:

        mocked_offers.side_effect = [*_get_offers_data([15, 16, 17, 18])]
        mocked_get_merge_history.side_effect = [[] for i in range(4)]
        # 00 stands for candidate_data used only for candidate_url creation and for one unpaired product
        mocked_products.side_effect = [*_get_products_data([105, 106, 105, 00, 00, 00, 00])]
        mocked_get_product_url.side_effect = [""] * 5

        evaluator = Evaluator("")

        evaluated_match_rows = await asyncio.gather(*[
            evaluator.eval_offer_matched(row) for row in _get_matching_results_matched()
        ])

        assert_matched_eval_results(evaluated_match_rows)

        alerts = evaluator.calc_log_metrics_matched(evaluated_match_rows, {"name": "model", "version": "666"})
        current_precision_thresh = alert_thresholds["INCORRECT_MATCHED_RATIO_THRESHOLD"]
        current_unpaired_thresh = 1 - alert_thresholds["UNPAIRED_MATCHED_RATIO_THRESHOLD"]
        assert alerts == {
            "critical": [{
                "model_info": {"name": "model", "version": "666"},
                "message": f"model_666 - Ratio of correct matches among currently paired offers is under {current_precision_thresh}, value: 0.5"
            }],
            "warning": [{
                "model_info": {"name": "model", "version": "666"},
                "message": f"model_666 - Ratio of paired matched offers is under {current_unpaired_thresh}, value: 0.75",
            }],
        }


@pytest.mark.asyncio
async def test_eval_offers_new_candidate():
    with patch('auto_monitoring.main.Evaluator.offer_download') as mocked_offers, \
        patch('auto_monitoring.main.Evaluator.get_product_url') as mocked_get_product_url, \
        patch('auto_monitoring.main.Evaluator.product_download') as mocked_products:

        mocked_offers.side_effect = [*_get_offers_data([20, 21, 22])]
        mocked_products.side_effect = [*_get_products_data([108, 109, 110, 108, 109, 110])]
        mocked_get_product_url.side_effect = [""] * 5

        evaluator = Evaluator("")

        evaluated_new_candidate_rows = await asyncio.gather(*[
            evaluator.eval_offer_new_candidate(row) for row in _get_matching_results_new_candidate()
        ])

        assert_new_candidate_eval_results(evaluated_new_candidate_rows)

        alerts = evaluator.calc_log_metrics_new_candidate(evaluated_new_candidate_rows, {"name": "model", "version": "666"})
        current_new_candidates_correct_thresh = alert_thresholds["NEW_PRODUCT_CORRECT_RATIO_THRESHOLD"]
        assert alerts == {
            "warning": [{
                'message': f'model_666 - Ratio of correct new_product is under {current_new_candidates_correct_thresh}, value: 0.67',
                'model_info': {'name': 'model', 'version': '666'}
            }],
        }


def test_coverage_calculation():
    counts_matched = {"": 1, "m_1": 2, "m_2": 3}
    counts_unknown = {"": 5, "m_1": 8, "m_2": 13, "m_3": 21}
    counts_new_candidate = {"m_1": 34, "m_3": 75, "m_4": 109}

    alerts, coverages = calculate_coverage(counts_matched, counts_unknown, counts_new_candidate)

    assert alerts == [{'message': 'm_2 - Coverage has fallen below 0.75, value: 0.19'}]
    assert coverages == {'m_2': 0.19, 'm_3': 0.78, 'm_1': 0.82, 'm_4': 1.0, 'total': 0.84}


@pytest.mark.asyncio
async def test_check_dates():
    with patch('auto_monitoring.main.Evaluator.offer_download') as mocked_offers, \
        patch('auto_monitoring.main.Evaluator.get_full_product_merge_history') as mocked_get_merge_history, \
        patch('auto_monitoring.main.Evaluator.get_product_url') as mocked_get_product_url, \
        patch('auto_monitoring.main.Evaluator.product_download') as mocked_products:

        mocked_offers.side_effect = [*_get_offers_data([15, 16, 17, 18])]
        mocked_get_merge_history.side_effect = [[] for i in range(4)]
        # 00 stands for candidate_data used only for candidate_url creation and for one unpaired product
        mocked_products.side_effect = [*_get_products_data([105, 106, 105, 00, 00, 00, 00])]
        mocked_get_product_url.side_effect = [""] * 5

        evaluator = Evaluator("")
        evaluator.min_days = 6

        evaluated_match_rows = await asyncio.gather(*[
            evaluator.eval_offer_matched(row) for row in _get_matching_results_matched()
        ])

        alerts = evaluator.calc_log_metrics_matched(evaluated_match_rows, {"name": "model", "version": "666"})
        assert alerts == {"critical": [], "warning": []}


def assert_new_candidate_eval_results(evaluated_new_candidate_rows):
    result0 = evaluated_new_candidate_rows[0]
    assert result0 == {
        'correct_decision': 1,
        'incorrect_new_candidate_details': [],
        'model_info': {'name': 'model', 'version': '666'},
        'paired_in_candidates': 1,
        'time': datetime(666, 6, 1, 0, 0)
    }
    result1 = evaluated_new_candidate_rows[1]
    assert result1 == {
        'correct_decision': 1,
        'incorrect_new_candidate_details': [],
        'model_info': {'name': 'model', 'version': '666'},
        'paired_in_candidates': 1,
        'time': datetime(666, 6, 3, 0, 0)
    }
    result2 = evaluated_new_candidate_rows[2]
    assert result2 == {
        'correct_decision': 0,
        'incorrect_new_candidate_details': [
            {
                'candidate_category': 1962, 'candidate_id': 108, 'candidate_name': 'Tvuj strejda', 'candidate_url': '',
                'decision': 'no', 'details': 'prediction=0.09,...', 'model_id': 'model_666', 'offer_id': 22, 'offer_name': 'Jeho baba',
                'offer_url': '', 'paired_product_category': 2337, 'paired_product_id': 110, 'paired_product_name': 'Muj strejda', 'paired_product_url': ''
            },
            {
                'candidate_category': 2337, 'candidate_id': 109, 'candidate_name': 'Jeho strejda', 'candidate_url': '',
                'decision': 'no', 'details': 'Name match', 'model_id': 'model_666', 'offer_id': 22, 'offer_name': 'Jeho baba',
                'offer_url': '', 'paired_product_category': 2337, 'paired_product_id': 110, 'paired_product_name': 'Muj strejda', 'paired_product_url': ''
            },
            {
                'candidate_category': 2337, 'candidate_id': 110, 'candidate_name': 'Muj strejda', 'candidate_url': '',
                'decision': 'no', 'details': 'prediction=0.31,...', 'model_id': 'model_666', 'offer_id': 22, 'offer_name': 'Jeho baba',
                'offer_url': '', 'paired_product_category': 2337, 'paired_product_id': 110, 'paired_product_name': 'Muj strejda', 'paired_product_url': ''
            }
        ],
        'model_info': {'name': 'model', 'version': '666'},
        'paired_in_candidates': 1,
        'time': datetime(666, 6, 5, 0, 0)
    }


def assert_unknown_eval_results(evaluated_unknown_rows):
    result0 = evaluated_unknown_rows[0]
    assert result0 == {
        "paired": 1, "correct_category": 1, "model_id": "model_666", "paired_in_candidates": 1, "paired_decision": "yes",
        "multi_yes": 1, "name_match_in_multi_yes": 0, "ean_match_in_multi_yes": 0, "unsupported_category": 0, 'time': datetime(666, 6, 1),
        "problem_details": [
            {
                "decision": "yes", "offer_id": 10, "offer_name": "Moje baba", "offer_url": "", "candidate_id": 100, "candidate_name": "Moje mama",
                "candidate_url": "", "candidate_category": 1962, "paired_product_id": 100, "paired_product_name": "Moje mama",
                "paired_product_url": "", "paired_product_category": 1962, "details": "prediction=0.71,...", "model_id": "model_666",
            },
            {
                "decision": "yes", "offer_id": 10, "offer_name": "Moje baba", "offer_url": "", "candidate_id": 21, "candidate_name": "Tvoje mama",
                "candidate_url": "", "candidate_category": 1962, "paired_product_id": 100, "paired_product_name": "Moje mama",
                "paired_product_url": "", "paired_product_category": 1962, "details": "prediction=0.89,...", "model_id": "model_666",
            },
        ],
        "comparisons": {
            "100": {
                "decision": "yes",
                "details": "prediction=0.71,...",
            },
            "21": {
                "decision": "yes",
                "details": "prediction=0.89,...",
            },
            "101": {
                "decision": "no",
                "details": "prediction=0.36,...",
            }
        },
    }
    result1 = evaluated_unknown_rows[1]
    assert result1 == {
        "paired": 1, "correct_category": 0, "model_id": "model_666", "paired_in_candidates": 0, "paired_decision": "",
        "multi_yes": 1, "name_match_in_multi_yes": 1, "ean_match_in_multi_yes": 0, "unsupported_category": 0, 'time': datetime(666, 6, 2),
        "problem_details": [
            {
                "decision": "yes", "offer_id": 11, "offer_name": "Tvoje baba", "offer_url": "", "candidate_id": 100, "candidate_name": "Moje mama",
                "candidate_url": "", "candidate_category": 1962, "paired_product_id": 102, "paired_product_name": "Cizi mama",
                "paired_product_url": "", "paired_product_category": 1965, "details": "prediction=0.89,...", "model_id": "model_666",
            },
            {
                "decision": "yes", "offer_id": 11, "offer_name": "Tvoje baba", "offer_url": "", "candidate_id": 21, "candidate_name": "Tvoje mama",
                "candidate_url": "", "candidate_category": 1962, "paired_product_id": 102, "paired_product_name": "Cizi mama",
                "paired_product_url": "", "paired_product_category": 1965, "details": "Name match", "model_id": "model_666",
            },
        ],
        "comparisons": {
            "100": {
                "decision": "yes",
                "details": "prediction=0.89,..."
            },
            "21": {
                "decision": "yes",
                "details": "Name match",
            },
            "101": {
                "decision": "no",
                "details": "prediction=0.31,...",
            },
        },
    }
    result2 = evaluated_unknown_rows[2]
    assert result2 == {
        "paired": 1, "correct_category": 0, "model_id": "model_666", "paired_in_candidates": 1, "paired_decision": "unknown", 'time': datetime(666, 6, 3),
        "multi_yes": 0, "name_match_in_multi_yes": 0, "ean_match_in_multi_yes": 0, "unsupported_category": 0, "problem_details": [], "comparisons": {},
    }
    result3 = evaluated_unknown_rows[3]
    assert result3 == {
        "paired": 0, "correct_category": 0, "model_id": "model_666", "paired_in_candidates": 0, "paired_decision": "", 'time': datetime(666, 6, 5),
        "multi_yes": 0, "name_match_in_multi_yes": 0, "ean_match_in_multi_yes": 0, "unsupported_category": 0, "problem_details": [], "comparisons": {},
    }


def assert_matched_eval_results(evaluated_matched_rows):
    result0 = evaluated_matched_rows[0]
    assert result0 == {
            "paired": 1, "model_info": {"name": "model", "version": "666"}, "name_match": 0, "ean_match": 0, "correct": 1,
            'time': datetime(666, 6, 1), "correct_category": 0, "paired_in_candidates": 1, "incorrect_match_details": [],
            "comparisons" : {
                "104": {
                    "details": "prediction=0.51,...",
                    "decision": "unknown"
                },
                "105": {
                    "details": "prediction=0.89,...",
                    "decision": "yes"
                },
                "106": {
                    "details": "prediction=0.36,...",
                    "decision": "no"
                },
            },
        }
    result1 = evaluated_matched_rows[1]
    assert result1 == {
            "paired": 1, "model_info": {"name": "model", "version": "666"}, "name_match": 0, "ean_match": 1, "correct": 1,
            'time': datetime(666, 6, 2), "correct_category": 1, "paired_in_candidates": 1, "incorrect_match_details": [],
            "comparisons": {
                "104": {
                    "details": "prediction=0.19,...",
                    "decision": "no"
                },
                "105": {
                    "details": "prediction=0.31,...",
                    "decision": "no"
                },
                "106": {
                    "details": "Ean match",
                    "decision": "yes"
                },
            },
        }
    result2 = evaluated_matched_rows[2]
    assert result2 == {
            "paired": 1, "model_info": {"name": "model", "version": "666"}, "name_match": 0, "ean_match": 0, "correct": 0,
            'time': datetime(666, 6, 3), "correct_category": 0, "paired_in_candidates": 1, "incorrect_match_details": [
                {
                    "decision": "yes", "offer_id": 17, "offer_name": "Jeho tata", "offer_url": "", "candidate_id": 104,
                    "candidate_name": "Muj deda", "candidate_url": "", "candidate_category": 2337, "paired_product_id": 105,
                    "paired_product_name": "Tvuj deda", "paired_product_url": "", "paired_product_category": 1962,
                    "details": "prediction=0.99,...", "model_id": "model_666",
                },
                {
                    "decision": "unknown", "offer_id": 17, "offer_name": "Jeho tata", "offer_url": "", "candidate_id": 105,
                    "candidate_name": "Tvuj deda", "candidate_url": "", "candidate_category": 1962, "paired_product_id": 105,
                    "paired_product_name": "Tvuj deda", "paired_product_url": "", "paired_product_category": 1962,
                    "details": "prediction=0.52,...", "model_id": "model_666",
                },
            ],
            "comparisons": {
                "104": {
                    "details": "prediction=0.99,...",
                    "decision": "yes"
                },
                "105": {
                    "details": "prediction=0.52,...",
                    "decision": "unknown"
                },
                "106": {
                    "details": "prediction=0.49,...",
                    "decision": "unknown"
                },
            },
        }
    result3 = evaluated_matched_rows[3]
    assert result3 == {
            "paired": 0, "model_info": {"name": "model", "version": "666"}, "name_match": 1, "ean_match": 0, "correct": 0,
            'time': datetime(666, 6, 5), "correct_category": 0, "paired_in_candidates": 0, "incorrect_match_details": [],
            "comparisons": {
                "104": {
                    "details": "Name match",
                    "decision": "yes"
                },
                "105": {
                    "details": "prediction=0.25,...",
                    "decision": "no"
                },
                "106": {
                    "details": "prediction=0.44,...",
                    "decision": "unknown"
                },
            },
        }


def _get_offers_data(ids):
    data_full = {
        10: {"id": 10, "name": "Moje baba", "match_name": "Moje baba", "product_id": 100, "url": ""},
        11: {"id": 11, "name": "Tvoje baba", "match_name": "Tvoje baba", "product_id": 102, "url": ""},
        12: {"id": 12, "name": "Jeho baba", "match_name": "Jeho baba", "product_id": 101, "url": ""},
        13: {"id": 13, "name": "Cizi baba", "match_name": "Cizi baba", "product_id": "", "url": ""},

        15: {"id": 15, "name": "Muj tata", "match_name": "Muj tata", "product_id": 105, "url": ""},
        16: {"id": 16, "name": "Tvuj tata", "match_name": "Tvuj tata", "product_id": 106, "url": ""},
        17: {"id": 17, "name": "Jeho tata", "match_name": "Jeho tata", "product_id": 105, "url": ""},
        18: {"id": 18, "name": "Cizi tata", "match_name": "Cizi tata", "product_id": "", "url": ""},

        20: {"id": 20, "name": "Muj tata", "match_name": "Moje teta", "product_id": 108, "url": ""},
        21: {"id": 21, "name": "Tvuj tata", "match_name": "Tvoje teta", "product_id": 109, "url": ""},
        22: {"id": 22, "name": "Jeho tata", "match_name": "Jeho teta", "product_id": 110, "url": ""},
    }
    return [data_full[i] for i in ids]


def _get_products_data(ids):
    data_full = {
        100: {"id": 100, "name": "Moje mama", "category_id": 1962},
        101: {"id": 101, "name": "Jeho mama", "category_id": 1963},
        102: {"id": 102, "name": "Cizi mama", "category_id": 1965},

        105: {"id": 105, "name": "Tvuj deda", "category_id": 1962},
        106: {"id": 106, "name": "Jeho deda", "category_id": 2337},
        104: {"id": 104, "name": "Muj deda", "category_id": 2337},

        108: {"id": 108, "name": "Tvuj strejda", "category_id": 1962, "founder_offer": {"offer_id": 20, "relation_time": "0666-06-04T00:00:00"}},
        109: {"id": 109, "name": "Jeho strejda", "category_id": 2337, "founder_offer": {"offer_id": 20, "relation_time": "0666-06-04T00:00:00"}},
        110: {"id": 110, "name": "Muj strejda", "category_id": 2337, "founder_offer": {"offer_id": 30, "relation_time": "0666-06-04T00:00:00"}},

        00: {},
    }

    return [data_full[i] for i in ids]
