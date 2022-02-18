eval_period = 14
eval_gap = 14
alert_thresholds = {
    "UNPAIRED_MATCHED_RATIO_THRESHOLD": 0.2,
    "INCORRECT_MATCHED_RATIO_THRESHOLD": 0.92,
    "COVERAGE_THRESHOLD": 0.75,
    "NEW_PRODUCT_CORRECT_RATIO_THRESHOLD": 0.7,
}
# Nr. of rows of rows to fetch from Dexter from a given table. -1 -> all rows.
# Use cautiously only when the full table is truly large. Otherwise performance
#   might actually be slower since a random shuffle is triggered together with
#   the limit.
n_limit_rows = {
    "matched": -1,
    "unknown": 500_000,
    "new_candidate": -1,
}
