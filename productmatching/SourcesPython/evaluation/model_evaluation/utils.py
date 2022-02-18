import os
import logging
import pandas as pd
import matplotlib.pyplot as plt

from url_generator import UrlGenerator

from utilities.helpers import argmax_full


def get_product_url(category_slug: str, product_slug: str) -> str:
    path = os.path.join(
        os.getcwd(), 'resources', 'candy',  'resources', 'url-generator-routes', 'routes.json')
    url = UrlGenerator(path, lang="cz", env="production")
    return url.get_url("heureka.product", category_seo=category_slug, product_seo=product_slug)


class DocumentCreator():
    def __init__(self, items_data: dict, candidates_data: dict):
        self.items_data = items_data
        self.candidates_data = candidates_data
        self.decision_order = {"yes": 0, "unknown": 1, "no": 2, "invalid data": 3, "failed prediction": 4}
        self.def_output_cols = [
            'content_decision',
            'decision',
            'item_name',
            'candidate_name',
            'item_url',
            'candidate_url',
            'item_id',
            'candidate_id',
            'category_id',
            'details',
            'candidate_source',
            'paired_id',
            'paired_name',
            'paired_url',
            'paired_category_id',
            'paired_id_equals_matched',
            'normalized_product_name',
            'normalized_offer_name',
            'offer_ean',
            'product_eans',
            'product_status',
        ]
        self.sheets = {}

    def create_final_excel(self, filepath: str, sheets_data: dict, unformatted_sheets: dict, complete_sheets: dict, append_row=True):
        with pd.ExcelWriter(filepath) as writer:
            sheets = {
                sheet_name: self.pdf_from_matches(matches_list, append_row) for sheet_name, matches_list in sheets_data.items()
            }
            sheets.update(unformatted_sheets)
            for sheet_name, pdf_sheet in sheets.items():
                pdf_sheet['content_decision'] = ""
                out_cols = sorted(pdf_sheet.columns, key=lambda x: self.def_output_cols.index(x))
                pdf_sheet[out_cols].to_excel(writer, sheet_name, index=False)
            for sheet_name, pdf_sheet in complete_sheets.items():
                pdf_sheet.to_excel(writer, sheet_name, index=False)

    def pdf_from_matches(self, matches_list: list, append_row=True) -> pd.DataFrame:
        pdf_out = pd.concat(
            [pd.DataFrame(rows) for rows in self.process_matches(matches_list, append_row)],
            ignore_index=True, sort=False, axis=0
        )
        return pdf_out

    def process_matches(self, matches_list: list, append_row: bool):
        return [self.process_match(matches, append_row) for matches in matches_list]

    def process_match(self, matches: list, append_row: bool) -> list:
        item_id = matches[0]["item_id"]
        item_data = self.items_data[item_id]
        item_row = self.create_item_part_of_row(item_data)
        output_rows = []
        if item_row:
            for match in matches:
                if match["candidate_id"] in self.candidates_data:
                    outr = self.create_candidate_part_of_row(
                        item_row,
                        self.candidates_data[match["candidate_id"]],
                        match
                    )
                    output_rows.append(outr)
            output_rows = sorted(output_rows, key=lambda x: self.decision_order[x['decision']])
            if append_row:
                output_rows.append({k: "" for k in outr.keys()})
        return output_rows

    def create_item_part_of_row(self, data: dict) -> dict:
        """
        Returns dict which contains general item's info.
        """
        try:
            pcs = data.get('product_category_slug', None)
            ps = data.get('product_slug', None)
            if pcs and ps:
                url = get_product_url(pcs, ps)
            else:
                url = "No paired product data"
            row = {
                'item_id': data['id'],
                'item_name': data['match_name'],
                'item_url': data['url'],
                'paired_id': data.get('product_id', "No paired product data"),
                'paired_name': data.get('product_name', "No paired product data"),
                'paired_category_id': data.get('product_category_id', "No paired product data"),
                'paired_url': url,
                'offer_ean': data.get('ean'),
            }
        except Exception:
            # it can happen e.g. when the offer is no longer active,
            # 'product_name' and few other fields. are missing in that case
            logging.exception(f"Cannot create row from: {data}")
            row = {
                'item_id': '',
                'item_name': '',
                'item_url': '',
                'paired_id': '',
                'paired_name': '',
                'paired_category_id': '',
                'paired_url': '',
            }

        return row

    def create_candidate_part_of_row(self, row_in: dict, comparisons_candidate: dict, match_details: dict) -> dict:
        """
        Returns dict for placing data of candidate from comparisons list into a
        tsv_candidates file row with certain tsv_candidates columns.
        """
        row = row_in.copy()
        ccs = comparisons_candidate.get('category_slug', None)
        cs = comparisons_candidate.get('slug', None)
        if ccs and cs:
            candidate_url = get_product_url(ccs, cs)
        else:
            candidate_url = ""

        row['decision'] = match_details['decision']
        row['details'] = match_details['details']
        row['candidate_name'] = comparisons_candidate.get('name', "")
        row['candidate_id'] = comparisons_candidate.get('id', "")
        row['category_id'] = comparisons_candidate.get('category_id', "")
        row['candidate_url'] = candidate_url
        row['candidate_source'] = match_details.get('sources', "")
        row['normalized_product_name'] = match_details.get('norm_product_name')
        row['normalized_offer_name'] = match_details.get('norm_offer_name')
        row['product_eans'] = ",".join([str(ean) for ean in comparisons_candidate['eans']]) if comparisons_candidate.get('eans', None) else ""
        row['product_status'] = match_details['candidate']['data'].get('status', {}).get('id', "")
        # add 0/1 flag if matched according to current pairing
        if 'paired_id' in row and row['decision'] == "yes":
            row['paired_id_equals_matched'] = int(row['candidate_id'] == row['paired_id'])

        return row


class ThresholdsEvaluator:
    """
    Class for processing of rematch results with different thresholds for xgb confidence.
    """
    def __init__(self, pdf_thresholds):
        self.pdf = pdf_thresholds
        assert len({"threshold", "precision", "matched_pct"} & set(self.pdf.columns)) == 3

    def plot_threshold_results(self, path):
        """
        Generates and saves a plot with lines corresponding to precision and matched with respect to chosen threshold.
        """
        fig, ax = plt.subplots()

        ax.plot(self.pdf.threshold, self.pdf.precision, color='magenta')
        ax.tick_params(axis='y', labelcolor='magenta')
        ax.set_ylabel("precision")
        ax.set_xlabel("threshold")

        ax2 = ax.twinx()
        ax2.plot(self.pdf.threshold, self.pdf.matched_pct, color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        ax2.set_ylabel("matched_pct", labelpad=12, rotation=270)

        plt.legend()
        plt.savefig(path, bbox_inches='tight', dpi=200)
        plt.close(fig)

    def get_weight_maximum_scores(self) -> pd.DataFrame:
        """
        When deciding which threshold to use in production, we have to somehow decide how we weight matched_pct against precision.
        Since it is unclear how to set the weight, we try multiple different weight combinations and find thresholds with the best score for each.
        """
        multipliers = [i/100 for i in range(10, 101, 1)]
        results = []
        for m in multipliers:
            scores = self.pdf.precision + m * self.pdf.matched_pct
            argmaxes = argmax_full(scores)
            results += [(self.pdf.threshold[i], self.pdf.precision[i], self.pdf.matched_pct[i], m, scores[i]) for i in argmaxes]

        pdf = pd.DataFrame(results)
        pdf.columns = ["threshold", "precision", "matched_pct", "multiplier", "score"]
        # percentage of maximal score obtained, just for info
        pdf["score_scaled"] = pdf.score / (1 + pdf.multiplier)
        return pdf

    @staticmethod
    def get_top_threshold(pdf_top_scores) -> float:
        """
        Digests the results of `get_weight_maximum_scores` method.
        The best score is the one that 'won' most times.
        If we have multiple winners, the highest corresponding threshold is chosen.
        """
        pdf_ = pd.DataFrame(pdf_top_scores.groupby("threshold").size())
        pdf_.columns = ["count"]
        pdf_.reset_index(inplace=True)
        pdf_.sort_values(["count", "threshold"], ascending=False, inplace=True)
        return pdf_.threshold.iloc[0]
