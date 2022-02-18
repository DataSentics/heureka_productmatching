import logging
import numpy as np
import pandas as pd


class ModelComparator():
    def __init__(self):
        self.sheet_names = ["matched", "new_product", "unknown", "no_candidates"]
        self.excel_0 = None
        self.excel_1 = None
        self.item_ids = set()
        self.items_decisions = pd.DataFrame()

    def load(self, excel_path: str, validation_excel_path: str):
        # load excels
        self.excel_0 = self.load_validation_excel(excel_path, self.sheet_names)
        self.excel_1 = self.load_validation_excel(validation_excel_path, self.sheet_names)
        # get all items ids, check intersection is positive
        self.item_ids = self.get_all_items_ids()
        # store item id and decision from both excels
        self.items_decisions = self.get_item_to_decisions_df()

    def get_all_items_ids(self) -> set:
        """
        Calculates the items intersection in both excels, which should be positive.
        Returns item ids union.
        """
        items_excel_0 = set()
        items_excel_1 = set()
        # get item ids in all selected sheets
        for sn in self.sheet_names:
            # one item should not be present in multiple sheets
            items_excel_0 |= set(self.get_items(self.excel_0, sn))
            items_excel_1 |= set(self.get_items(self.excel_1, sn))

        intersection = items_excel_0 & items_excel_1
        union = items_excel_0 | items_excel_1

        logging.info(f"Will work with {len(intersection)}/{len(union)} items occuring in both excels ")
        # excels are probably wrongly chosen if there are no common items
        assert len(intersection) > 0, "At least one common offer is needed"
        return union

    @staticmethod
    def load_validation_excel(path: str, sheet_names: list) -> dict:
        """
        Loads specified sheets from validation excel in given path.
        Skips deserved sheets if they are not present.
        """
        excel_sheets = {}
        for sheet_name in sheet_names:
            try:
                sheet = pd.read_excel(path, engine="openpyxl", sheet_name=sheet_name)
                # remove columns with a lot (80%) of missing values  (e.g. content decision, orig. paired product)
                excel_sheets[sheet_name] = sheet.dropna(axis="columns", thresh=sheet.shape[0]*0.2)
            except KeyError:
                # return empty sheet, no items with given decision found
                logging.warning(f"Sheet {sheet_name} not present in {path}")
        logging.info(f"Excel {path} loaded")
        return excel_sheets

    def get_item_to_decisions_df(self) -> pd.DataFrame:
        """
        Creates dataframe with final decisions from both excels for each item id.
        """
        # all decisions from first and second excel
        decisions_first_excel = pd.DataFrame()
        decisions_second_excel = pd.DataFrame()

        for sn in self.sheet_names:
            # append all item ids from first excel with final decision 'sn'
            decisions_first_excel = pd.concat(
                [
                    decisions_first_excel,
                    pd.DataFrame(
                        {
                            "item_id": list(self.get_items(self.excel_0, sn)),
                            "first_model_decision": sn
                        }
                    )
                ],
                ignore_index=True
            )
            # the same for the second excel
            decisions_second_excel = pd.concat(
                [
                    decisions_second_excel,
                    pd.DataFrame(
                        {
                            "item_id": list(self.get_items(self.excel_1, sn)),
                            "second_model_decision": sn
                        }
                    )
                ],
                ignore_index=True
            )
        # get both decisions to item ids by merge, NaN in some column if item was present in only one excel
        # those items will not occur in table and output sheets
        return pd.merge(decisions_first_excel, decisions_second_excel, how="outer", on="item_id")

    @staticmethod
    def get_items(excel: dict, sheet_name: str) -> np.array:
        """
        Returns unique item ids from fiven excel and sheet or empty array if no items found.
        """
        try:
            return excel[sheet_name]["item_id"].drop_duplicates().dropna().to_numpy(dtype=np.int64)
        except KeyError:
            return np.array([])

    def create_counts_table(self) -> pd.DataFrame:
        """
        Creates table with number of items for all final decision combinations present in excels.
        """
        # get counts
        stats = self.items_decisions.groupby(["first_model_decision", "second_model_decision"]).size()
        # fill NaN with 0 when no items present
        return stats.unstack().fillna(0)

    def get_rows_for_decision_combination(self, decision_0: str, decision_1: str, separete_by: int = 3) -> pd.DataFrame:
        """
        Returns rows containing all occureces of items, which have specified decision combination in first and second excel.
        Rows between first and second excel are separated by one empty line, different items are separated by more empty rows.
        """
        # select item ids with given decision combination
        selected_item_ids = self.items_decisions[
            (self.items_decisions["first_model_decision"] == decision_0)
            &
            (self.items_decisions["second_model_decision"] == decision_1)
        ]["item_id"].to_numpy()

        logging.info(f"{decision_0}-{decision_1}: {len(selected_item_ids)} items")

        empty_row = pd.DataFrame({c: [""] for c in self.excel_0[decision_0].columns})
        empty_space = pd.DataFrame({c: [""]*separete_by for c in self.excel_0[decision_0].columns})

        # put all dataframes to list and concat it at the end, more efficient than concat in each iteration
        dfs_to_concat = []
        # create output rows
        for iid in selected_item_ids:
            # rows from first excel
            first_excel_rows = self.excel_0[decision_0][self.excel_0[decision_0]["item_id"] == iid]
            # rows from second excel
            second_excel_rows = self.excel_1[decision_1][self.excel_1[decision_1]["item_id"] == iid]
            # add empty rows and add the new block containing one offer info to output dataframe
            dfs_to_concat.extend([first_excel_rows, empty_row, second_excel_rows, empty_space])

        if dfs_to_concat:
            output_df = pd.concat(dfs_to_concat)
        else:
            output_df = pd.DataFrame()

        return output_df

    def create_comparison_excel(self, filepath: str,  separate_by: int = 3):
        """
        Main function to create the comparison excel
        """
        sheets = {}
        # create table with items count for decision combinations
        sheets.update({"decisions_table": self.create_counts_table()})

        present_sheets_first = [k for k,v in self.excel_0.items() if not v.empty]
        present_sheets_second = [k for k,v in self.excel_1.items() if not v.empty]

        # put items with given decisions combination to separate sheet
        for sn0 in present_sheets_first:
            for sn1 in present_sheets_second:
                sheet_name = f"{sn0}-{sn1}"
                # create block containing all items with given combination
                rows = self.get_rows_for_decision_combination(sn0,sn1,separate_by)
                if len(rows) > 0:
                    sheets.update({sheet_name:rows})
        # write all sheets to excel
        with pd.ExcelWriter(filepath) as writer:
            for name, sheet in sheets.items():
                if name == "decisions_table":
                    # write also index to keep row names in the table
                    sheet.to_excel(writer, name, index=True)
                else:
                    sheet.to_excel(writer, name, index=False)