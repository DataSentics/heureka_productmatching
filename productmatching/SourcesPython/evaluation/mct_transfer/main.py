import os
import asyncio
import argparse
import mlflow
import logging
from utilities.args import str2bool
from utilities.kafka_utils import kafka_output

from comparison.utils import ModelComparator
from utilities.notify import notify

from utilities.component import process_input

from utilities.remote_services import get_remote_services
from candy.logic.candy import Candy


DECISIONS_TO_TOPICS = {
    "matched": "matching-ng-item-validation-matched-cz",
}

# there are different naming used, this dicts maps one to another
DECISION_TO_FINAL_DECISION = {
    "matched": "yes",
    "new_product": "no",
    "unknown": "unknown"
}


def get_message_data_matched(item_comparisons):
    """
    Parse comparison rows into fields needed for kafka message for matched offers.
    """

    final_decision = DECISION_TO_FINAL_DECISION["matched"]
    item_id = item_comparisons["item_id"].iloc[0]
    candidates = []
    comparisons = []
    item_dict = {
        "id": str(int(item_id)),
        "match_name": item_comparisons["item_name"].iloc[0],
        # not needed
        "shop_id": "",
    }
    for _, cand in item_comparisons.iterrows():
        candidates.append({
            "data": {
                "id": str(int(cand["candidate_id"])),
                "name": cand["candidate_name"],
                "category_id": cand["category_id"],
            },
            # not needed
            "distance": "",
            # not needed
            "relevance": "",
        })
        comparisons.append({
            "candidate": {
                "id": str(int(cand["candidate_id"])),
            },
            "decision": cand["decision"],
            "details": cand["details"],
        })

    # final candidate
    final_candidate = ""
    if final_decision == "yes":
        matched_candidates = [c["candidate"] for c in comparisons if c["decision"] == "yes"]
        # safety check, it should never happen
        if len(matched_candidates) != 1:
            raise ValueError(f"{len(matched_candidates)} candidates matched when only one should be matched")
        final_candidate = {"id": matched_candidates[0]["id"]}

    all_offer_info = {
        "item_dict": item_dict,
        "final_decision": final_decision,
        "final_candidate": final_candidate,
        "candidates": candidates,
        "comparisons": comparisons,
    }

    return all_offer_info


async def mct_transfer(args):

    remote_services = await get_remote_services(["kafka"])
    logging.info(f"Remote services initialized")

    # comparator used only for processing excel
    matching_excel = ModelComparator.load_validation_excel(args.excel_path, args.decisions_to_validate)
    model_info = {
        "name": f"{args.model_name}",
        "tags": {
            "categories": args.categories,
            "cache_address": args.cache_address,
        },
        "version": "preprod",

    }
    # not used anywhere, category of final candidate is used
    possible_categories = ""

    # using only matched decision for now
    for decision in args.decisions_to_validate:
        assert decision in DECISION_TO_FINAL_DECISION.keys(), f"Unknown decision {decision}"

        logging.info(f"Processing '{decision}' items")

        items = matching_excel[decision]
        # remove candidates with invalid data
        items = items[items["decision"] != "invalid data"]
        item_ids = ModelComparator.get_items(matching_excel, decision)

        cnt = 0
        for item_id in item_ids:
            all_offer_data = {}
            item_comparisons = items[items["item_id"] == item_id]
            # TODO: similar logic for different decisions can be implemented analogously if needed
            if decision == "matched":
                # do not send message if only varying matched wanted and the matched product is the same as on Heureka
                if args.only_varying_matches and int(max(item_comparisons["paired_id_equals_matched"].dropna())) == 1:
                    all_offer_data = {}
                else:
                    all_offer_data = get_message_data_matched(item_comparisons)

            # send only if message data provided
            if all_offer_data:
                msg = Candy._get_final_message(
                    final_decision=all_offer_data["final_decision"],
                    item=all_offer_data["item_dict"],
                    final_candidate=all_offer_data["final_candidate"],
                    candidates=all_offer_data["candidates"],
                    comparisons=all_offer_data["comparisons"],
                    pc_message=possible_categories,
                    model_info=model_info
                )
                await kafka_output(
                    remote_services,
                    msg,
                    key=str(item_id),
                    topic=DECISIONS_TO_TOPICS[decision]
                )
                cnt += 1

        logging.info(f"Sent {cnt} offers with {decision} to {DECISIONS_TO_TOPICS[decision]}")

    await remote_services.close_all()


@notify
def main(args):
    asyncio.run(mct_transfer(args))


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), force=True)
    parser = argparse.ArgumentParser()

    parser.add_argument("--excel-path", required=True)
    parser.add_argument("--data-directory", default="/data")
    parser.add_argument("--decisions-to-validate", type=str, default="matched")
    parser.add_argument("--only-varying-matches", type=str2bool, default=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--categories", type=str, required=True)
    parser.add_argument("--cache-address", type=str, required=True)

    args = parser.parse_args()

    args.excel_path = process_input(args.excel_path, args.data_directory)
    args.decisions_to_validate = args.decisions_to_validate.split("@")

    logging.info(args)

    with mlflow.start_run():
        main(args)
