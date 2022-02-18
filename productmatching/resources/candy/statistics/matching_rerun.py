import asyncio
import ujson
import csv
import logging

from collections import defaultdict, namedtuple
from typing import List, Tuple
from unittest.mock import patch
from os import path

from buttstrap.remote_services import RemoteServices
from candy.config import init_config
from llconfig import Config
from pathlib import Path

from candy.logic.implementation.client import ClientCandy
from candy.logic.candy import Decision
from candy.logic.providers import Candidate

COMPARE_DIR = './compare/'
COMPARE_ONLY_FINAL_DIR = './compare_only_final/'
MATCHAPI_DUMP_DIR = './dump/'
MATCHAPI_DUMP_TWO_DIR = './dump_two/'
COMPARED_DECISIONS_DIR = './compared/'

TOPIC_NAME_FINAL = [
    'matching-ng-item-matched-cz',
]

TOPIC_NAME = [
    'matching-ng-item-new-candidate-cz',
    'matching-ng-unknown-match-cz',
]


async def init_conf() -> Tuple[RemoteServices, Config]:
    config = init_config()

    config["KAFKA"]["bootstrap.servers"] = "kafka:29092"
    config["KAFKA"]["consumer"]["group.id"] = "matching_rerun"
    config["KAFKA"]["consumer"]["enable.auto.commit"] = True
    config["KAFKA"]["consumer"]["default.topic.config"] = {
        "auto.offset.reset": "earliest"
    }

    remote_services = RemoteServices(
        config,
        rest=['matchapi'],
        kafka=['kafka'],
        redis=['redis_offers']
    )

    await remote_services.init()
    return remote_services, config


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


async def process_items(files: Tuple):
    topic_name = files.dump_file.rstrip('.txt')

    first_pass_messages = collect_messages_from_txt(MATCHAPI_DUMP_DIR + files.dump_file)

    async def gather_secondary_decision(matchapi_message: dict, topic=topic_name):
        with open(MATCHAPI_DUMP_TWO_DIR + f'{topic}.txt', 'a') as file_txt:
            result = ujson.dumps(matchapi_message)
            file_txt.write(result + '\n')

    with patch("uuid.uuid4") as mocked_uuid4, \
            patch('candy.logic.candy.Candy._get_items_data') as mocked_items_data, \
            patch('candy.logic.implementation.client.ClientCandy._get_candidate_data') as mocked_candidate_data, \
            patch('candy.logic.implementation.client.ClientCandy.get_remains_from_process_queue') as mocked_get_remains_from_process_queue, \
            patch('candy.logic.implementation.client.ClientCandy.ack') as mocked_ack, \
            patch('candy.logic.candy.Candy._get_candidates') as mocked_candidates_data:
        remote_services, config = await init_conf()

        mocked_get_remains_from_process_queue.side_effect = _get_remains_from_process_queue()

        candy = ClientCandy(remote_services, config)
        candy.output_item_matched = candy.output_item_not_matched = \
            candy.output_item_unknown = gather_secondary_decision

        await candy.init_queue_tasks()
        for index, messages in enumerate(chunks(first_pass_messages, 10)):
            logging.info(f'Bulk of messages number {index}')
            mocked_uuid4.side_effect = [message['uuid'] for message in messages]

            mocked_items_data.side_effect = [_get_items_data(messages)] + \
                                            [_get_items_generator_data(messages, i) for i in range(len(messages))]
            mocked_candidate_data.side_effect = [_get_candidate_data(messages, i) for i in range(len(messages))]
            mocked_candidates_data.side_effect = [_get_candidates(messages)]

            await candy.process_items_candidates([message['item']['id'] for message in messages])

        await remote_services.close_all()


def _get_candidates(messages) -> defaultdict:
    """
    The method is used for mocking candy's method with identical name.
    Returns:
        defaultdict: dict with key as an item id and candidates for the item as a value.
    """
    item_candidates = defaultdict(lambda: defaultdict(dict))

    for message in messages:
        item_candidates[message['item']['id']] = \
            {c['candidate']['id']: Candidate(
                id=c['candidate']['id'],
                distance=float(c['candidate']['distance']) if c['candidate'].get('distance') else None,
                relevance=float(c['candidate']['relevance']) if c['candidate'].get('relevance') else None,
                source=c['candidate']['source'] if 'source' in c['candidate'] else None) for c in message['comparisons']}

    return item_candidates


def _get_items_data(messages):
    value = [message['item']['data'] for message in messages]

    async def fraud():
        return value

    return fraud()


def _get_items_generator_data(messages, set_: int):
    data = [messages[set_]["item"]["data"]]

    async def fraud():
        return data

    return fraud()


def _get_candidate_data(messages, set_):
    data = messages[set_]
    value = [candidate['candidate']['data'] for candidate in data['comparisons']]

    async def fraud():
        return value

    return fraud()


def _ack(item_id):

    async def fraud():
        return

    return fraud()


def _get_remains_from_process_queue():

    async def fraud():
        return []

    return fraud

def compare_all_decisions(files: tuple, final: bool):
    """Compares all decisions for candidates where final decisions is not available"""
    topic_name = files.dump_file.rstrip('.txt')

    if path.exists(MATCHAPI_DUMP_TWO_DIR + files.dump_file):
        first_batch = collect_first_batch_decisions(files.decisions_file, final=final)
        dump_two_messages = collect_messages_from_txt(MATCHAPI_DUMP_TWO_DIR + files.dump_file)
        result_comparison = compare_first_second_batch(first_batch, dump_two_messages, topic_name, final=final)
        create_result_tsv(result_comparison, topic_name)
    else:
        logging.warning(f"Path does not exists {MATCHAPI_DUMP_TWO_DIR + f'{files.dump_file}'}")


def collect_first_batch_decisions(compare_tsv, final=False):
    """Goes through tsv file and collets all decisions made by MatchAPI and content.
    Returns list od dict like {'UUID':[{'candidate_id':{'content_decision':'', 'decision':''}}]"""
    first_batch_decisions = {}

    dir_ = COMPARE_ONLY_FINAL_DIR if final else COMPARE_DIR

    with open(dir_ + compare_tsv, mode='r') as tsv_file:
        tsv_reader = csv.DictReader(tsv_file, delimiter='\t')
        for row in tsv_reader:
            if row['uuid'] == '':
                continue
            if not row['uuid'] in first_batch_decisions:
                first_batch_decisions[row['uuid']] = {}
            first_batch_decisions[row['uuid']].update({
                row['candidate_id']: {
                    'content_decision': row['content_decision'],
                    'first_matchapi_decision': row['decision'],
                    'first_matchapi_details': row['details'],
                }
            })

    return first_batch_decisions


def collect_messages_from_txt(file_name: str) -> List[dict]:
    """ Collects all messages from already processed items.
     Returns:
         List[dict]: list of messages received from MatchAPI and stored into a txt file.
    """
    with open(file_name, 'r') as file_txt:
        messages_dict_list = [ujson.loads(message) for message in file_txt]
    return messages_dict_list


def create_result_tsv(result_comparison, topic_name):
    """
    Creates a tsv file with compared decisions made by content, matchapi first pass and matchapi second pass.
    """
    tsv_columns = ['content_decision', 'first_matchapi_decision', 'second_matchapi_decision', 'status', 'uuid',
                   'item_id', 'item_name', 'candidate_id', 'candidate_name', 'first_matchapi_details', 'second_matchapi_details']

    with open(COMPARED_DECISIONS_DIR + f'{topic_name}.tsv', 'w', newline='\n') as tsvfile:
        writer = csv.DictWriter(tsvfile, fieldnames=tsv_columns, delimiter='\t', restval='---')
        writer.writeheader()

        for comparison in result_comparison:
            writer.writerow(comparison)


def get_decision_weight(decision: str) -> int:
    decision = decision.lower().strip()

    if decision == Decision.yes.value:
        return 3
    elif decision == Decision.unknown.value:
        return 2
    elif decision == Decision.no.value:
        return 1
    elif decision == "":
        logging.warning(f"No decision for {decision}")
        return 0
    else:
        raise ValueError(f"Unknown decision {decision}")



def compare_first_second_batch(first_batch_decision: List[dict],
                               second_batch_messages: List[dict],
                               topic_name: str,
                               final=False):
    """
    Compare content, matchapi first pass and matchapi second pass decisions.
    Args:
        first_batch_decision (List): Decisions made by a content and MatchAPI in a first attempt.
        second_batch_messages (List): Messages collected from a second attempt of running same items and candidates
            through MatchAPI.
        topic_name (str): name of topic where messages stored.
        final (bool): If a current file contains only final decisions.

    Returns:
        List[dict]: returns list with ordered decisions ex.: worse, better, unchanged.
    """
    decision_better = []
    decision_worse = []
    decision_unchanged = []

    for message in second_batch_messages:
        item_uuid = message['uuid']
        message_from_first_batch = first_batch_decision[item_uuid]

        result = {
            'uuid': item_uuid,
            'item_id': message['item']['id'],
            'item_name': message['item']['data']['match_name']
        }

        if not final:
            candidates = message['comparisons']
        else:
            if message['final_decision'] == 'yes':
                candidates = [{'candidate': message['candidate'], 'decision': message['final_decision'], 'details': message['comparisons'][0]["details"]}]
            else:
                candidates = message['comparisons']

                first_batch_not_final = collect_first_batch_decisions(topic_name + '.tsv')
                message_from_first_batch = first_batch_not_final[item_uuid]

        for candidate in candidates:
            candidate_id = candidate['candidate']['id']
            content_decision = message_from_first_batch[candidate_id]['content_decision']
            first_matchapi_decision = message_from_first_batch[candidate_id]['first_matchapi_decision']
            second_matchapi_decision = candidate['decision']

            result['candidate_id'] = candidate_id
            result['candidate_name'] = candidate['candidate']['data']['name']
            result['content_decision'] = content_decision
            result['first_matchapi_decision'] = first_matchapi_decision
            result['second_matchapi_decision'] = second_matchapi_decision
            result['first_matchapi_details'] = message_from_first_batch[candidate_id]['first_matchapi_details']
            result['second_matchapi_details'] = candidate['details']

            weight_content_decision = get_decision_weight(content_decision)
            weight_first_matchapi_decision = get_decision_weight(first_matchapi_decision)
            weight_second_matchapi_decision = get_decision_weight(second_matchapi_decision)

            if weight_content_decision == 0 and weight_first_matchapi_decision != weight_second_matchapi_decision:
                result['status'] = 'undefined'
                decision_unchanged.append(result.copy())
            elif weight_first_matchapi_decision == weight_second_matchapi_decision:
                result['status'] = 'unchanged'
                decision_unchanged.append(result.copy())
            elif content_decision == Decision.yes.value and (
                    weight_second_matchapi_decision > weight_first_matchapi_decision):
                result['status'] = 'better'
                decision_better.append(result.copy())
            elif content_decision == Decision.no.value and (
                    weight_second_matchapi_decision < weight_first_matchapi_decision):
                result['status'] = 'better'
                decision_better.append(result.copy())
            else:
                result['status'] = 'worse'
                decision_worse.append(result.copy())

    combine_lists = decision_worse + decision_better + decision_unchanged
    return combine_lists


def remove_old_dump_files():
    [f.unlink() for f in Path(MATCHAPI_DUMP_TWO_DIR).glob("*.txt") if f.is_file()]


async def main():
    remove_old_dump_files()

    logging.basicConfig(
        datefmt='%Y-%m-%d %H:%M:%S',
        format="[%(asctime)s] %(levelname)-8s - %(message)s",
        level=logging.INFO,
    )

    extract_file = namedtuple("extracted_file", "decisions_file dump_file")

    files_final = [(extract_file(topic + '.only_final.tsv', topic + '.txt'), True) for topic in TOPIC_NAME_FINAL]
    file_other = [(extract_file(topic + '.tsv', topic + '.txt'), False) for topic in TOPIC_NAME]
    all_files = files_final + file_other

    for extract_files, final in all_files:
        logging.info(f"Processing files {extract_files.decisions_file} and {extract_files.dump_file}")
        await process_items(extract_files)
        compare_all_decisions(extract_files, final=final)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except asyncio.CancelledError:
        print("Main task cancelled")
