#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import pickle
from paq.evaluation.eval_utils import metric_max_over_ground_truths, exact_match_score
from paq.paq_utils import load_jsonl
from pprint import pp


def eval_retriever(refs, preds, hits_at_k, print_incorrect, export_calibraiton_path):
    incorrects = []
    hit_conf = []
    miss_conf = []
    res_score_pairs = []

    for k in hits_at_k:
        scores = []
        dont_print = False
        for r, p in zip(refs, preds):
            if hits_at_k[-1] > len(p['retrieved_qas']):
                print(f'Skipping hits@{k} eval as {k} is larger than number of retrieved results')
                dont_print = True
            ref_answers = r['answer']
            em = any([
                metric_max_over_ground_truths(exact_match_score, pred_answer['answer'][0], ref_answers)
                for pred_answer in p['retrieved_qas'][:k]
            ])
            scores.append(em)

            if em:
                hit_conf.append(p["retrieved_qas"][0]['score'])
            else:
                miss_conf.append(p["retrieved_qas"][0]['score'])

            if not em and print_incorrect:
                incorrects.append({
                    "input_qa": p["input_qa"],
                    "retrieved_qa": p["retrieved_qas"][0]
                })

            if export_calibraiton_path and k == 1:
                res_score_pairs.append((em, p["retrieved_qas"][0]['score']))

        if not dont_print:
            print(f'{k}: {100 * sum(scores) / len(scores):0.1f}% \n({sum(scores)} / {len(scores)})')

    if export_calibraiton_path:
        with open(export_calibraiton_path, 'wb') as f:
            pickle.dump(res_score_pairs, f)

    return incorrects, sum(hit_conf) / len(hit_conf), sum(miss_conf) / len(miss_conf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=str, help="path to retrieval results to eval, in PAQ's retrieved results jsonl format")
    parser.add_argument('--references', type=str, help="path to gold answers, in jsonl format")
    parser.add_argument('--hits_at_k', type=str, help='comma separated list of K to eval hits@k for', default="1,10,50")
    parser.add_argument('--print_incorrect', type=bool, help='whether to print incorrectly answers ans', default=False)
    parser.add_argument('--export_calibration', type=str, help='export calibraion report result as a pickle file')
    args = parser.parse_args()

    refs = load_jsonl(args.references)
    preds = load_jsonl(args.predictions)
    assert len(refs) == len(preds), "number of references doesnt match number of predictions"

    hits_at_k = sorted([int(k) for k in args.hits_at_k.split(',')])
    incorrects, hit_avg_score, miss_avg_score = eval_retriever(refs, preds, hits_at_k, args.print_incorrect, args.export_calibration)

    if args.print_incorrect:
        print("Below are incorrectly answers questions:")
        pp(incorrects)

    print(f"Hits' avrrage score: {hit_avg_score}")
    print(f"Misses' average score: {miss_avg_score}")
