import os
import sys
import argparse

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vggt_ba.utils import JsonRetrieval
from reranking.reranking import LightGlueRanker


def parse_args():
    parser = argparse.ArgumentParser(description="Process image retrieval JSON files.")

    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to input JSON with image retrieval information"
    )

    parser.add_argument(
        "--feats_dir",
        type=str,
        required=True,
        help="Path to directory with local features of all images"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path of directory to output JSON file"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Processing file: {args.input_json}")

    retrieval_data = JsonRetrieval(args.input_json)
    reranker = LightGlueRanker(args.feats_dir)

    new_topk_column = []

    for query in tqdm(retrieval_data):
        query_path = query['query_path']
        ref_paths = query['retrieval']['ref_paths']
        results = reranker.rerank(
            query_path,
            ref_paths
        )
        permutation = results['permutation']
        new_scores = results['inliers_cnt']
        tws = query['retrieval']['ref_poses']
        new_topk_field = [
            {
                'path': ref_paths[idx],
                'rank': i+1,
                'score': new_scores[idx],
                'Tws': tws[idx]
            }
            for i, idx in enumerate(permutation)
        ]
        new_topk_column.append(new_topk_field)

    rerank_df = retrieval_data.df.drop(columns=['topk'])
    rerank_df['topk'] = new_topk_column
    os.makedirs(args.output_dir, exist_ok=True)
    json_fname = os.path.basename(args.input_json)
    output_fpath = os.path.join(args.output_dir, json_fname)

    print(f"Dumping results to {output_fpath}")
    rerank_df.to_json(output_fpath)


if __name__ == '__main__':
    main()
