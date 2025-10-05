import os
import sys
import argparse

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vggt_ba.vggt_ba.vggt_query_pose import VggtQueryPoseEstimation
from vggt_ba.utils import JsonRetrieval



def parse_args():
    parser = argparse.ArgumentParser(description="Process image retrieval JSON files.")

    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to input JSON with image retrieval information"
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to Crocodl dataset"
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

    print("Loading VGGT")
    pose_estimator = VggtQueryPoseEstimation()

    all_results = []
    for query in tqdm(retrieval_data):
        topk = query['retrieval']
        results = pose_estimator.get_query_pose(
            query['query_path'],
            topk['ref_paths'][:3],
            topk['ref_poses'][:3],
            args.dataset_path,
            sample_size = 3
        )
        all_results.append(results)

    query_pose_df = retrieval_data.df.drop(columns=['topk'])
    query_pose_df['pose_est'] = all_results
    os.makedirs(args.output_dir, exist_ok=True)
    json_fname = os.path.basename(args.input_json)
    output_fpath = os.path.join(args.output_dir, json_fname)

    print(f"Dumping results to {output_fpath}")
    query_pose_df.to_json(output_fpath)


if __name__ == '__main__':
    main()
