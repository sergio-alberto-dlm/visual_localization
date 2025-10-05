import sys, os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reranking.reranking import Extractor, extract_and_save
from utils.images_txt import get_all_imagestxt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract local features of all dataset images with SuperPoint"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset file or directory"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to dump predictions."
    )
    parser.add_argument(
        "--building",
        type=str,
        required=True,
        help="HYDRO or SUCCULENT."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dataset_path = os.path.abspath(args.dataset_path)
    dump_path = os.path.abspath(args.output_path)
    imgs_txts = get_all_imagestxt(input_dataset_path)
    imgs_txts = {
        k: v
        for k, v in imgs_txts.items()
        if args.building in k
    }
    print(f"All sessions to process: {imgs_txts}")

    for dataset_to_session_path, df_like in imgs_txts.items():
        print(f"Extracting local features of session {dataset_to_session_path}")
        session_dump_path = os.path.join(dump_path, dataset_to_session_path)
        os.makedirs(session_dump_path, exist_ok=True)
        image_to_feats_df = extract_and_save(
            data=df_like,
            dataset_path = input_dataset_path,
            root_dump_path = dump_path
        )
        output_csv_path = os.path.join(session_dump_path, 'local_feats.csv')
        print(f"Dumping local features into {output_csv_path}")
        image_to_feats_df.to_csv(output_csv_path)


if __name__ == '__main__':
    main()
