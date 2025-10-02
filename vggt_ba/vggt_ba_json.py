import os
import json

from tqdm import tqdm
import pandas as pd

from vggt_ba import bundle_adjustment, CudaInference as Vggt
#FIXME. No module named vggt



CROCODL_DATASET = '/media/emmanuel/nvme_storage/datasets/crocodl_dataset'
if not os.path.exists(CROCODL_DATASET):
    raise RuntimeError("Please, adjust CROCODL_DATASET path")


def predict_from_json_entry(model, sample, k=5):
    query_path = sample['query_path']
    query_topk = sample['topk']
    query_topk = {d['rank']: d for d in query_topk}

    vggt_input_paths = [
        query_topk[i].get('abs_path')
        for i in range(1, k+1)
    ] + [query_path]
    ranks = [
        query_topk[i].get('rank')
        for i in range(1, k+1)
    ] + [0]
    score = [
        query_topk[i].get('score')
        for i in range(1, k+1)
    ] + [1.0]

    vggt_input_abspaths = [
        os.path.join(CROCODL_DATASET, img_path)
        for img_path in vggt_input_paths
    ]

    predictions = model.run(vggt_input_abspaths)
    predictions['retrieval_rank'] = ranks
    predictions['retrieval_score'] = score
    predictions['dataset_paths'] = vggt_input_paths
    predictions['image_names'] = [
        os.path.basename(path) for path in vggt_input_abspaths
    ]
    #ba_results = bundle_adjustment(predictions)    
    #predictions['intrinsic_ba'] = ba_results['intrinsic']
    #predictions['extrinsic_ba'] = ba_results['extrinsic']

    return query_path, predictions


def predict_from_json(model, json_path, **kwargs):
    if not os.path.exists(json_path):
        raise ValueError(f"{json_path} does not exist")
    df = pd.read_json(json_path, lines=True)
    data = [
        predict_from_json_entry(model, df.iloc[i], **kwargs)
        for i in tqdm(range(len(df)))
    ]
    keys_to_keep = {'extrinsic', 'intrinsic', 'retrieval_rank', 'retrieval_score', 'dataset_paths'}

    json_ready_data = {}
    for query_path, data_dict in data:
        filtered_data = {
            key: data_dict[key].tolist()  # Convert NumPy arrays to lists
            for key in keys_to_keep if key in data_dict
        }
        json_ready_data[query_path] = filtered_data

    # Save to JSON file
    with open('filtered_data.json', 'w') as f:
        json.dump(json_ready_data, f, indent=2)


def main():
    print("Loading vggt")
    vggt = Vggt()
    json_path = "/home/emmanuel/Desktop/crocodl_challenge/experiments/visual_localization/megaloc.json"
    print("Starting predicting")
    predictions = predict_from_json(vggt, json_path)
    print(predictions.keys())


if __name__ == '__main__':
    main()
