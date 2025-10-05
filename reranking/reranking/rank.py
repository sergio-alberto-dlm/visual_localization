import os
from typing import List

from .matcher import Matcher
from .extractor import DumpedFeatures


class LightGlueRanker:
    def __init__(self, features_dir: os.PathLike, ransac_thresh: int=4):
        self.lightglue = Matcher(ransac_thresh)
        self.features_map = DumpedFeatures(features_dir)

    def rerank(
        self,
        query_img_path: os.PathLike,
        ref_img_paths: List[os.PathLike],
    ) -> dict:
        query_local_feats = self.features_map[query_img_path]
        inliers_cnt = [
            self.lightglue.get_inliers_count(
                query_local_feats,
                self.features_map[ref_img_path]
            )
            for ref_img_path in ref_img_paths
        ]

        ranks = sorted(
            range(len(inliers_cnt)),
            key=lambda i: inliers_cnt[i],
            reverse=True
        )

        print(ranks, inliers_cnt)
        return {
            'permutation': ranks,
            'inliers_cnt': inliers_cnt
        }
