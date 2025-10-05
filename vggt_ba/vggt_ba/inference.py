import os
import gc
import pickle

import torch
import requests

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

from .load_fn import preprocess_images


device = "cuda"


class CudaInference:
    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available. Check your environment.")
        self.model = None
        self.load_model()

    def load_model(self) -> None:
        self.model = VGGT()
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        self.model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
        self.model.eval()
        self.model.to(device)

    def clear_model(self) -> None:
        del self.model
        self.model = None

    def run(self, img_path_list: list = None, img_list: list = None) -> dict:
        if self.model is None:
            raise ValueError("Model must be loaded first.")
        
        if img_path_list is not None and img_list is not None:
            raise ValueError("Only one of the following args must be passed: img_path_list, img_list")
        gc.collect()
        torch.cuda.empty_cache()

        if img_path_list is not None:
            images = load_and_preprocess_images(img_path_list).to(device)
        elif img_list is not None:
            images = preprocess_images(img_list).to(device)
        else:
            raise ValueError("At least one of the following args must be passed: img_path_list, img_list")

        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad():
            with torch.amp.autocast(device, dtype=dtype):
                predictions = self.model(images)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        # Clean up
        torch.cuda.empty_cache()

        # Convert tensors to numpy
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension

        depth_map = predictions["depth"]  # (S, H, W, 1)
        world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
        predictions["world_points_from_depth"] = world_points

        return predictions



class ApiInference:
    def __init__(self, url: str = 'http://127.0.0.1:8000') -> None:
        self.url = url

    def run(self, img_path_list: list = None, img_list: list = None) -> dict:
        if img_path_list is not None and img_list is not None:
            raise ValueError("Only one of the following args must be passed: img_path_list, img_list")
        if img_path_list is not None:
            payload = pickle.dumps(img_path_list) 
        elif img_list is not None:
            payload = pickle.dumps(img_list)
        else:
            raise ValueError("At least one of the following args must be passed: img_path_list, img_list")
        response = requests.post(
            self.url + "/predict",
            data = payload,
            headers={"Content-Type": "application/octet-stream"}
        )
        predictions = pickle.loads(response.content)
        return predictions
