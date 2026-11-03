"""Generate a T-pose mesh from one image with the Rodin (Hyper3D) API."""

import os
import time

import numpy as np
import requests
from PIL import Image

BASE_URL = "https://api.hyper3d.com/api/v2"
API_KEY_ENV = "RODIN_API_KEY"


def apply_mask(rgb_path, mask_path, out_path, mask_threshold=127):
    """Save the image as RGBA with the background outside the mask transparent."""
    rgb = Image.open(rgb_path).convert("RGBA")
    mask = Image.open(mask_path).convert("L")
    if rgb.size != mask.size:
        mask = mask.resize(rgb.size, resample=Image.NEAREST)
    rgba = np.array(rgb)
    rgba[..., 3] = (np.array(mask) > mask_threshold).astype(np.uint8) * 255
    Image.fromarray(rgba).save(out_path)


def _api_key():
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise RuntimeError(f"Set {API_KEY_ENV} to your Rodin API key to generate a mesh.")
    return key


def _post(endpoint, api_key, **kwargs):
    response = requests.post(f"{BASE_URL}/{endpoint}", headers={"Authorization": f"Bearer {api_key}"},
                             timeout=120, **kwargs)
    response.raise_for_status()
    return response.json()


def generate_mesh(image_path, output_dir, poll_interval=5, timeout=1800):
    """Submit an image to Rodin and download the T-pose mesh (base.obj and textures) to output_dir."""
    api_key = _api_key()
    with open(image_path, "rb") as f:
        files = {"images": (os.path.basename(image_path), f.read(), "image/png")}
    data = {
        "tier": "Regular",
        "TAPose": True,
        "geometry_file_format": "obj",
        "material": "PBR",
        "mesh_mode": "Raw",
        "use_original_alpha": True,
    }
    task = _post("rodin", api_key, files=files, data=data)
    if "uuid" not in task:
        raise RuntimeError(f"Rodin task submission failed: {task}")
    task_uuid, subscription_key = task["uuid"], task["jobs"]["subscription_key"]
    print(f"Submitted Rodin task {task_uuid}")

    start = time.time()
    while True:
        jobs = _post("status", api_key, json={"subscription_key": subscription_key})["jobs"]
        if jobs and all(job["status"] in ("Done", "Failed") for job in jobs):
            break
        if time.time() - start > timeout:
            raise TimeoutError(f"Rodin task {task_uuid} did not finish within {timeout} s")
        time.sleep(poll_interval)
    if any(job["status"] == "Failed" for job in jobs):
        raise RuntimeError(f"Rodin task {task_uuid} failed: {jobs}")

    os.makedirs(output_dir, exist_ok=True)
    for item in _post("download", api_key, json={"task_uuid": task_uuid})["list"]:
        dest = os.path.join(output_dir, item["name"])
        response = requests.get(item["url"], timeout=300)
        response.raise_for_status()
        with open(dest, "wb") as f:
            f.write(response.content)
        print(f"Downloaded {dest}")
    if not os.path.isfile(os.path.join(output_dir, "base.obj")):
        raise RuntimeError(f"Rodin result in {output_dir} has no base.obj")
