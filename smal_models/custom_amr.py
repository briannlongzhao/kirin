import torch
import pickle
from typing import Dict
from pathlib import Path

from AniMer.amr.models.amr import AMR
from AniMer.amr.configs import get_config
from AniMer.amr.models.smal_warapper import SMALLayer
from AniMer.amr.utils.geometry import perspective_projection


def load_amr(checkpoint_path):
    model_cfg = str(Path(checkpoint_path).parent.parent / '.hydra/config.yaml')
    model_cfg = get_config(model_cfg, update_cachedir=True)

    # Override paths for SMAL
    model_cfg.defrost()
    base_path = Path(checkpoint_path).parent.parent.parent.parent
    model_cfg.SMAL.MODEL_PATH = str(base_path / model_cfg.SMAL.MODEL_PATH)
    model_cfg.SMAL.SHAPE_PRIOR_PATH = str(base_path / model_cfg.SMAL.SHAPE_PRIOR_PATH)
    model_cfg.SMAL.POSE_PRIOR_PATH = str(base_path / model_cfg.SMAL.POSE_PRIOR_PATH)
    model_cfg.freeze()

    # Override some config values, to crop bbox correctly
    if (model_cfg.MODEL.BACKBONE.TYPE == 'vit') and ('BBOX_SHAPE' not in model_cfg.MODEL):
        model_cfg.defrost()
        assert model_cfg.MODEL.IMAGE_SIZE == 256, f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()
    if (model_cfg.MODEL.BACKBONE.TYPE == 'dinov2') and ('BBOX_SHAPE' not in model_cfg.MODEL):
        model_cfg.defrost()
        assert model.cfg.MODEL.IMAGE_SIZE == 252, f"MODEL.IMAGE_SIZE ({model_cfg.MODEL_IMAGE_SIZE}) should be 252 for dino backbone"
        model_cfg.MODEL_BBOX_SHAPE = [252, 252]
        model_cfg.freeze()

    # Update config to be compatible with demo
    if ('PRETRAINED_WEIGHTS' in model_cfg.MODEL.BACKBONE):
        model_cfg.defrost()
        model_cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS')
        model_cfg.freeze()

    model = CustomAMR.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg, map_location='cpu')
    return model, model_cfg


class CustomAMR(AMR):
    """Custom AMR class that returns SMAL keypoints instead of AniMer keypoints"""
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        smal_model_path = cfg.SMAL.MODEL_PATH
        with open(smal_model_path, 'rb') as f:
            smal_cfg = pickle.load(f, encoding="latin1")
        # Do not use SMAL class which returns AniMer keypoints
        self.smal = SMALLayer(**smal_cfg)

    
    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        """
        Same as AMR.forward_step, but returns only necessary outputs for SMAL keypoints.
        """
        x = batch['img']
        batch_size = x.shape[0]
        conditioning_feats, cls = self.backbone(x[:, :, :, 32:-32])  # [256, 192]
        pred_smal_params, pred_cam, _ = self.smal_head(conditioning_feats)
        output = {}
        output['cls_token'] = cls
        output['cls_feats'] = self.class_token_head(cls)
        output['pred_cam'] = pred_cam
        output['pred_smal_params'] = {k: v.clone() for k, v in pred_smal_params.items()}
        focal_length = batch['focal_length']
        pred_cam_t = torch.stack([
            pred_cam[:, 1], pred_cam[:, 2], 2 * focal_length[:, 0] / (self.cfg.MODEL.IMAGE_SIZE * pred_cam[:, 0] + 1e-9)
        ], dim=-1)
        output['pred_cam_t'] = pred_cam_t
        output['focal_length'] = focal_length
        pred_smal_params['global_orient'] = pred_smal_params['global_orient'].reshape(batch_size, -1, 3, 3)
        pred_smal_params['pose'] = pred_smal_params['pose'].reshape(batch_size, -1, 3, 3)
        pred_smal_params['betas'] = pred_smal_params['betas'].reshape(batch_size, -1)
        smal_output = self.smal(**pred_smal_params, pose2rot=False)
        pred_keypoints_3d = smal_output.joints
        pred_vertices = smal_output.vertices
        output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, -1, 3)
        output['pred_vertices'] = pred_vertices.reshape(batch_size, -1, 3)
        pred_cam_t = pred_cam_t.reshape(-1, 3)
        focal_length = focal_length.reshape(-1, 2)
        pred_keypoints_2d = perspective_projection(
            pred_keypoints_3d, translation=pred_cam_t, focal_length=focal_length / self.cfg.MODEL.IMAGE_SIZE
        )
        output['pred_keypoints_2d'] = pred_keypoints_2d.reshape(batch_size, -1, 2)
        return output
    