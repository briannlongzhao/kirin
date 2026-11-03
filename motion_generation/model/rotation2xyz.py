# This code is based on https://github.com/Mathux/ACTOR.git
class Rotation2xyz:
    """Animal motions are represented by joint positions, so no SMPL forward kinematics is needed."""

    def __init__(self, device, dataset=None):
        self.device = device
        self.dataset = dataset

    def __call__(self, x, mask, pose_rep, *args, **kwargs):
        if pose_rep != "xyz":
            raise NotImplementedError(f"Only xyz joint positions are supported, got pose_rep={pose_rep}")
        return x
