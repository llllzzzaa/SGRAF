import torch
from torch.utils.data import Sampler
from mmengine.dist import get_dist_info
from mmdet.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class SceneAwareSampler(Sampler):
    """
    Ensure each batch contains only one scene (Day or Night)
    """

    def __init__(self,
                 dataset,
                 batch_size,
                 shuffle=True,
                 seed=0):

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        self.rank, self.world_size = get_dist_info()

        # ----------------------------------
        # ★★★ CRITICAL FIX ★★★
        # make sure dataset is initialized
        # ----------------------------------
        if hasattr(dataset, 'full_init'):
            dataset.full_init()
        # ----------------------------------

        self.scene_to_indices = {}

        for idx in range(len(dataset)):
            data_info = dataset.get_data_info(idx)
            scene = int(data_info['scene'])
            self.scene_to_indices.setdefault(scene, []).append(idx)

        # remove small groups
        self.scene_to_indices = {
            k: v for k, v in self.scene_to_indices.items()
            if len(v) >= batch_size
        }

        if len(self.scene_to_indices) == 0:
            raise RuntimeError(
                'No scene has enough samples to form one batch!'
            )

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed)

        all_batches = []

        for indices in self.scene_to_indices.values():
            indices = torch.tensor(indices)

            if self.shuffle:
                indices = indices[torch.randperm(len(indices), generator=g)]

            indices = indices.tolist()

            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size:
                    all_batches.append(batch)

        if self.shuffle:
            all_batches = torch.tensor(all_batches)
            all_batches = all_batches[
                torch.randperm(len(all_batches), generator=g)
            ].tolist()

        # DDP split
        all_batches = all_batches[self.rank::self.world_size]

        for batch in all_batches:
            for idx in batch:
                yield idx

    def __len__(self):
        return sum(
            len(v) // self.batch_size
            for v in self.scene_to_indices.values()
        )
