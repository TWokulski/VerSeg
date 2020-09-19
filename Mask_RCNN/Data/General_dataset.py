import os
import time
import torch
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor, as_completed


class GeneralizedDataset:

    def __init__(self, max_workers=1, verbose=False):
        self.max_workers = max_workers
        self.verbose = verbose
        self.ids = None
        self.aspect_ratios = None
        self.id_compare_fn = None

    def __getitem__(self, item):
        img_id = self.ids[item]
        image = self.get_image(img_id)
        image = transforms.ToTensor()(image)

        if self.train:
            target = self.get_target(img_id)
        else:
            target: {}

        return image, target

    def __len__(self):
        return len(self.ids)

    def check_dataset(self, file_to_be_checked):

        if os.path.exists(file_to_be_checked):
            info = [line.strip().split(", ") for line in open(file_to_be_checked)]
            self.ids, self.aspect_ratios = zip(*info)
            return

        start_time_for_checking = time.time()
        print("Checking the dataset...")

        tp_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        sequence_quality = torch.arange(len(self)).chunk(self.max_workers)
        tasks = [tp_executor.submit(self._check, sequence.tolist()) for sequence in sequence_quality]

        outs = []
        for future in as_completed(tasks):
            outs.extend(future.result())

        if not hasattr(self, "id_compare_fn"):
            self.id_compare_fn = lambda x: int(x)
        outs.sort(key=lambda x: self.id_compare_fn(x[0]))

        with open(file_to_be_checked, "w") as f:
            for img_id, aspect_ratio in outs:
                f.write("{}, {:.4f}\n".format(img_id, aspect_ratio))

        info = [line.strip().split(", ") for line in open(file_to_be_checked)]

        self.ids, self.aspect_ratios = zip(*info)
        print("checked id file: {}".format(file_to_be_checked))
        print("{} samples are OK; {:.1f} seconds".format(len(self), time.time() - start_time_for_checking))

    def _check(self, sequence):
        out = []
        for i in sequence:
            img_id = self.ids[i]
            target = self.get_target(img_id)

            boxes = target["boxes"]
            labels = target["labels"]
            masks = target["masks"]

            try:
                assert len(boxes) > 0, "{}: len(boxes) = 0".format(i)
                assert len(boxes) == len(labels), "{}: len(boxes) != len(labels)".format(i)
                assert len(boxes) == len(masks), "{}: len(boxes) != len(masks)".format(i)

                out.append((img_id, self._aspect_ratios[i]))
            except AssertionError as e:
                if self.verbose:
                    print(img_id, e)
        return out
