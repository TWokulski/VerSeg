import os
import re
import torch


def save_checkpoint(model, optimizer, epochs, checkpoint_path, **kwargs):
    checkpoint = {"model": model.state_dict(),
                  "optimizer": optimizer.state_dict(),
                  "epochs": epochs
                  }

    for k, v in kwargs.items():
        checkpoint[k] = v

    prefix, ext = os.path.splitext(checkpoint_path)
    checkpoint_path = "{}-{}{}".format(prefix, epochs, ext)
    torch.save(checkpoint, checkpoint_path)


def save_best(model, optimizer, epochs, model_path, **kwargs):
    best = {"model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epochs": epochs
            }

    for k, v in kwargs.items():
        best[k] = v

    prefix, ext = os.path.splitext(model_path)
    model_path = "{}-{}{}".format(prefix, epochs, ext)
    torch.save(best, model_path)


class CocoConversion:
    def __init__(self):
        self.buffer = []

    def write(self, s):
        self.buffer.append(s)

    def __str__(self):
        return "".join(self.buffer)

    def get_AP(self):
        txt = str(self)
        values = re.findall(r"(\d{3})\n", txt)
        values = [int(v) / 10 for v in values]
        result = {"bbox AP": values[0], "mask AP": values[11]}
        return result

    def get_AR(self):
        txt = str(self)
        values = re.findall(r"(\d{3})\n", txt)
        values = [int(v) / 10 for v in values]
        result = {"bbox AR": values[8], "mask AR": values[19]}
        return result