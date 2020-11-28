import re
import torch


def save_checkpoint(model, optimizer, epochs, checkpoint_path, **kwargs):
    checkpoint = {"model": model.state_dict(),
                  "optimizer": optimizer.state_dict(),
                  "epochs": epochs
                  }

    for k, v in kwargs.items():
        checkpoint[k] = v

    checkpoint_path = "{}-{}".format(checkpoint_path, epochs)
    torch.save(checkpoint, checkpoint_path)


def save_best(model, optimizer, epochs, model_path, **kwargs):
    best = {"model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epochs": epochs
            }
    for k, v in kwargs.items():
        best[k] = v

    model_path = "{}-{}{}".format((model_path + 'best'), epochs, '.pth')
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
        result = {"bbox AP": values[0], "mask AP": values[12]}
        return result

    def get_AR(self):
        txt = str(self)
        values = re.findall(r"(\d{3})\n", txt)
        values = [int(v) / 10 for v in values]
        result = {"bbox AR": values[8], "mask AR": values[20]}
        return result

    def get_AF1(self):
        txt = str(self)
        values = re.findall(r"(\d{3})\n", txt)
        values = [int(v) / 10 for v in values]
        if (values[0] + values[8]) == 0.0:
            dice_box = 0.0
        else:
            dice_box = 2 * (values[0] * values[8]) / (values[0] + values[8])

        if (values[12] + values[20]) == 0.0:
            dice_mask = 0.0
        else:
            dice_mask = 2 * (values[12] * values[19]) / (values[12] + values[20])
        dice_box = float(format(dice_box, ".2f"))
        dice_mask = float(format(dice_mask, ".2f"))
        result = {"bbox F1Score": dice_box, "mask F1Score": dice_mask}
        return result
