"""Sample-weighted interval and epoch curves; retain raw losses in JSONL."""

from __future__ import annotations


class LossCurves:
    def __init__(self, mode="both", interval=25):
        if mode not in ("raw", "window", "epoch", "both") or interval < 1:
            raise ValueError("invalid loss curve mode or interval")
        self.mode, self.interval = mode, interval
        self.window, self.epoch = {}, {}
        self.window_steps = 0
        self.previous_window = None

    @staticmethod
    def add(bucket, losses, examples):
        for key, value in losses.items():
            total, weight = bucket.get(key, (0.0, 0))
            bucket[key] = (total + float(value) * examples, weight + examples)

    @staticmethod
    def means(bucket):
        return {key + "_mean": total / weight for key, (total, weight) in bucket.items()}

    def update(self, run, losses, step, examples, *, epoch_end=False, epoch_number=0, final=False):
        self.add(self.window, losses, examples)
        self.add(self.epoch, losses, examples)
        self.window_steps += 1
        full_window = self.window_steps >= self.interval
        if full_window or final:
            values = self.means(self.window)
            if self.mode in ("window", "both"):
                values["updates_in_window"] = self.window_steps
                if self.previous_window is not None:
                    values["loss/total_delta"] = values["loss/total_mean"] - self.previous_window
                run.log(values, step, "train_window/" if full_window else "train_window_partial/")
            if full_window:
                self.previous_window = values.get("loss/total_mean")
                self.window, self.window_steps = {}, 0
        if epoch_end or final:
            if self.mode in ("epoch", "both"):
                run.log(
                    {
                        **self.means(self.epoch),
                        "last_update": step,
                        "complete_epoch": int(epoch_end),
                    },
                    epoch_number,
                    "train_epoch/" if epoch_end else "train_epoch_partial/",
                )
            if epoch_end:
                self.epoch = {}

    def state_dict(self):
        return {
            "window": self.window,
            "epoch": self.epoch,
            "window_steps": self.window_steps,
            "previous_window": self.previous_window,
        }

    def load_state_dict(self, state):
        for key in ("window", "epoch", "window_steps", "previous_window"):
            setattr(self, key, state[key])
