"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

from datetime import datetime
import os
import utils
import wandb


class Logger:
    def __init__(self, variant):

        self.no_wandb = variant["no_wandb"]
        self.log_path = self.create_log_path(variant)
        utils.mkdir(self.log_path)
        print(f"Experiment log path: {self.log_path}")

    def log_metrics(self, outputs, iter_num, total_transitions_sampled):
        print("=" * 80)
        print(f"Iteration {iter_num}")
        outputs["evaluation/total_transitions_sampled"] = total_transitions_sampled
        for k, v in outputs.items():
            print(f"{k}: {v}")

        if not self.no_wandb:
            wandb.log(outputs, step=iter_num, commit=True)

    def create_log_path(self, variant):
        now = datetime.now().strftime("%Y.%m.%d/%H%M%S")
        exp_name = variant["exp_name"]
        prefix = variant["save_dir"]
        seed = variant["seed"]
        return f"{prefix}/{now}-{exp_name}/{seed}"
