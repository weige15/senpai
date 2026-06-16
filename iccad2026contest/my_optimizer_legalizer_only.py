#!/usr/bin/env python3
"""Legalizer-only wrapper for my_optimizer.py."""

import my_optimizer as _full


class MyOptimizer(_full.MyOptimizer):
    def __init__(self, verbose: bool = False):
        _full.FloorplanOptimizer.__init__(self, verbose)
        self.device = _full.torch.device("cuda" if _full.torch.cuda.is_available() else "cpu")
        self.model = None
        self.checkpoint_loaded = False


Optimizer = MyOptimizer
