"""Validation-loss-only stopping policy."""


class EarlyStopping:
    def __init__(self, patience=5):
        self.patience, self.best, self.bad_epochs = patience, float("inf"), 0

    def update(self, loss):
        improved = loss < self.best
        if improved:
            self.best, self.bad_epochs = loss, 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience
