import time

class Logger:
    
    def __init__(self, project: str = "cs336", name: str = None, config: dict = None):
        import wandb
        self.wandb = wandb
        self.start_time = time.time()
        self.run = wandb.init(project=project, name=name, config=config)
    
    def log(self, step: int, metrics: dict):
        """Log metrics at a given step with wallclock time."""
        entry = {"step": step, "time": time.time() - self.start_time, **metrics}
        self.wandb.log(entry, step=step)
    
    def finish(self):
        """Finish the wandb run."""
        self.wandb.finish()