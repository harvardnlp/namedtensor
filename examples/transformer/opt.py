from torch.optim.lr_scheduler import _LRScheduler

class NoamSchedule(_LRScheduler):
    def __init__(self, opt, model_size, warmup):
        self.warmup = warmup
        self.model_size = model_size
        super(NoamSchedule, self).__init__(opt, -1)

    def get_lr(self):
        step = self.last_epoch + 1
        return [base_lr * (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))
                for base_lr in self.base_lrs]

def std_schedule(model):
    opt = torch.optim.Adam(model.parameters(), lr=2, betas=(0.9, 0.98), eps=1e-9)
    schedule = NoamSchedule(opt, model.src_embed[1].d_model, 4000)
    return opt, schedule
