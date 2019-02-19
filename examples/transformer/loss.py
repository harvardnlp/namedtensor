
@dataclass
class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__():

    generator : Mod
    criterion : Mod
    opt : Mod
    schedule : Mod

    def __call__(self, x, y, norm):
        print(x.shape, y.shape)
        x = self.generator(x)
        print(x.shape, y.shape)
        einassert("btv,bt", (x, y))
        loss = self.criterion((x.contiguous().view(-1, x.size(-1))),
                              (y.float().contiguous().view(-1))) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.schedule.step()
            self.opt.zero_grad()
        return loss.data.item() * norm
