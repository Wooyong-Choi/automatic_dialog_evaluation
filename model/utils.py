
def collate_fn(batch):
    return list(zip(*batch))