"""Scheduling only: preserve all frozen rows and prioritize fixed primary contrasts."""
def primary_first(rows,joint):
    ranks={(seed,arm):i for i,(seed,arm) in enumerate((s,a) for s in [2026110257,2026110258] for a in ['F',joint])}
    def key(item):
        index,row=item
        return (ranks[(row['seed'],row['arm'])],index) if row['kind']=='fixed_step' and row['step']==6000 and (row['seed'],row['arm']) in ranks else (len(ranks),index)
    return [row for _,row in sorted(enumerate(rows),key=key)]
