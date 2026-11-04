"""A61 exact context composition: main native/pairs retained, only multi replaced."""
import numpy as np


def align(sequences,columns,context):
    if any(len(c)!=len(sequences) for c in columns) or len({q['id'] for q in sequences})!=len(sequences):raise ValueError('complete unique same-axis source columns required')
    lookup={q['id']:(q,*[c[i] for c in columns]) for i,q in enumerate(sequences)}
    if len({q['id'] for q in context})!=len(context) or set(lookup)!={q['id'] for q in context}:raise ValueError('task/source sequence coverage differs')
    for q in context:
        source=lookup[q['id']][0]
        if source['kind']!=q['kind'] or not np.array_equal(source['labels'],q['labels']):raise ValueError('source and task original labels/kind differ')
    return tuple(map(list,zip(*(lookup[q['id']] for q in context))))


def compose(main,main_columns,source,source_columns,context):
    if len(main_columns)!=len(source_columns):raise ValueError('same complete source column schema required')
    kept=[i for i,q in enumerate(main) if q['kind']!='multi'];keep_ids={main[i]['id'] for i in kept}
    if keep_ids&{q['id'] for q in source} or any(q['kind']!='multi' for q in source):raise ValueError('only disjoint new multi axes may replace a task bank')
    qs=[main[i] for i in kept]+list(source);columns=tuple([m[i] for i in kept]+list(s) for m,s in zip(main_columns,source_columns))
    return align(qs,columns,context)
