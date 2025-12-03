#little exercise to understand the difference between using .apply and .transform
import pandas as pd
import numpy as np
df = pd.DataFrame({
    'group': ['A', 'A', 'A', 'B', 'B', 'B'],
    'value': [1, np.nan, 3, 10, np.nan, 30]
})

print(df)

print("\nUsing apply():")
result_apply = df.groupby('group')['value'].apply(lambda x: x.interpolate())
print(result_apply)
print("Result index:", result_apply.index)

# Interpolating per group with transform()
print("\nUsing transform():")
result_transform = df.groupby('group')['value'].transform(lambda x: x.interpolate())
print(result_transform)
print("Result index:", result_transform.index)

# Assign back to df (works only with transform)
df['interp_apply'] = result_apply  
df['interp_transform'] = result_transform 

print("\nFinal DataFrame after transform:")
print(df)

'''
what happens:
  group  value
0     A    1.0
1     A    NaN
2     A    3.0
3     B   10.0
4     B    NaN
5     B   30.0

Using apply():
2 groups (A and B) are created
it returns one object per group
problem: Pandas attaches to the result both the group name ('A' or 'B')
and the original row index (1 to 5) --> the result is labeled as MultiIndex

group   
A      0     1.0
       1     2.0
       2     3.0
B      3    10.0
       4    20.0
       5    30.0
Name: value, dtype: float64
Result index: MultiIndex([('A', 0),
            ('A', 1),
            ('A', 2),
            ('B', 3),
            ('B', 4),
            ('B', 5)],
           names=['group', None])

when trying to df['interp_apply'] = result_apply
pd finds NON COMPATIBILITY: "TypeError: incompatible index of inserted column with frame index"

Using transform():
0     1.0
1     2.0
2     3.0
3    10.0
4    20.0
5    30.0
Name: value, dtype: float64
Result index: RangeIndex(start=0, stop=6, step=1)
Traceback (most recent call last):
  File "/Users/giuliadirocco/opt/anaconda3/lib/python3.8/site-packages/pandas/core/frame.py", line 11610, in _reindex_for_setitem
    reindexed_value = value.reindex(index)._values
  File "/Users/giuliadirocco/opt/anaconda3/lib/python3.8/site-packages/pandas/core/series.py", line 4918, in reindex
    return super().reindex(
  File "/Users/giuliadirocco/opt/anaconda3/lib/python3.8/site-packages/pandas/core/generic.py", line 5360, in reindex
    return self._reindex_axes(
  File "/Users/giuliadirocco/opt/anaconda3/lib/python3.8/site-packages/pandas/core/generic.py", line 5375, in _reindex_axes
    new_index, indexer = ax.reindex(
  File "/Users/giuliadirocco/opt/anaconda3/lib/python3.8/site-packages/pandas/core/indexes/base.py", line 4279, in reindex
    target = self._wrap_reindex_result(target, indexer, preserve_names)
  File "/Users/giuliadirocco/opt/anaconda3/lib/python3.8/site-packages/pandas/core/indexes/multi.py", line 2490, in _wrap_reindex_result
    target = MultiIndex.from_tuples(target)
  File "/Users/giuliadirocco/opt/anaconda3/lib/python3.8/site-packages/pandas/core/indexes/multi.py", line 211, in new_meth
    return meth(self_or_cls, *args, **kwargs)
  File "/Users/giuliadirocco/opt/anaconda3/lib/python3.8/site-packages/pandas/core/indexes/multi.py", line 590, in from_tuples
    arrays = list(lib.tuples_to_object_array(tuples).T)
  File "pandas/_libs/lib.pyx", line 2894, in pandas._libs.lib.tuples_to_object_array
ValueError: Buffer dtype mismatch, expected 'Python object' but got 'long'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "apply_transform.py", line 23, in <module>
    df['interp_apply'] = result_apply  # ❌ will break or misalign
  File "/Users/giuliadirocco/opt/anaconda3/lib/python3.8/site-packages/pandas/core/frame.py", line 3950, in __setitem__
    self._set_item(key, value)
  File "/Users/giuliadirocco/opt/anaconda3/lib/python3.8/site-packages/pandas/core/frame.py", line 4143, in _set_item
    value = self._sanitize_column(value)
  File "/Users/giuliadirocco/opt/anaconda3/lib/python3.8/site-packages/pandas/core/frame.py", line 4867, in _sanitize_column
    return _reindex_for_setitem(Series(value), self.index)
  File "/Users/giuliadirocco/opt/anaconda3/lib/python3.8/site-packages/pandas/core/frame.py", line 11617, in _reindex_for_setitem
    raise TypeError(
TypeError: incompatible index of inserted column with frame index
'''