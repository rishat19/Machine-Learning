from enum import IntEnum

DataProcessTypes = IntEnum('DataProcessTypes', ('standardization', 'normalization'))
SetType = IntEnum('SetType', ('train', 'valid', 'test'))
TaskTypes = IntEnum('TaskTypes', ('classification', 'regression'))
