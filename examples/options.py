import numpy as np
from enum import Enum


class MimoUpdaterType(Enum):
    FREQUENCYDOMAIN = 1
    TIMEDOMAIN = 2

class PhaseRec(Enum):
    INTERNAL = 1
    EXTERNAL = 2
    NONE = 3

class ECalc(Enum):
    LMS = 1
    SBD = 2
    MRD = 3
    CMA = 4

