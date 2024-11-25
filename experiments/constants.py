from experiments import curriculum as curriculum_lib
from models import rnn, transformer, universal_transformer
from tasks.regular import cycle_navigation, even_pairs, modular_arithmetic, parity_check

MODEL_BUILDERS = {
    "rnn": rnn.ElmanRNN,
    "transformer": transformer.Transformer,
    "universal_transformer": universal_transformer.UniversalTransformer,
    # "looped_transformer": transformer.LoopedTransformer,
    # adaptive
}

CURRICULUM_BUILDERS = {
    "fixed": curriculum_lib.FixedCurriculum,
    "regular_increase": curriculum_lib.RegularIncreaseCurriculum,
    "reverse_exponential": curriculum_lib.ReverseExponentialCurriculum,
    "uniform": curriculum_lib.UniformCurriculum,
}

TASK_BUILDERS = {
    "modular_arithmetic": modular_arithmetic.ModularArithmetic,
    "parity_check": parity_check.ParityCheck,
    "even_pairs": even_pairs.EvenPairs,
    "cycle_navigation": cycle_navigation.CycleNavigation,
}

TASK_LEVELS = {
    "modular_arithmetic": "regular",
    "parity_check": "regular",
    "even_pairs": "regular",
    "cycle_navigation": "regular",
}
