import numpy as np


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))



def sine_rampdown(current, rampdown_length, initial_value, final_value,):
    """Sine rampdown from initial_value to final_value."""
    if rampdown_length == 0:
        return initial_value
    else:
        period = rampdown_length*2
        amplitude = (initial_value-final_value)/2
        vertical_shift=(initial_value+final_value)/2
        phase_shift=0
        cos_value = amplitude * np.cos(2 * np.pi * (current - phase_shift) / period) + vertical_shift
    return cos_value