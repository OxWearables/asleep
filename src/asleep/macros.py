# for capture24 label conversion
SLEEPNET_LABELS = {
    -1: -1,
    0: 0,
    1: 0,
    2: 0,
    3: 1,
}

# for predictions
SLEEPNET_BINARY_LABELS = {
    -1: 'non_wear',
    0: 'wake',
    1: 'sleep',
    2: 'sleep',
    3: 'sleep',
    4: 'sleep',
}

SLEEPNET_THRE_CLASS_LABELS = {
    -1: 'non_wear',
    0: 'wake',
    1: 'NREM',
    2: 'NREM',
    3: 'NREM',
    4: 'REM',
}
