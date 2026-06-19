"""Paths and radar/signal constants."""
import os

# Data roots — place the (non-redistributable) dataset next to the repo, or
# override via the RHB_TRAIN_ROOT / TEST_ROOT environment variables.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

TRAIN_ROOT = os.environ.get("RHB_TRAIN_ROOT", os.path.join(_REPO, "..", "RHB_train"))
TEST_ROOT = os.environ.get("TEST_ROOT", os.path.join(_REPO, "..", "track2_testdata"))

# FMCW radar parameters
SAMP_F = 5e6
FREQ_SLOPE = 60.012e12
ADC_SAMPLES = 256

# Heart-rate band (bpm) and PPG sampling rate
HR_LO, HR_HI = 45, 150
PPG_FS = 30
