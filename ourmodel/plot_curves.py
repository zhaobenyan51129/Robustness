import numpy as np
import sys
import matplotlib.pyplot as plt




















if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) < 12:
        print(sys.argv)
        raise Exception('not enough argument for getTuningCurve(output_suffix, res_suffix, conLGN_suffix, conV1_suffix, res_fdr, setup_fdr, data_fdr, fig_fdr, nOri, fitTC, fitDataReady)')