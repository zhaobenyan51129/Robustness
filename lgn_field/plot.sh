#!/bin/bash
source /home/zhaobenyan/miniconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate neuro
set -e


repo=/home/zhaobenyan/model/repos/patchV1 
cfg_fdr=/home/zhaobenyan/model/repos/patchV1/src
fdr0=/home/zhaobenyan/model
res_fdr=/home/zhaobenyan/model/resource 
setup_fdr=/home/zhaobenyan/model/setup 
data_fdr=/home/zhaobenyan/model/patchfast


trial_suffix=merge_test
fig_fdr=${fdr0}/${trial_suffix} # figures and configs
res_suffix=minimal-patch
LGN_V1_suffix=test_lgn
V1_connectome_suffix=test_v1
TF=8
ori=1
nOri=6

readNewSpike=True
usePrefData=False
collectMeanDataOnly=False 

fitTC=False
if [ "$fitTC" = True ]; then
	OPstatus=0
else
	OPstatus=1
fi

cp ${repo}/img_proc.py ${fig_fdr}

pid=""
# echo python ${fig_fdr}/plotLGN_response_${trial_suffix}.py ${trial_suffix}_${ori} ${LGN_V1_suffix} ${data_fdr} ${fig_fdr}
# python ${fig_fdr}/plotLGN_response_${trial_suffix}.py ${trial_suffix}_${ori} ${LGN_V1_suffix} ${data_fdr} ${fig_fdr} &
# pid+="${!} "

# echo python ${fig_fdr}/plotV1_response_${trial_suffix}.py ${trial_suffix} ${res_suffix} ${LGN_V1_suffix} ${V1_connectome_suffix} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr} ${TF} ${ori} ${nOri} ${readNewSpike} ${usePrefData} ${collectMeanDataOnly} ${OPstatus}
# python ${fig_fdr}/plotV1_response_${trial_suffix}.py ${trial_suffix} ${res_suffix} ${LGN_V1_suffix} ${V1_connectome_suffix} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr} ${TF} ${ori} ${nOri} ${readNewSpike} ${usePrefData} ${collectMeanDataOnly} ${OPstatus} &
# pid+="${!} "

echo python ${fig_fdr}/plotFrameOutput_${trial_suffix}.py ${trial_suffix}_${ori} ${res_suffix} ${LGN_V1_suffix} ${V1_connectome_suffix} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr}
python ${fig_fdr}/plotFrameOutput_${trial_suffix}.py ${trial_suffix}_${ori} ${res_suffix} ${LGN_V1_suffix} ${V1_connectome_suffix} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr}

wait $pid
date
conda activate base