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
one_ori=1

readNewSpike=True
usePrefData=False
collectMeanDataOnly=False 
plotOnly=False
fitTC=False
singleOri=True

if [ "$fitTC" = True ]; then
	OPstatus=0
else
	OPstatus=1
fi

if [ ! -d "$fig_fdr" ]; then
  mkdir -p $fig_fdr
fi

cp -r ${cfg_fdr}/patchTC ${fig_fdr}/patchTC_${trial_suffix}
cp -r ${cfg_fdr}/${patch_cfg} ${fig_fdr}/patch_${trial_suffix}.cfg

cp ${repo}/src/plotV1_response.py ${fig_fdr}/plotV1_response_${trial_suffix}.py
cp ${repo}/src/plotLGN_response.py ${fig_fdr}/plotLGN_response_${trial_suffix}.py
cp ${repo}/src/plotFrameOutput.py ${fig_fdr}/plotFrameOutput_${trial_suffix}.py
cp ${repo}/src/plotLGN_gallery.py ${fig_fdr}/plotLGN_gallery_${trial_suffix}.py
cp ${repo}/src/generate_input.py ${fig_fdr}/generate_input_${trial_suffix}.py

if [ "$singleOri" = False ]; then
	cp ${repo}/src/getTuningCurve.py ${fig_fdr}/getTuningCurve_${trial_suffix}.py
else
    echo one_ori=$one_ori
fi
cp ${repo}/img_proc.py ${fig_fdr}
cp ${cfg_fdr}/plotV1_response.py ${fig_fdr}
echo plotting files copied

export plotOnly
export nOri
export TF
export res_fdr
export setup_fdr
export data_fdr
export fig_fdr
export res_suffix
export trial_suffix
export LGN_V1_suffix
export V1_connectome_suffix
export fitTC
export collectMeanDataOnly
export fitDataReady 
export patch
export usePrefData
export readNewSpike
export singleOri
export plotOnly

plot_fig(){
	pid=""
	echo python ${fig_fdr}/plotLGN_response_${trial_suffix}.py ${trial_suffix}_${ori} ${LGN_V1_suffix} ${data_fdr} ${fig_fdr}
	python ${fig_fdr}/plotLGN_response_${trial_suffix}.py ${trial_suffix}_${ori} ${LGN_V1_suffix} ${data_fdr} ${fig_fdr} &
	pid+="${!} "

	echo python ${fig_fdr}/plotV1_response_${trial_suffix}.py ${trial_suffix} ${res_suffix} ${LGN_V1_suffix} ${V1_connectome_suffix} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr} ${TF} ${ori} ${nOri} ${readNewSpike} ${usePrefData} ${collectMeanDataOnly} ${OPstatus}
	python ${fig_fdr}/plotV1_response_${trial_suffix}.py ${trial_suffix} ${res_suffix} ${LGN_V1_suffix} ${V1_connectome_suffix} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr} ${TF} ${ori} ${nOri} ${readNewSpike} ${usePrefData} ${collectMeanDataOnly} ${OPstatus} &
	pid+="${!} "

	echo python ${fig_fdr}/plotFrameOutput_${trial_suffix}.py ${trial_suffix}_${ori} ${res_suffix} ${LGN_V1_suffix} ${V1_connectome_suffix} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr}
	python ${fig_fdr}/plotFrameOutput_${trial_suffix}.py ${trial_suffix}_${ori} ${res_suffix} ${LGN_V1_suffix} ${V1_connectome_suffix} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr} &
	pid+="${!} "
	wait $pid
}
date
conda activate base