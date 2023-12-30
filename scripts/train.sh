port=5000
NUM_GPUS=8
echo PORT: $port
echo number of GPUs: $NUM_GPUS
now="$(date +'%m%d')"
echo DATE: $now

btz=2 #for alpaca-data
#btz=16 #for openchat-data
# epoch=20
epoch=10
# epoch=3
# epoch=1
# epoch=4
save_every=5
nodenum=0
# nodenum=6
# nodenum=3
# nodenum=10
trainnum=1k
# trainnum=3k
# trainnum=4k
job=wizard${trainnum}_llama2_gpu${NUM_GPUS}_epo${epoch}_btz${btz}_run1
# job=tree${trainnum}_${nodenum}nodes_llama2_gpu${NUM_GPUS}_epo${epoch}_btz${btz}_run0
# job=alpaca${trainnum}_llama2_gpu${NUM_GPUS}_epo${epoch}_btz${btz}_run0
# job=tree${trainnum}_mix3level_llama2_gpu${NUM_GPUS}_epo${epoch}_btz${btz}_run0
# job=tree_curr_easy_llama2_gpu${NUM_GPUS}_epo${epoch}_btz${btz}
# job=tree_curr_mid_llama2_gpu${NUM_GPUS}_epo${epoch}_btz${btz}
# job=tree_curr_hard_llama2_gpu${NUM_GPUS}_epo${epoch}_btz${btz}
# job=tree_reversecurr_hard_llama2_gpu${NUM_GPUS}_epo${epoch}_btz${btz}
# job=tree_reversecurr_easy_llama2_gpu${NUM_GPUS}_epo${epoch}_btz${btz}
# job=tree_openchat_llama2_epo5_run0
# job=openchat6k_llama2_epo5_run0
exp=${now}_$job #!

echo Experiment: $exp

BASE_MODEL_PATH=./MODELS/LLaMA2_13B_with_EOT_token 
datapath=./DATA/tree1k_3nodes
datapath=$datapath/tree
echo DATA: $datapath
echo MODEL: $BASE_MODEL_PATH

TARGET_FOLDER=./LLMOUT/llama2_treeouts/${exp}
log=./LLMOUT/llama2_treelogs/$exp
mkdir -p $TARGET_FOLDER

env=./envs/ochatnew/bin
$env/wandb login your_wandb_key

CUDA_LAUNCH_BLOCKING=1 \
$env/deepspeed  --num_gpus=$NUM_GPUS --module ochat.training_deepspeed.train \
    --model_type openchat_v3.2 \
    --model_path $BASE_MODEL_PATH \
    --data_path $datapath \
    --save_path $TARGET_FOLDER \
    --save_every $save_every \
    --epochs=$epoch \
    --batch_size_per_gpu=$btz \
    --deepspeed \
    --deepspeed_config ochat/training_deepspeed/deepspeed_config.json \
    >>$log.log 2>>${log}a.log
