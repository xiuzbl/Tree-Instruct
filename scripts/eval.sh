port=5000
echo PORT: $port
now="$(date +'%m%d')"
echo DATE: $now

modeltype=openchat_v3.2
# modeltype=openchat
echo MODEL-TYPE: $modeltype
# echo Experiment $job

modelpath=your_saved_model_path
echo MODEL-PATH: $modelpath

log=./LLMOUT/treelogs/eval_$exp
env=./envs/ochatnew/bin

$env/python -m ochat.serving.openai_api_server \
    --model-type=$modeltype \
    --model $modelpath \
    --engine-use-ray --worker-use-ray 
    # --max-num-batched-tokens 2048