port=5000
echo PORT: $port
now="$(date +'%m%d')"
echo DATE: $now
modeltype=openchat_v3.2
# modeltype=$4

maxnum=805
# maxnum=10
temperature=0.7
# max_out_token=600

modelpath=your_saved_modelpath
job=your_experiment_name

echo JOB: $job
echo MAX-NUM: $maxnum
# echo MAX-OUTPUT-TOKEN: $max_out_token
echo MODEL-PATH: $modelpath

echo Begin generating answers
$env/python apiinfer.py \
    --max_num=$maxnum \
    --output_file $modelpath \
    --model_type $modeltype  \
    --exp=$job
    # --max_out_token=$max_out_token \
    # --model_path $modelpath \

echo Finish generating answers
