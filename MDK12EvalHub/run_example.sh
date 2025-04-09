#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1
# export AUTO_SPLIT=1
export LMUData=/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/zhangkaipeng-24043/zkp/MDK12mini-medium
# turn USE_VLLM to 1 if you want to use vLLM for faster inference
export USE_VLLM=0
export VLLM_WORKER_MULTIPROC_METHOD=fork
# export VLLM_WORKER_MULTIPROC_METHOD=spawn

# match all available data with S(School) in the name
data_list=()
for file in "$LMUData"/*.tsv; do
    data=$(basename "$file" .tsv)
    case "$data" in
        *S*)
            data_list+=("$data")
            # break
            ;;
        *)
            echo "Skipping data: $data"
            ;;
    esac
done

echo "Processing data: ${data_list[@]}"
python run.py --data "${data_list[@]}" --model Eureka --verbose --mode infer --reuse --work-dir outputs_temp

# master_port=39991
# torchrun --master_port=$master_port --nproc-per-node=2 \
#   run.py --data "${data_list[@]}" \
#   --model Eureka --verbose --mode infer --reuse --work-dir outputs_temp
