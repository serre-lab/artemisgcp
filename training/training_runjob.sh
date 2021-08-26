#!/usr/bin/env bash


export "CUDA_DEVICE_ORDER"="PCI_BUS_ID"

while getopts ":m:a:e:" opt; do
  case $opt in
    m)
      echo "-m was triggered, Parameter: $OPTARG" >&2
      model_uri=$OPTARG
      ;;
    a)
      echo "-a was triggered, Parameter: $OPTARG" >&2
      annotation_uri=$OPTARG
      ;;
    e)
      echo "-e was triggered, Parameter: $OPTARG" >&2
      embedding_uri=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done


python main_training.py --e ${embedding_uri} --a ${annotation_uri} --m ${model_uri} #--gpus 0 --vids '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_nih/prj_andrew_holmes/inference/i3d_try_ccv' --base_res_dir '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_nih/prj_andrew_holmes/inference/inference_i3d/'
