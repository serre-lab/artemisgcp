export "CUDA_DEVICE_ORDER"="PCI_BUS_ID"
python lstm_jobshed2.py --gpus 0 --vids '/cifs/data/tserre_lrs/projects/prj_nih/prj_andrew_holmes/inference/test_vids_fallon' --base_res_dir '/cifs/data/tserre_lrs/projects/prj_nih/prj_andrew_holmes/inference/results_fallon_bsln/' #4 5 6 7

