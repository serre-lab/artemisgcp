export "CUDA_DEVICE_ORDER"="PCI_BUS_ID"
python jobshed2.py --gpus 0 --vids '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_nih/prj_andrew_holmes/inference/i3d_try_ccv' --base_res_dir '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_nih/prj_andrew_holmes/inference/inference_i3d/' #4 5 6 7
