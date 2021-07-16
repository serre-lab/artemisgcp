from joblib import Parallel, delayed
import queue as Queue
import os
import time
import numpy as np
import argparse


TASKSET_CODES = {
    0: '0x3',
    1: '0xc',
    2: '0x30',
    3: '0xc0',
    4: '0x300',
    5: '0xc00',
    6: '0x3000',
    7: '0xc000'
}

'''
    TASKSET_CODES = {
            0:'0x30000',
            1:'0xc0000',
            2:'0x300000',
            3:'0xc00000',
    }
'''


def main(args):
    available_GPUs = [int(x) for x in list(args.gpus)]
    N_GPU = len(available_GPUs)
    video_list = args.vids

    with open(video_list, 'rb') as f:
        nvids = len(f.readlines())
    # 108000
    cmd = "CUDA_VISIBLE_DEVICES={} taskset={}  python /cifs/data/tserre/CLPS_Serre_Lab/projects/prj_nih/prj_andrew_holmes/inference/get_embedding_nopad.py \
       --video_name={} \
        --batch_size=128 \
        --first_how_many=108000 \
        --exp_name={} \
        --base_result_dir={} "

    # Put indices in queue
    q = Queue.Queue(maxsize=N_GPU)
    for i in range(N_GPU):
        q.put(available_GPUs[i])

    def runner(x):
        gpu = q.get()

        time.sleep(np.random.rand() / 2)
        while True:
            try:
                os.mknod(video_list + '.lock')
                break
            except (KeyboardInterrupt, SystemExit):
                raise
            except OSError:
                time.sleep(np.random.rand() / 2)
                #print('waiting on lock')

        with open(video_list, 'r+') as f:
            pos = f.tell()
            line = f.readline()[:-1]
            while line and line.startswith('R'):
                pos = f.tell()
                line = f.readline()[:-1]  # to avoid the escape
            if line != '':
                f.seek(pos)
                f.write('R' + line[1:])
        os.remove(video_list + '.lock')

        if line != '':
            #print(x, gpu, line)
            video_name = line.strip()
            exp_name = video_name.split('/')[-2]
            emb_name = video_name.split('/')[-1].replace('.mp4', '.p')
            processed_embs = []
            if os.path.exists(args.base_res_dir + exp_name):
                processed_embs = os.listdir(args.base_res_dir + exp_name)
            
            if emb_name not in processed_embs:
                #import pdb; pdb.set_trace()
                print(cmd.format(gpu, TASKSET_CODES[gpu], video_name, exp_name, args.base_res_dir))
                os.system(cmd.format(gpu, TASKSET_CODES[gpu], video_name, exp_name, args.base_res_dir))
        q.put(gpu)

    # Change loop
    Parallel(n_jobs=N_GPU, backend="threading")(
        delayed(runner)(i) for i in range(nvids))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs='+', required=True)
    parser.add_argument("--vids", type=str, required=True)
    parser.add_argument("--base_res_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)
