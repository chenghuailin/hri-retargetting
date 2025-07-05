import os
import subprocess
from HRI_retarget import ROOT, DATA_ROOT
from tqdm import tqdm
import time 
from datetime import timedelta
from queue import Queue
from threading import Thread

# 文件夹路径
folder_path = os.path.join(DATA_ROOT,"motion/human/HumanML3D/new_joints")
starting_time = time.time()


# 遍历文件夹及其子文件夹
todo_files = []
task_queue = Queue()

for root, dirs, files in os.walk(folder_path):
    num_files = len(files)
    
    for idx, filename in tqdm(enumerate(files)):
        # 获取文件的完整路径
        file_path = os.path.join(root, filename)


        # 使用subprocess执行Shell命令
        # 示例：使用wc命令统计文件行数
        if "npy" in file_path:
            # result = subprocess.run(["python", os.path.join(ROOT,"retarget/beat_g1_inspirehands.py"), file_path], capture_output=True, text=True)
            # # 打印命令输出
            # print("Command output:")
            # print(result.stdout)

            # # 如果命令执行失败，打印错误信息
            # if result.returncode != 0:
            #     print("Error:", result.stderr)    
            todo_files.append(file_path) 
            task_queue.put((len(todo_files),file_path))

total_num = len(todo_files)

def worker(id):
    gpu_id = id % 8
    while not task_queue.empty():
        idx, file_path = task_queue.get()
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print("#"*50)
        print(f"Current Progress: {idx}/{total_num}", f" Elapsed time: {str(timedelta(seconds=time.time() - starting_time)).split('.')[0]}")
        # env["PYTHONPATH"] = os.pathsep.join([os.path.join(ROOT, "src"), os.path.join(ROOT, "utils")])
        try:
            cmd = f"~/miniconda3/envs/retarget/bin/python {os.path.join(ROOT,'retarget/humanml3d_g1_29.py')} {file_path}"
            print(cmd)
            process = subprocess.Popen([cmd], env=env, stdout=subprocess.PIPE, text=True, shell=True)

            # 实时读取输出
            while True:
                print(id)
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())

            # 获取返回值
            return_code = process.wait()
            print("Return code:", return_code)
        finally:
            task_queue.task_done()

threads = [Thread(target=worker, args=(i,)) for i in range(16)]
for t in threads:
    t.start()
for t in threads:
    t.join()    