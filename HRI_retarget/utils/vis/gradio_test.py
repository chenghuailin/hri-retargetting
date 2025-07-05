import sys 
import os 
os.environ["all_proxy"] = ''
os.environ["ALL_PROXY"] = ''

import gradio as gr
import numpy as np
import requests
import threading
import multiprocessing
import time

# from deploy_g1 import deploy_g1

is_running = False
thread_pool = []

def start_fn():
    global is_running
    is_running = True
    return

def stop_fn():
    global is_running
    is_running = False
    return

def is_running_fn():
    global is_running, thread_pool
    t = None 
    idx = 0
    while True:
        idx += 1
        if t == None:
            if is_running:
                t = multiprocessing.Process(target=test_run)
                t.start()
        elif not is_running:
            t.terminate()
            del t 
            t = None
        time.sleep(0.05)

        if idx % 20 == 0:
            print(is_running, t, idx)
        
def test_run():
    while True:
        print("i am running")
        time.sleep(0.2)


main_t = threading.Thread(target=is_running_fn, daemon=True)
main_t.start()


with gr.Blocks() as demo:
    start_button = gr.Button("Start")
    stop_button = gr.Button("Stop", scale=3)


    start_button.click(
        fn=start_fn,
    )
    stop_button.click(
        fn=stop_fn,
    )


demo.launch(share=False, server_name="0.0.0.0", server_port=7990)


