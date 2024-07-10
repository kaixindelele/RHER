import base64
from copy import deepcopy
import cloudpickle
import numpy as np
import os
import os.path as osp
import string
import subprocess
from subprocess import CalledProcessError
import sys
from textwrap import dedent
import time
import zlib

# 导入待执行的函数
from spinup_utils.mpi_tools import mpi_fork
from RHER_torch.train_torch import launch
from RHER_torch.torch_arguments import get_args

DIV_LINE_WIDTH = 80


def call_experiment(thunk, net, thunk_params_dict_list, args, cpu_num, **kwargs):
    """
        :params_dict thunk:待启动的函数
        :params_dict params_dict:批量参数名
        :params kwargs: 其他的一些没考虑到的参数~用处不大，没事儿最好别写这个,容易造成混乱~    
        正常的函数，传入参数之后，就会直接执行。
        但是通过这个神奇的lambda，就可以即把参数传进去，又不执行。返回出一个函数
        再次调用的时候，只需要将返回值，加上括号，即当一个无参数传入的函数执行就可以了。
    """
    def thunk_plus():
        # Fork into multiple processes
        mpi_fork(cpu_num)
        # Run thunk
        thunk(net, thunk_params_dict_list, args)
    # lambda封装会让tune_func.py中导入MPI模块报初始化错误。
    # thunk_plus = lambda: thunk(params_dict)
    # mpi_fork(len(params_dict))
    pickled_thunk = cloudpickle.dumps(thunk_plus)
    encoded_thunk = base64.b64encode(zlib.compress(pickled_thunk)).decode('utf-8')
    # 默认mpi_fork函数和run_entrypoint.py是在同一个文件夹spinup_utils，因此获取mpi的绝对路径
    # 如果不行的话，自己添加entrypoint的绝对路径就行
    base_path = mpi_fork.__code__.co_filename
    run_entrypoint_path = base_path.replace(base_path.split('/')[-1], '')
    entrypoint = osp.join(run_entrypoint_path, 'run_entrypoint.py')
    # entrypoint = osp.join(osp.abspath(osp.dirname(__file__)), 'run_entrypoint.py')
    
    # subprocess的输入就是一个字符串列表，正常在命令行，该怎么输入，这个就该怎么写。
    cmd = [sys.executable if sys.executable else 'python', entrypoint, encoded_thunk]
    print("tune_exps_pid:", os.getpid())
    try:
        subprocess.check_call(cmd, env=os.environ)
    except CalledProcessError:
        err_msg = '\n'*3 + '='*DIV_LINE_WIDTH + '\n' + dedent("""
            Check the traceback above to see what actually went wrong. 
            """) + '='*DIV_LINE_WIDTH + '\n'*3
        print("err_msg", err_msg)
        raise


if __name__ == '__main__':
    # 直接检查是什么主机
    import time
    # time.sleep(60*60*1)
    # 这里可以选你的GPU的序号,如果你是单机多卡的话.
    # gpus = [0, 2, 3]
    gpus = [0]
    # 这里确定你每个GPU下,要跑几个实验.
    cpu_num = 5
    gpu_id = 0
    # 100, 3stack 0.98 yes_targ/100 0.985/50 0.98/ 50 0.98 pd_no_tart./

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[gpu_id])

    params_dict = {
        # 随机种子
        "sd": [1000, 2000, 3000, 4000, 5000],
        # 每个episode的更次次数,默认是40
        "un": [40],
        # 接近阶段的距离阈值，默认是5厘米
        'ath': [0.05],
        # 操作的阈值，默认是5厘米
        'dth': [0.05],
        # 回合默认的done的类型，默认为0
        "done": [0],
        # 这块代码写的比较乱，下面这个参数是废除了的。在done=0的时候，控制奖励函数的阈值只有ath和dth
        "dish": [0],
        # 这个是策略的更新模式，q1表示loss_pi = -q1_pi.mean()
        'pm': ['q1'],
        # 下面是特殊阶段的目标重标记次数，如果不考虑的话，要和底下的ng一致。
        'nnn': [4],
        # 下面是经验池的大小
        'rf': [1e6],
        # 下面是mask的模式，2表示论文中提到的目标空间编码方式。也不用管
        'mask': [2],
        # 下面是目标重标记的次数，默认是4
        'ng': [4],
        # 正奖励的值，默认是0
        'rp': [0.0],
        # 负奖励的值，默认是-1
        'rn': [-1.0],
        # batch size的大小，默认是2048
        'bs': [2048],
        # 随机策略的采样概率，默认是0.2
        'rd': [0.2],
        # 当前策略的采样比例，默认是0.5，除去0.2的随机策略，也就是0.4的概率选用引导概率，0.4的概率选用当前策略
        # 在多物体实验中，这个参数不变，0.5就是稳定的。
        'rr': [0.5],
        # 衰减系数，默认是0.98
        'gamma': [0.98,],
        # 切换引导策略的 成功率阈值，默认是0.8，即当 某个阶段的测试成功率超过0.8时，就替换为引导策略
        'st': [0.8,],
        # 是否渲染，默认是0
        're': [0],
        # 是否策略蒸馏，默认不蒸馏
        'pd': [0],
        # 任务类型
        'env': ['FetchThreePush-v1'],
        # 'env': ['FetchThreePush-v1', 'FetchThreeStack-v1'],
        # 'env': ['FetchDoublePush-v1', 'FetchStack-v1'],
        # 'env': ['FetchPush-v1', 'FetchPickAndPlace-v1'],        
    }
    args = get_args()
    from RHER_torch.td3_per_her_bc import TD3Torch

    """
        done = 0: ours, only done = 1 if ag_index = dg.
        done = 1: all_done, done = 1 if r = rp
        done = 2: no_done, done = 0
    """
    # mpi_fork(cpu_num)
    import itertools
    # 将字典变为排列组合列表
    params_list = [list(value) for value in itertools.product(*params_dict.values())]
    # 将列表列表变为单个文件的字典列表
    params_dict_list = [{key: cur_param.pop(0) for key, value in params_dict.items()} for cur_param in params_list]
    print(params_dict_list)
    print("num_exps:", len(params_dict_list),
          "cycle_num:", len(params_dict_list) // cpu_num)
    # input("sure continue ?")
    # 每次传入cpu_num数个字典。
    batch_count = 0
    for i in range(0, len(params_dict_list), cpu_num):
        cur_params_dict_list = params_dict_list[i:i+cpu_num]
        if batch_count % len(gpus) == gpu_id:
            call_experiment(thunk=launch,
                            net=TD3Torch,
                            thunk_params_dict_list=cur_params_dict_list,
                            args=args,
                            cpu_num=cpu_num,
                            )

        batch_count += 1

