# RHER: (Ralay-HER)--A revolutionary variant of HER!
The official code for paper “[Relay Hindsight Experience Replay: Self-Guided Continual Reinforcement Learning for Sequential Object Manipulation Tasks with Sparse Rewards](https://arxiv.org/abs/2208.00843)”

## 1. Abstract:
> Exploration with sparse rewards remains a challenging research problem in reinforcement learning (RL). Especially for sequential object manipulation tasks, the RL agent always receives negative rewards until completing all sub-tasks, which results in low exploration efficiency. To solve these tasks efficiently, we propose a novel **self-guided continual RL** framework, Relay-HER (RHER). RHER first decomposes a sequential task into new sub-tasks with increasing complexity and ensures that the simplest sub-task can be learned quickly by utilizing Hindsight Experience Replay (HER). Secondly, we design a multi-goal & multi-task network to learn these sub-tasks simultaneously. Finally, we propose a Self-Guided Exploration Strategy (SGES). With SGES, the learned sub-task policy will guide the agent to the states that are helpful to learn more complex sub-task with HER. By this self-guided exploration and relay policy learning, RHER can solve these sequential tasks efficiently stage by stage. The experimental results show that RHER significantly outperforms vanilla-HER in sample-efficiency on five singleobject and five complex multi-object manipulation tasks (e.g., Push, Insert, ObstaclePush, Stack, TStack, etc.). The proposed RHER has also been applied to learn a contact-rich push task on a physical robot from scratch, and the success rate reached 10/10 with only 250 episodes.

## 2. Contributions:
(1) For common complex sequential object manipulation tasks with sparse rewards, this paper develops an elegant and sample efficient **self-guided continual RL framework**, RHER.

(2) To achieve self-guided exploration, we propose a **multigoal & multi-task** network to learn multiple sub-tasks with different complexity simultaneously.

(3) The proposed RHER method is more sample-efficient than vanilla-HER and state-of-the-art methods, which are validated in the standard manipulation tasks from the OpenAI Gym;

(4) To verify that the RHER is suitable for common sequential object manipulation tasks, we conduct three extra typical single-object tasks, five more complex multi-object tasks, and even a physical robot task.


I had release all codes for single-object tasks, if this paper is accepted, I will release the code for the pytorch version immediately.

-----

Although the mainstream tasks are soft robot and deformable object, my work provides an more effecient RL scheme for RL-Robotics community.

RHER is efficient and concise enough to be a new benchmark for the manipulation tasks with sparse rewards.

## 3. Suitable tasks:
Complex sequential object manipulation tasks, in which both objects (Num <= 3) and goals are within the workspace of the robot.

![RHER_multi_obj](https://user-images.githubusercontent.com/28528386/199898455-aa75683a-6803-4101-a48b-11425c924aae.png)

Fig1. Multi-object tasks graphs.


![Fig_multi_obj](https://user-images.githubusercontent.com/28528386/199915337-a5649596-fd22-40a4-a027-fed6ccb35342.png)

Fig2. Learning curve of multi-object tasks.

Unsuitable tasks:
Stroke tasks: Slide, Tennis.

-----

## 4. Motivation:
HER works for simple reach tasks, but faces low sample efficient for manipulation tasks.
![image](https://user-images.githubusercontent.com/28528386/200155407-c5461a1f-ef55-4f97-8537-bab87af11d8b.png)

Each epoch means 19 * 2 * 50 = 1900 episodes!

Reported in 'Multi-Goal Reinforcement Learning: Challenging Robotics Environments and Request for Research'

I found an implicit problem of HER:

## 5. HER introduces an implicit **non-negative** sparse reward problem for manipulation tasks

HER has an implicit **non-negative** sparse reward problem caused by indentical achieved goals! 
![HER_nnsr](https://user-images.githubusercontent.com/28528386/200154197-02e6ca8a-a16d-4a1e-a60b-ca1fd40600d4.png)

Fig. 3. Illustration of the problem of non-negative sparse rewards with HER. For a typical sequential task, push task, the agent fails to push the object
to the desired position, and even fails to change the object position. So all original rewards are -1, and all hindsight rewards are 0, the latter can also be regarded as a kind of sparse reward problem, but with non-negative rewards.

## 6. A diagram of RHER:
![RHER_overall](https://user-images.githubusercontent.com/28528386/200154505-0c295992-9794-40dc-98da-cb482ff65c08.png)

Fig4. A diagram of RHER, of which the key components are shown in the yellow rectangles. This framework achieves self-guided exploration for a sequential task.

### 6.1 A. Task Decomposition and Rearrangement
![RHER_task](https://user-images.githubusercontent.com/28528386/200154536-f60bae8b-98ad-45e8-9314-2b628552e90a.png)

Fig5. Sequential task decomposition and rearrangement.

### 6.2 B. Multi-goal & Multi-task RL Model.
![RHER_goal_encoding](https://user-images.githubusercontent.com/28528386/200154666-3f5cdd74-36df-45c9-b2ea-99f9ab1ea1b0.png)

Fig6. Multi-goal & Multi-task RL Model.


### 6.3 C. Maximize the Use of All Data by HER.
1. In the RHER framework, updating a policy can not only use its own explored data but also relabel the data collected by other policies by HER. 

2. Coincidentally, for continual RL, the agent also needs to generate non-negative samples by HER.

### 6.4 D. Self-Guided Exploration Strategy (SGES)
![RHER-SGES](https://user-images.githubusercontent.com/28528386/200154765-f9610d50-f392-436a-853c-4ec6ce5fcb5d.png)

Fig7. Illustration of Self-Guided Exploration Strategy (SGES) in a toy push task. The black solid curve represents actual trajectory with SGES.

### 6.5 E. Relay Policy Learning.
![RHER_relay](https://user-images.githubusercontent.com/28528386/199898834-72cd34df-c00c-48c3-9cef-0afb4d0946c2.png)

Fig8. A diagram of relay policy learning for a task with 3 stages. By using HER and SGES, RHER can solve the whole sequential task stage by stage with sample efficient. 

**Like students for scientific research, who are guided by advisers and other researchers until they need to explore a new field.**

## 7. Other interesting motivation:
1. Don’t overambitious, agent need pay more attention to the goal which can be changed by itself.
2. One step at a time, gradually reach the distant goal.
3. Standing on the shoulders of giants, we can avoid many detours, just like scientific research.


## 8. Some interesting experiments that don't have space to show in the article:

1. Why learn a reach policy alone, instead of directly designing a simpler P-controller?

a) I really did do a comparison experiment~ In the manipulation tasks without obstacle, the effect of P-controller is not much different from that of RHER, and some are even faster because it can also reach the object quickly. 

But P-controller is much worse than RHER in tasks with obstacle, because RHER has the ability to adapt to the environment.

b) As for the tasks of multiple blocks, especially DPush, it is difficult to design a base controller that can push object1 to the specified position and reach the vicinity of object2, but RHER can deal with it.

## 9. Training Videos:
### 9.1 Training process for stack.

<details open="" class="details-reset border rounded-2">
  <summary class="px-3 py-2 border-bottom">
    <svg aria-hidden="true" viewBox="0 0 16 16" version="1.1" data-view-component="true" height="16" width="16" class="octicon octicon-device-camera-video">
    <path fill-rule="evenodd" d="..."></path>
</svg>
    <span aria-label="Video description Stack-RHER.mp4" class="m-1">RHER.mp4</span>
    <span class="dropdown-caret"></span>
  </summary>

  <video src="https://user-images.githubusercontent.com/28528386/192075197-11b1b6b1-3991-45da-ab75-4bed0cf10b54.mp4" data-canonical-src="https://user-images.githubusercontent.com/28528386/192075197-11b1b6b1-3991-45da-ab75-4bed0cf10b54.mp4" controls="controls" muted="muted" class="d-block rounded-bottom-2 width-fit" style="max-height:640px;">

  </video>
</details>


### 9.2 Training process for DrawerBox.

<details open="" class="details-reset border rounded-2">
  <summary class="px-3 py-2 border-bottom">
    <svg aria-hidden="true" viewBox="0 0 16 16" version="1.1" data-view-component="true" height="16" width="16" class="octicon octicon-device-camera-video">
    <path fill-rule="evenodd" d="..."></path>
</svg>
    <span aria-label="Video description Stack-RHER.mp4" class="m-1">RHER.mp4</span>
    <span class="dropdown-caret"></span>
  </summary>

  <video src="https://user-images.githubusercontent.com/28528386/193175188-b09d57cc-44c5-4609-9356-91bcbf2ba503.mp4" data-canonical-src="https://user-images.githubusercontent.com/28528386/193175188-b09d57cc-44c5-4609-9356-91bcbf2ba503.mp4" controls="controls" muted="muted" class="d-block rounded-bottom-2 width-fit" style="max-height:640px;">

  </video>
</details>

### 9.3 Training process for Real World Task.
<details open="" class="details-reset border rounded-2">
  <summary class="px-3 py-2 border-bottom">
    <svg aria-hidden="true" viewBox="0 0 16 16" version="1.1" data-view-component="true" height="16" width="16" class="octicon octicon-device-camera-video">
    <path fill-rule="evenodd" d="..."></path>
</svg>
    <span aria-label="Video description RHER.mp4" class="m-1">RHER.mp4</span>
    <span class="dropdown-caret"></span>
  </summary>

  <video src="https://user-images.githubusercontent.com/28528386/180405215-7410531f-01f3-41cf-bdae-808b896fb778.mp4" data-canonical-src="https://user-images.githubusercontent.com/28528386/180405215-7410531f-01f3-41cf-bdae-808b896fb778.mp4" controls="controls" muted="muted" class="d-block rounded-bottom-2 width-fit" style="max-height:640px;">

  </video>
</details>

## 11. Reproduce:

### Baselines
Our baselines is based on [OpenAI baselines](https://github.com/openai/baselines), and gym is based on [OpenAI gym](https://github.com/openai/gym/tree/0.18.0)

OpenAI Baselines is a set of high-quality implementations of reinforcement learning algorithms.

These algorithms will make it easier for the research community to replicate, refine, and identify new ideas, and will create good baselines to build research on top of. Our DQN implementation and its variants are roughly on par with the scores in published papers. We expect they will be used as a base around which new ideas can be added, and as a tool for comparing a new approach against existing ones. 

## Prerequisites 
Baselines requires python3 (>=3.5) with the development headers. You'll also need system packages CMake, OpenMPI and zlib. Those can be installed as follows
### Ubuntu 
    
```bash
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
```

## Virtual environment
```bash
conda create -n rher python=3.6
```

## Tensorflow versions
The master branch supports Tensorflow 1.14.

## Installation
- Clone the repo and cd into it:
    ```bash
    git clone https://github.com/kaixindelele/RHER.git
    cd RHER
    ```
- If you don't have TensorFlow installed already, install your favourite flavor of TensorFlow. In most cases, you may use
    ```bash 
    conda install tensorflow-gpu==1.14 # if you have a CUDA-compatible gpu and proper drivers
    ```
    
    and
    
    ```bash 
    pip install -r requirement.txt
    ```
    

### MuJoCo (200)
Some of the baselines examples use [MuJoCo](http://www.mujoco.org) (multi-joint dynamics in contact) physics simulator, which is proprietary and requires binaries and a license (license can be obtained from [mujoco-free-license](https://github.com/kaixindelele/RHER/blob/main/gym/mjkey.txt)

### MuJoCo-py (2.0.2.1)
Instructions on setting up MuJoCo can be found [mujoco-py(2.0.2.1)](https://github.com/openai/mujoco-py/tree/v2.0.2.1)

## Training models
run in terminal
```bash
bash run_rher_push.sh
```

or
run in pycharm
```bash
python -m baselines.run_rher_np1.py
```
