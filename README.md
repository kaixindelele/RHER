# RHER: (Ralay-HER)
The official code for paper “[Relay Hindsight Experience Replay: Continual Reinforcement Learning for Robot Manipulation Tasks with Sparse Rewards](https://arxiv.org/abs/2208.00843)”

~~If this paper is accepted, I will release the code for the pytorch version immediately, which can be easier applied for other manipulation tasks~~

This paper had been rejected by RAL becuase my poor writing ...

Now that I'm redescribing the whole story, I've expanded the RHER to N (N <= 3) blocks of manipulation tasks!

Although the mainstream tasks are soft robot and deformable object, my work provides an more effecient RL scheme for our community.

RHER is efficient and concise enough to be a new benchmark for the manipulation tasks with sparse rewards.

**Suitable tasks:**
Long-horizon manipulation tasks, in which both objects (Num <= 3) and goals are within the workspace of the robot.

**Unsuitable tasks:**
Stroke tasks: Slide, Tennis.


## Update!!! The multi-objects manipulation has been solved!

RHER can learn the stack task just within 300 (epoch) * 50 (episode) * 50 (step) = **750 k steps**, which means that RHER is the fastest model-free RL algorithm for these tasks.


<p float="middle">
  <img src="https://github.com/kaixindelele/RHER/blob/main/RHER.jpg" />
</p>

> HER has an implicit **virtual-positive** sparse reward problem caused by invariant achieved goals! 

> To solve this problem, RHER:
> 1) first decomposes and rearranges the original long-horizon task into new sub-tasks with incremental complexity.
> 2) design a multi-task network to learn the sub-tasks in ascending order of complexity. 
> 3) To solve the virtual-positive sparse reward problem, we propose a Random-Mixed Exploration Strategy (RMES), in which the achieved goals of the sub-task with higher complexity are quickly changed under the guidance of the one with lower complexity.

> Inspired by the idea of a relay, when a traveler needs to explore further, he/she needs to be escorted by some experts,
then he/she can quickly pass through the area that the expert is familiar with, and finally explore new areas by himself/herself.


> The experimental results indicate the significant improvements in sample efficiency of RHER compared to vanilla-HER in five typical robot manipulation tasks, including Push, PickAndPlace, Drawer, Insert, and ObstaclePush. The proposed RHER method has also been applied to learn a contact-rich push task on a physical robot from scratch, and the success rate reached 10/10 with only 250 episodes

## Other interesting motivation:
1. Don’t overambitious, agent need pay more attention to the goal which can be changed by itself.
2. One step at a time, gradually reach the distant goal.
3. Standing on the shoulders of giants, we can avoid many detours, just like scientific research.

## Some interesting experiments that don't have space to show in the article:

1. Why learn a reach policy alone, instead of directly designing a simpler P-controller?

a) I really did do a comparison experiment~ In the manipulation tasks without obstacle, the effect of P-controller is not much different from that of RHER, and some are even faster because it can also reach the object quickly. 

But P-controller is much worse than RHER in tasks with obstacle, because RHER has the ability to adapt to the environment.

b) And for the tasks of multiple blocks, especially PushTwoOjbect, it is difficult to design a base controller that can push object1 to the specified position and reach the vicinity of object2, but RHER can deal with it.


## Training process for stack.

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


## Training process for DrawerBox.

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

# Baselines
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
