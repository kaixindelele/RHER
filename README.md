# RHER: (Ralay-HER)--A powerful variant of HER
The official code for the paper “[Relay Hindsight Experience Replay: Self-Guided Continual Reinforcement Learning for Sequential Object Manipulation Tasks with Sparse Rewards](https://arxiv.org/abs/2208.00843)”

## Natter:
Yesterday, I review the Reincarnating RL (https://agarwl.github.io/reincarnating_rl/), and found that Jump Start RL (JSRL) has the state-distribution problem when using the guide-policy, while our Self-Guided Exploration Strategy (SGES) does not. Because JSRL uses the guide-policy with a certain trajectory, then transfers to learning-policy, this combination has the state-distribution problem naturally~

Our SGES mixes guide-policy and learning-policy with the same probability so that they have the same state-distribution~

## 中文版碎碎念：2023-02-18-10am
最近看了《Dichotomy of Control: Separating What You Can Control from What You Cannot》，其实我文中所描述的一致非负奖励，也是其中的一种情况，只不过序列问题操作任务，如果按照这个描述，就太大了，论文中，还是以“夹爪无法改变物体的位置，而导致用HER反思后，会存在 隐式的非负稀疏奖励为切入点” 比较合适。借着这个机会，我想分享一下论文外的感触。

对于智能体的探索和学习，尤其是稀疏奖励下的效率问题，我从18年底就开始尝试了，到现在都四年多了，我自己对强化学习的research也算是一种长序列稀疏奖励过程。

在我长时间无法做出顶会顶刊的工作的时候，我也在反思，为什么我会赶个早集，却连口热乎的都吃不上？为何我的学习效率如此低下？

现在回想起来，一直陷入局部最“优”的我，直到21年，才有点变化。

就是我曾经花了很长的时间来复现著名的HER算法，这篇工作的方案是如此的简洁优雅，以至于后面出现了快两千的引用。先解释下HER算法，HER算法本质上是修改目标，让智能体有一个类似的“安慰奖”：虽然它没完成给定的任务，但如果一开始的目标，是它刚才完成的，那它岂不是可以认为已经成功了？

这听起来有点废话文学，但实际上，如果它完成任务的某些部分，下次任务真的给定了这个目标，它上次学到的是真的能用上！

所以回过来看，我之前探索的那些失败的经历也是有用。比如说，我花了2个月时间复现OpenAI Baseline版本的HER，因为我一直无法用Pytorch复现出和它一样的性能，所以我几乎尝试了每一个超参数和设置。在尝试批量调参的过程中，因为手动启动程序的效率太低了，因此将spinningup的MPI教程改成了能网格搜索超参的模式，教程如下：https://blog.csdn.net/hehedadaq/article/details/114685906。

虽然我在这么长的探索过程中，没有做出新颖的工作，但是在过程中，我发现了一个新问题，就是渲染过程中，我发现，每次FetchPush任务在探索前期，都会出现机械臂不知道贴近物块的情况，我当时最直观的想法就是，这个肯定会影响探索效率！ok，找到了问题，我开始第一步尝试，就是在探索过程中，让智能体先学接近，再学操作（等我做完整个实验，开始写文章，文献调研的时候，才知道，这是SHER这篇文章的处理方案），然后发现效果比不上，接近和操作一起学，探索的时候混合探索。也就是我自己的RHER方案原型。

跳出这个实验的细节，对于我自己来说，前面对HER的探索，对HER的理解，批量调参的技能，让我后面处理原版RHER算法直接推广到其他多物体操作任务无法work时，有了基础，在文章中的体现就是，我做了六七项设置的修改，终于发挥了RHER的潜力，达到了无模型强化学习在多物体操作任务上的sota.

过去的“失败”经验和技能，是会帮助后面的成长的，文章中是这样，文章外也是一样。

因为今天是一个周六的上午，所以我有时间再梳理下，对于稀疏奖励下的复杂序列任务。有几个点可以帮助智能体快速学习：

1. 设定一个循序渐进的目标，一开始就好高骛远是不合理的。文中的体现就是对任务的拆解，文外的体现就是给自己也设定任务清单。

2. 学习过程中，要找到哪些是自己可控的，哪些是环境或者其他智能体导致的。对自己可控的经验，你的学习才有效果。文中的体现就是，要降低INNR比例，让夹爪尽可能的影响物块，文外的体现就是，如果你的行为改变不了事件的发生，那么这个事件和你的因果性就很低，你的总结反思，都是低效率的。

3. 探索的过程中，有贵人相助，有专家导师的手把手指导，是快速通过前人已经探索明白的领域的捷径。文中是让复杂子任务的探索，受到已经学会的简单子任务的策略的引导。文外的也和那后理解，作为研究生的我，会在导师，师兄师姐的指导下，进行研究的入门，会阅读各位前辈专家的论文和代码以及博客。然后开始我自己的探索，总结和分享。

4. 还有一个点，是这篇文章没有讨论的，是我下一篇文章的内容，等有机会了再和大家分享。

至于第三点，我再讲一下，文中提到了一个自我引导探索。因为文章涉及的内容太多了，我自己的写作技能还没有达到驾轻就熟的程度，所以有段内容，没能优雅的加到introduction里面。
自我引导，需要回答两个问题，一个是如何评价一个策略是专家策略，另外一个是如何利用好专家策略？
现实生活中，专家，是需要有客观评价指标的，职业，学历，专业，文章，专利，突出的项目经历等等，需要有明确的评价体系，来衡量某个人在某个特定的领域是否是专家。
另外一个就是如何更好的利用专家的指导？其实切换到人的视角，应该说“专家如何指导，才会更加高效”更为礼貌。
睁眼看专家操作，是很难学会的；专家看着你的操作，偶尔点拨你一下，你学习效率肯定会变高；但最好的还是“手把手”指导，既能有自己的探索，在走偏的时候，又能有人及时规正。

这两个问题，对于RHER文章来说，我们讨论的是序列物体操作任务，它虽然本身是一类常见且通用的任务，但是它还是具有特殊性的，它可以非常好的切分成多个阶段。
因此我们可以很好的评估每个子策略的成功率，来衡量“专业程度”。
第二个问题就是，我们用的是混合探索，既能保证效率，又能保证策略不会出现离线数据分布不匹配的情况。

因此，整个RHER算法，可以获得极高的样本效率，完全可以作为常见序列操作任务的骨干算法，其中的自我引导探索，也可以推广到多智能体探索和分层强化中~

最后，希望这次投稿能够顺利~~


## 1. Abstract:
> Learning with sparse rewards remains a challenging problem in reinforcement learning (RL). Especially for sequential object manipulation tasks, the RL agent always receives negative rewards until completing all of the sub-tasks, which results in low exploration efficiency. To tackle the sample inefficiency for sparse reward sequential object manipulation tasks, we propose a novel self-guided continual RL framework, named Relay Hindsight Experience Replay (RHER). RHER decomposes the sequential task into several sub-tasks with increasing complexity and ensures that the simplest sub-task can be learned quickly by applying HER. Meanwhile, a multi-goal & multi-task network is designed to learn all sub-tasks simultaneously. In addition, a SelfGuided Exploration Strategy (SGES) is proposed to accelerate exploration. With SGES, the already learned sub-task policy will guide the agent to the states that are helpful to learn more complex sub-task with HER. Therefore, RHER can learn sparse reward sequential tasks efficiently stage by stage. The proposed RHER trains the agent in an end-to-end manner and is highly adaptable to avariousmanipulation tasks with sparse rewards. The experimental results demonstrate the superiority and high efficiency of RHER on a variety of single-object and multi-object manipulation tasks (e.g., ObstaclePush, DrawerBox, TStack, etc.). We perform a real robot experiment that agents learn how to accomplish a contact-rich push task from scratch. The results show that the success rate of the proposed method RHER reaches 10/10 with only 250 episodes.

## 2. Contributions:
(1) For common complex sequential object manipulation tasks with sparse rewards, this paper develops an elegant and sample efficient **self-guided continual RL framework**, RHER.

(2) To achieve self-guided exploration, we propose a **multi-goal & multi-task** network to learn multiple sub-tasks with different complexity simultaneously.

(3) The proposed RHER method is more sample-efficient than vanillaHER and other state-of-the-art methods, which are validated in the standard manipulation tasks from the OpenAI Gym. Further, to validate the versatility of RHER, we design eight sequential object manipulation tasks, including five complex multi-object tasks, which are available at this libary. The results show that the proposed RHER method consistently outperforms the vanilla-HER in terms of sample efficiency and performance.

(4) The proposed RHER learns a contact-rich task on a physical robot from scratch within 250 episodes in real world.


**I had release all codes for single-object tasks, if this paper is accepted, I will release the codes for multi-object tasks with the pytorch version immediately.**

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

![HER_INNR_ac](https://user-images.githubusercontent.com/28528386/218956856-837d2a52-8a8b-44e0-a1d8-11747dc24422.png)




Fig. 3. Illustration of the difference of HER and RHER. (a) The problem of Identical Non-Negative Rewards (INNR) with HER. (b) The proposed RHER solves the INNR problem by Self-Guided Exploration Strategy (SGES). (c) The **surprising** results of comparation of RHER and HER in FetchPush (If our codes is not open source, it may seem a bit outrageous~ Today, I read the controversy of the corpus indexer of NIPS and rethink our results. There should be no bug in my project, because the efficiency of the real machine is really high~).

## 6. A diagram of RHER:
![RHER_overall](https://user-images.githubusercontent.com/28528386/218956892-6a720c84-cd56-4cbb-8864-bc6d0fe6d3c9.png)


Fig4. A diagram of RHER, of which the key components are shown in the yellow rectangles. This framework achieves self-guided exploration for a sequential task.

### 6.1 A. Task Decomposition and Rearrangement
![RHER_task](https://user-images.githubusercontent.com/28528386/218956945-77d1eff5-c153-40d7-a536-a2f2a6505c73.png)


Fig5. Sequential task decomposition and rearrangement.

### 6.2 B. Multi-goal & Multi-task RL Model.
![RHER_goal_encoding](https://user-images.githubusercontent.com/28528386/218956993-c763ab25-da0e-4a74-95ad-5927da6d553a.png)



Fig6. Multi-goal & Multi-task RL Model.


### 6.3 C. Maximize the Use of All Data by HER.
1. In the RHER framework, updating a policy can not only use its own explored data but also relabel the data collected by other policies by HER. 

2. Coincidentally, for continual RL, the agent also needs to generate non-negative samples by HER.

### 6.4 D. Self-Guided Exploration Strategy (SGES)

**Like students for scientific research, who are guided by advisers and other researchers until they need to explore a new field.**

![RHER-SGES](https://user-images.githubusercontent.com/28528386/218957053-cb0c035a-aab1-4ffe-a0b2-f7723cae82e9.png)


Fig7. Illustration of Self-Guided Exploration Strategy (SGES) in a toy push task. The black solid curve represents actual trajectory with SGES.

### 6.5 E. Relay Policy Learning.
![RHER_relay](https://user-images.githubusercontent.com/28528386/218957081-12a0961d-4d50-4c8f-9776-d08c72db6627.png)


Fig8. A diagram of relay policy learning for a task with 3 stages. By using HER and SGES, RHER can solve the whole sequential task stage by stage with sample efficient. 

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

### 9.4 Testing process of TPush and TStack with Success Rate about 80%.

<details open="" class="details-reset border rounded-2">
  <summary class="px-3 py-2 border-bottom">
    <svg aria-hidden="true" viewBox="0 0 16 16" version="1.1" data-view-component="true" height="16" width="16" class="octicon octicon-device-camera-video">
    <path fill-rule="evenodd" d="..."></path>
</svg>
    <span aria-label="Video description TStack-RHER.mp4" class="m-1">RHER.mp4</span>
    <span class="dropdown-caret"></span>
  </summary>

  <video src="https://user-images.githubusercontent.com/28528386/200292389-ddd96b5b-e57a-42b9-bc16-ff893c6b3b8c.mp4" data-canonical-src="https://user-images.githubusercontent.com/28528386/200292389-ddd96b5b-e57a-42b9-bc16-ff893c6b3b8c.mp4" controls="controls" muted="muted" class="d-block rounded-bottom-2 width-fit" style="max-height:640px;">

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
