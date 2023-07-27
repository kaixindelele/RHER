# RHER: (Ralay-HER)--A Powerful Variant of HER
The official code for the paper “[Relay Hindsight Experience Replay: Self-Guided Continual Reinforcement Learning for Sequential Object Manipulation Tasks with Sparse Rewards](https://arxiv.org/abs/2208.00843)”

论文最新版在仓库的[RHER_git.pdf](https://github.com/kaixindelele/RHER/blob/main/RHER_Git.pdf)中可以查看。

前一版本的中文版可以在[RHER_old_中文版](https://github.com/kaixindelele/RHER/blob/main/RHER-old-%E4%B8%AD%E6%96%87%E7%89%88.pdf)查看。

欢迎**引用**和讨论细节。

💥💥💥<strong> 7.24. 本文已被Neurocomputing接收，感谢一路以来所有帮助和支持的朋友和老师！
</strong>


💥💥💥<strong> 7.10. 
It is noteworthy that in a recent work, [RoMo-HER](https://arxiv.org/abs/2306.16061), based on the RHER, has combined model-based schemes to further improve sample efficiency in the classic tasks of FetchPush and FetchPickAndPlace.

Moreover, it can be seen from the paper that the authors have **independently reproduced the performance of our open-source code**.

Lastly, based on the experimental results, even with the addition of model-based approaches, the improvement in sample efficiency is still limited, demonstrating that RHER is indeed highly efficient in these two standard tasks.

 </strong>


 # Starchart

[![Star History Chart](https://api.star-history.com/svg?repos=kaixindelele/RHER&type=Date)](https://star-history.com/#kaixindelele/RHER&Date)


 ![~FAYHELW24F4F{G57ZBT}GX](https://github.com/kaixindelele/RHER/assets/28528386/86fb2a22-412f-4ebd-9364-780424997646)


 没想到，真的有同学，基于RHER做算法改进！
 
 当初说在readme说，希望RHER能成为新的benchmark，好像隐约能看到点希望了。

 作为一个延毕的博士生，看到这篇工作能够得到别人的认可，以及对领域的微小帮助，还是有点莫名的感慨。
 
 最后希望RHER中一些有意思的操作，可以对其他领域有一些启发，比如多智能体的合作，比如逆强化学习，比如“我的世界”这种复杂序列任务的探索，比如基于大语言模型的引导探索。

 愿世界美好
 

## 最新中文版碎碎念－RHER对我的反向指导：2023-06-23-00am
<details><summary><code>查看最新中文版碎碎念</code></summary>
  综合来看RHER对我的反向指导有以下几点：
  １.　目标可以是远大的，但要从够一够能够的着的子目标开始努力，这样会事半功倍，效率比较高
  ２.　对于有人擅长的领域，自己不熟悉的领域，要学会寻求专家智能体的手把手教学，这样效率极高
  ３.　技能、经验、资源和认知的积累，对你完成复杂任务来说，也是一个必要的过程
  ４.　在自己探索复杂任务前，要充分意识到目标和自身能力中间的差距，学会切分子任务，学会寻求专家智能体的帮助和指导。
  
  最近熟悉的朋友们都顺利毕业了，而我还在等这篇工作被正式接收，才能毕业，夜深时刻，记录一下最近的感悟。
  
  读博是我自己的一个重大抉择，2018年底的时候，我认为决策，是真正的智能，而当时强化刚开始兴起。
  
  我愿意花四年时间去探索这个领域，希望能在博士期间，做出点拓宽人类认知边界的工作。
  
  现在已经四年整了，我其实还是在和自己的认知作斗争。
  
  我的研究内容其实是我自己研究过程的一个验证。
  
  比如说，我在受到导师指导的过程中发现，刚开始的时候，导师的经验是非常靠谱的，但随着我学习的过程，导师的领域知识已经逐渐OOD了，那么接下来每次组会的详细指导，都可能不会有更好的探索结果。
  
  那么我就提出了我的第一篇有意思的工作，dense2sparse，先dense奖励函数，手把手的教学，然后再sparse奖励函数进行策略提高。
  
  哈哈，这个Motivation是我的论文里面无法说的，希望导师看到了，应该也能理解。
  
  反过来，我的研究，对我自己的学习和生活，也能起到一个反向的指导作用。  
  
  比如说RHER里面隐含的一些道理，在我做ChatPaper的时候，就有很多体现。
  
  从2017年的时候，我就希望能够建立一个学术论文翻译平台，希望各位作者能够将自己写的论文，翻译成ta的母语版本，方便本国广大的普通工程师和低年级，或者跨学科的其他研究者阅读。
  
  但是在当时，对我这个智能体来说，这是一个无法实现的目标。
  
  我无论做什么，都没法实现它，甚至于，我的行为对结果都不会有显著的影响。
  
  一般来说，我不是一个双标的人，希望别人付出的同时，我一般也会要求我自己先做到。

  我希望大家一起翻译英文论文，那就从我开始做起。
  
  而我只能做我能做的事情，我坚持在知乎和CSDN上分享我自己的阅读笔记，分享我自己的学习教程。  
  
  我也坚持开源，一直维护DRLib开源仓库，为社区提供一套比较好用的强化画图的脚本，一套比较好用的HER的复现代码。
  
  在这个过程中，我自己的能力也有了少许提升，我在社区也有了少许的影响力。
  
  等到2023年3月1号，我把手头上的两篇小论文投出去之后，我开始学习LLM的知识，一方面感慨新知识的爆炸，另外一方面也发现ChatGPT的API为了我的目标提供了一个新的机会，我可以利用Chat来做英文论文的总结！
  
  现在的形势变了，不仅环境变了，我也变了。
  
  环境提供了一个新机会，我自己的知识、技能、认知、资源也有了提高。
  
  我还把目标从全文翻译，降低到了全文总结。我的技术栈从编程小白，变成了稍微懂些爬虫，数据处理，并发的小白。
  
  因此我从3.5号开始开发ChatPaper的原型代码，在GPT和朋友的帮助下，3.8号基本上就做出来了。
  
  经过三四年的科研训练，我知道论文的总结，需要哪些关键的内容，因此我可以写出一些比较靠谱的prompts。
  
  第一次看到GPT总结出来的论文效果，让我感到十分的惊艳。
  
  我维护的强化QQ群群友的反馈，让我感觉到，也许这是一个大家都需要的工具。
  
  发到知乎上，知乎的关注者的反馈也非常好。
  
  当时类似的工具只有chatpdf，且他们并没有针对学术做优化，也无法全文总结。
  
  因此，我感觉我做出了一个比较有用的工具。
  
  经过4天的使用和优化，我于3.12号开源到GitHub上。
  
  从这一刻开始，我体会到了环境快速反馈，且dense奖励的快乐了。
  
  因为我之前的社区，以及知乎和B站三年积累下来的关注者，帮我做了第一轮的流量推广。
  
  3.13号，HuggingFace的AK大佬转发了ChatPaper。
  
  多方帮助下，前两天项目的star数迅速上升到了一两百。
  
  3.15号，不知道是哪位大佬的关照，上了GitHub的热榜，直到3.18号。
  
  这时候star数指数级上升，直奔1K star。
  
  读博四年，我第一次充分的感受到工作被认可的成就感，虽然不是我的学术成果。

  之后的发展就非常有意思了，作为第一个将Chat应用到科研领域的开源项目，有非常多的同学帮助我一起维护这个项目。
  
  然后我开始尝试带团队，但很明显，我的技能、认知和经验，已经不足以处理这种突然出现的新任务了。
  
  但智能体 智能 的体现在于和环境的交互和学习。
  
  而我的策略则是，偏保守的探索，虽然现在回过头来看，我可能错过了一些机会，或者有一些决策是非最优的。
  
  但没有办法，人生无法reset，我只有一次探索机会，我的认知不够的情况下，我只能做出那样的决策。

  现在已经过去三个月了，经过这三个月的探索和学习，我现在的认知和技能又有了一些提升，如果下次再有类似的机会，我想，我应该会做的更好。这也是RHER里面，自我引导持续强化学习 的一个体现。
  
  再回到最初的目标，17年的时候，我就想着，能够把最新的英文学术成果，翻译成中文。
  
  现在是我距离这个目标最近的一次。

  我有了一个比较火的开源项目，让我有机会认识更多厉害的人，也会有更多的人认可我，愿意帮助我。

  比如说学术版GPT的作者，他做了一个arXiv的全文翻译的功能。

  我第一次看到这个功能的时候，我就说，他创建了一个学术“巴别塔”。

  我知道，17年的目标，大概是可以够的着了，如果把那个目标当成是desired goal的话，现在应该已经接近最后一个stage了。

  在得到qingxu的支持下，以及黄老师的帮助下，我们终于将这个功能上线到了chatpaper.org上。

  现在距离我们最后的目标，把所有的优质论文翻译展示到首页上，形成一个非常好的文献阅读社区，还有一定的距离。

  但是我相信，我们正在step by step的去实现它。

  我相信我们现在做的这项工作是有价值的，世界上那么多非英语母语的工程师、低年级的学生、其他领域的研究者，他们如果可以用母语去阅读最新的科技论文时，是可以极大的拓宽大家的认知的。
  
  而语言这个门槛如果存在的话，在实现{学习最新科技进展}目标的过程中，凭空多了一个gap，或者用RHER中的概念，就是多了一个学习英语的stage，否则根本无法完成这个任务。

  RHER解决的是RL agent在复杂序列任务中探索和学习效率的问题，我一直希望能将这个工作拓展到其他领域。

  前段时间有同学将这个工作拓展到了Model-based架构中，可惜因为写作原因，第一次投稿被拒了。

  我也想过其他的领域，比如多智能体的合作探索，但目前的强化仿真任务都很难切分子任务和离散的任务阶段，除了物体操作任务之外，也就我的世界比较合适，但暂时我对我的世界的环境还不熟悉，其他感兴趣的同学可以试试这个结合。

  经过这么长时间的反思，我发现，最符合的还是我们人类的日常任务，很多任务都是复杂序列任务，前置任务没完成，是无法解决最终目标的。而人类场景的一些任务，让传统的强化去完成，可能比较困难，后面需要让LLM结合自我引导的一些概念，去实现才比较合适。

  PhD本意是哲学博士，底层的方法论确实是有异曲同工之妙。我现在非常希望RHER这篇工作能被顺利接受，这样我就可以写我的博士大论文了，我想我的博士论文应该是对社区有帮助的。
  
  GPT4的出现，一度让我感到焦虑和对人类未来感到担忧，我担心一个认知和决策能力超过多数人类的模型的出现，会让很多人成为GPT的附庸，听从GPT的决策。

  经过这三个月的交互和反馈，我还是坚持当时开源ChatPaper时的努力，我希望我和伙伴们的工作，能够帮助我们人类自己在AI快速迭代的时代中，也能和AI一起进化。
  
  不管是AI4Science，还是其他，我希望AI4Human。

  为了实现这样的目标，后面还有很多东西要做，还有很多坑需要探索。

  好在每一次遇到我不会的问题，都能得到对应domain的expert agent的guidance，让我得以快速解决问题且学会新的技能，从而去完成更加有挑战性的工作。
  
  希望在这个有限的，仅有一次的生命中，在这个极具变化的时代中，做出点微小的贡献，希望在我擅长的领域，可以帮助到其他的learning agent。

  太晚了，思路零散，想到什么说什么，先发出来，和各位共勉。
  
  
  
  
  
</details>

We express our gratitude for the expert guidance! 

With the advice of the expert, we evaluated the cost of our solution in detail. After all, there is no free lunch, but by comparison, our solution only needs to pay a small price, which can greatly reduce the learning time of the whole task. It can be easily calculated from the table that in the multi-object tasks, the memory and computation time have a simple linear relationship with the number of objects, and the linear increase coefficient is very low.

![image](https://github.com/kaixindelele/RHER/assets/28528386/2fb27eb7-12db-4f9f-8679-24c33dada3dc)

In addition, with the advice of experts, we investigated more than 40 paper on multi-task reinforcement learning in the past five years, and found that our zero-padding encoding also has certain promotion value in the field of multi-task reinforcement learning, especially in the field of robot manipulation tasks, dynamic objects and goal tasks. RHER's framework really suits the new backbone in the field of robot manipulation tasks.

Even in the era of LLMs, the idea of self-guided exploration can also enable LLms-based methods to achieve a stable exploration. We're trying it out, so stay tuned!






💥💥💥<strong> 4/12/2023. RHER vs SOTA HER-based method, EBP, based on [Energy-Based Hindsight Experience Prioritization](https://github.com/ruizhaogit/EnergyBasedPrioritization)
</details>
</strong>

![RHER_SOTA](https://user-images.githubusercontent.com/28528386/231344134-d5a46362-afb8-42c0-8e18-ce8c16ba8960.png)

Under the more realistic single CPU core setting, although the EBP algorithm has achieved a great improvement in sample efficiency compared with HER, it still has obvious disadvantages compared with our RHER algorithm due to the lack of self-guided exploration. All experiments were conducted with the same five random seeds (1000-5000) and the hyperparameter clip_energy of EBP is 0.5.

## Natter 2023-02-18-10am:
Yesterday, I review the Reincarnating RL (https://agarwl.github.io/reincarnating_rl/), and found that Jump Start RL (JSRL) has the state-distribution problem when using the guide-policy, while our Self-Guided Exploration Strategy (SGES) does not. Because JSRL uses the guide-policy with a certain trajectory, then transfers to learning-policy, this combination has the state-distribution problem naturally~

Our SGES mixes guide-policy and learning-policy with the same probability so that they have the same state-distribution~

这种自我引导的方案，在大语言模型策略中仍然好用~

This self-guided approach is still useful in large language model.



## 中文版碎碎念：2023-02-18-10am
<details><summary><code>查看中文版碎碎念</code></summary>
最近看了《Dichotomy of Control: Separating What You Can Control from What You Cannot》，其实我文中所描述的一致非负奖励，也是其中的一种情况，只不过序列问题操作任务，如果按照这个描述，就太大了，论文中，还是以“夹爪无法改变物体的位置，而导致用HER反思后，会存在 隐式的非负稀疏奖励为切入点” 比较合适。借着这个机会，我想分享一下论文之外的感触。

对于智能体的探索和学习，尤其是稀疏奖励下的效率问题，我从18年底就开始尝试了，到现在已经四年多了，我发现我对强化学习的research，也算是一种长序列稀疏奖励过程。

在我长时间无法做出顶会顶刊的工作的时候，我也在反思，为什么我会赶个早集，却连口热乎的都吃不上？为何我的学习效率如此低下？

现在回想起来，一直陷入局部最“优”的我，直到21年，才有点变化。

21年底，我曾经花了很长的时间来复现著名的HER算法，这篇工作的方案是如此的简洁优雅，以至于后面出现了快两千的引用。先解释下HER算法，HER算法本质上是修改目标，让智能体有一个类似的“安慰奖”：虽然它没完成给定的任务，但如果当初设定的目标，就是它刚才完成的，那它岂不是可以认为已经成功了？

这听起来有点废话文学，但实际上，如果它完成任务的某些部分，下次任务真的给定了这个目标，它上次学到的知识是真的能用上！

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

最后，感谢各位专家无偿的批评和指导，以便让这篇工作有更好的呈现，让我自己也能够更好的成长~~~
</details>

<details><summary><code>English version</code></summary>
  I recently read "Dichotomy of Control: Separating What You Can Control from What You Cannot," and the consistent non-negative rewards I described in the text are actually one of the cases. However, in the sequence problem operation task, if it's described this way, it would be too broad. In the paper, it's more appropriate to use the "implicit non-negative sparse reward after using HER for reflection due to the gripper's inability to change the object's position" as the starting point. Taking this opportunity, I would like to share some thoughts outside of the paper.

I have been trying to explore and learn about agents, especially the efficiency of sparse rewards, since the end of 2018. It's been more than four years now, and I found that my research on reinforcement learning can also be considered a long sequence sparse reward process.

When I was unable to produce top-tier conference and journal papers for a long time, I reflected on why I was up early but couldn't even eat a hot meal. Why was my learning efficiency so low?

Looking back now, I, who was stuck in the local optimum, did not see any changes until 2021.

At the end of 2021, I spent a long time reproducing the famous HER algorithm. The approach of this work is so simple and elegant that it has received nearly 2,000 citations. To explain the HER algorithm, it essentially modifies the goal to give the agent a sort of "consolation prize": Although it did not complete the given task, if the original goal was what it just accomplished, can it be considered successful?

This may sound like nonsense literature, but in fact, if it completes some parts of the task, the next time the task is truly given this goal, the knowledge it learned last time can really be used!

So looking back, my previous explorations and failures were also useful. For example, I spent 2 months reproducing the OpenAI Baseline version of HER, as I couldn't reproduce the same performance with Pytorch. I tried almost every hyperparameter and setting. In the process of batch tuning, since manually launching the program was too inefficient, I changed the spinningup's MPI tutorial into a grid search hyperparameter mode. The tutorial is here: https://blog.csdn.net/hehedadaq/article/details/114685906.

Although I didn't produce any novel work during this long exploration process, I discovered a new problem. In the rendering process, I found that every time the FetchPush task was in the early stages of exploration, the robotic arm didn't know how to get close to the object. My most intuitive idea at the time was that this would definitely affect the efficiency of exploration! Ok, I found the problem, and my first attempt was to let the agent learn to approach before learning to operate (when I finished the entire experiment and started writing the article, I realized during the literature review that this was the approach of the SHER paper). Then I found that the effect was not as good as learning to approach and operate together, and I mixed exploration during the exploration process. This is the prototype of my RHER scheme.

Stepping out of the details of this experiment, for me, the previous exploration of HER, the understanding of HER, and the skills of batch tuning laid the foundation for me to later deal with the original RHER algorithm when it couldn't be directly extended to other multi-object operation tasks. In the article, this is reflected in the fact that I made six or seven setting modifications, finally unleashing the potential of RHER and achieving the state of the art in model-free reinforcement learning for multi-object operation tasks.

Past "failed" experiences and skills will help with future growth, both in and out of the article.

Since today is a Saturday morning, I have some time to sort through the complex sequential tasks under sparse rewards. There are a few points that can help an agent learn quickly:

Set progressive goals; it is unreasonable to aim too high from the start. This is reflected in the article by breaking down tasks, and outside of the article by setting task lists for oneself.

During the learning process, identify what is within your control and what is caused by the environment or other agents. Effective learning occurs when focusing on controllable experiences. In the article, this is represented by lowering the INNR ratio, allowing the gripper to influence the block as much as possible. Outside the article, if your actions cannot change the outcome of an event, then the causality between you and the event is low, and any reflection or conclusions drawn will be inefficient.

In the exploration process, having the help of mentors and experts is a shortcut to quickly navigate through areas that others have already explored. In the article, the exploration of complex subtasks is guided by the strategies of simpler subtasks that have already been learned. Outside the article, as a graduate student, I will receive guidance from mentors, senior students, and read papers, code, and blogs from experts in the field before starting my own exploration, summarization, and sharing.

There is another point not discussed in this article, which will be the content of my next article. I'll share it with everyone when the opportunity arises.

Regarding the third point, the article mentioned self-guided exploration. Because the content of the article is vast and my writing skills have not yet reached a proficient level, I couldn't elegantly include this in the introduction. Self-guidance needs to answer two questions: how to evaluate if a strategy is an expert strategy, and how to make good use of expert strategies? In real life, experts need objective evaluation criteria such as occupation, education, expertise, publications, patents, outstanding project experience, and so on. Another question is how to make better use of expert guidance? From a human perspective, it would be more polite to say "How can experts guide more efficiently?" Watching experts work is difficult to learn from; having an expert watch your actions and occasionally give pointers can improve your learning efficiency. However, the best approach is hands-on guidance, allowing for personal exploration while having someone to correct you when you deviate.

These two questions, for the RHER article, are about sequential object manipulation tasks. Although they are a common and general category of tasks, they still have unique characteristics and can be well-divided into multiple stages. Thus, we can evaluate the success rate of each sub-strategy to determine their "expertise level." The second question is addressed by using a hybrid exploration approach that ensures efficiency while preventing policy mismatches due to offline data distribution.

Therefore, the RHER algorithm achieves extremely high sample efficiency and can serve as a backbone algorithm for common sequential manipulation tasks. Its self-guided exploration can also be extended to multi-agent exploration and hierarchical reinforcement learning.

Lastly, I'd like to thank all the experts for their invaluable criticism and guidance, which has allowed this work to be presented more effectively and helped me grow as well."
</details>

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


# Starchart

[![Star History Chart](https://api.star-history.com/svg?repos=kaixindelele/RHER&type=Date)](https://star-history.com/#kaixindelele/RHER&Date)
