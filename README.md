# dnd-lstm [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/qihongl/dlstm-demo/master)

A lstm cell with a differentiable neural dictionary described in Ritter et al. (2018). 


<img src="https://github.com/qihongl/dnd-lstm/blob/master/figs/dnd-lstm-cell.png" width=500>


### Task description 

`src/contextual-choice.ipynb` contains an 
<a href="https://en.wikipedia.org/wiki/Two-alternative_forced_choice#Behavioural_experiments">evidence accumulation task </a>
with "context". 

In the i-th trial,

- At time t, the model receives noisy observation, x_t (e.g. random dots moving around, slightly drifting to left/right)
and a "context vector" for this trial, call it context_i (e.g. an image of an apple)
- The task is to respond the average direction of x_t (i.e. sign), analogous to making a left/right button press. Let's denote the response target by y_i.  
- If the model never seen trial i before, it has to integrate x_t over time to figure out the average direction - evidence accumulation.
- Additionally, context is trial unique - context_i is always paired with y_i, for all i. Therefore if context_i (e.g. the apple image) reoccur, the model can respond y_i (left button press) directly without doing evidence accumulation. The model wants to respond earlier because this maximizes cumulative return. 

Note that this is only possible if the model stored (context_i, y_i) into its episodic memory buffer (DND). So this task can demonstrate if the model can use episodic memory to guide its choices. 

### Results

Behaviorally, when the model encounters a previously-seen trial, the choice accuracy jumps to ceiling immediately. By task design, this is only possible if the model can retrieve the correct episodic memory. 
- Without a relevant memory, there is no way to perform better than chance before t=5, because inputs before time 5 are noisy. 

<img src="https://github.com/qihongl/dnd-lstm/blob/master/figs/correct-rate.png" width=450>

A 
<a href="https://en.wikipedia.org/wiki/Principal_component_analysis">PCA</a>
analysis of the memory content shows that the choice is encoded in the memory: 

<img src="https://github.com/qihongl/dnd-lstm/blob/master/figs/pc-v.png" width=450>


### References

- Ritter, S., Wang, J. X., Kurth-Nelson, Z., Jayakumar, S. M., Blundell, C., Pascanu, R., & Botvinick, M. (2018). Been There, Done That: Meta-Learning with Episodic Recall. arXiv [stat.ML]. Retrieved from http://arxiv.org/abs/1805.09692

    - also see Blundell et al. 2016, Pritzel et al. 2017 and Kaiser et al 2017... 

- Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T. P., Harley, T., … Kavukcuoglu, K. (2016). Asynchronous Methods for Deep Reinforcement Learning. Retrieved from http://arxiv.org/abs/1602.01783


### Extra note 

1. A variant of the DND part is implemented in 
<a href="https://princetonuniversity.github.io/PsyNeuLink/">psyneulink</a> 
as <a href="https://princetonuniversity.github.io/PsyNeuLink/MemoryFunctions.html?highlight=dnd#psyneulink.core.components.functions.statefulfunctions.memoryfunctions.ContentAddressableMemory">    pnl.ContentAddressableMemory</a>. 

2. The original paper uses A3C. I'm doing A2C instead - no asynchronous parallel rollouts. 
