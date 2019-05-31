# dnd-lstm

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/qihongl/dlstm-demo/master)

demo, a lstm cell with a differentiable neural dictionary described in Ritter et al. (2018), though I'm not doing <a href="https://arxiv.org/abs/1602.01783">A3C</a>

the notebooks are hopefully self explanatory. you can run these notebooks with jupyter binder though it is pretty slow. 

this is implemented in <a href="https://princetonuniversity.github.io/PsyNeuLink/">psyneulink</a> as <a href="https://princetonuniversity.github.io/PsyNeuLink/MemoryFunctions.html?highlight=dnd#psyneulink.core.components.functions.statefulfunctions.memoryfunctions.DND">statefulfunctions.memoryfunctions.DND</a>


# some description

`src/contextual-choice.ipynb` contains an evidence accumulation task with "context". 
In the i-th trial,

- at time t, the model receives noisy observation, x_t (e.g. random dots moving around, slightly drifting to left/right)
and a "context vector" for this trial, call it context_i (e.g. an image of an apple)
- the task is to judge if the sum of the observation sequence is positive or negative. Let's denote this target by y_i (e.g. left/right button press)

If I haven't seen trial i before. I have to integrate x_t over time - evidence accumulation.
And because context_i is always paired with y_i by task design, so if I have seen trial i before, and remember that the target for context_i (e.g. image of an apple) is y_i (left button press). Then there is no need to do evidence accumulation, you can just pick y_i. The model wants to respond earlier because this reduces the loss, but this is only possible if the model stored (context_i, y_i) into its episodic memory buffer (DND).

<br>

References: 

- Ritter, S., Wang, J. X., Kurth-Nelson, Z., Jayakumar, S. M., Blundell, C., Pascanu, R., & Botvinick, M. (2018). Been There, Done That: Meta-Learning with Episodic Recall. arXiv [stat.ML]. Retrieved from http://arxiv.org/abs/1805.09692

- also see Blundell et al. 2016, Pritzel et al. 2017 and Kaiser et al 2017... 
