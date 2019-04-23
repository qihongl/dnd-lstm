# dnd-lstm

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/qihongl/dlstm-demo/master)

demo, a lstm cell with a differentiable neural dictionary described in Ritter et al. (2018), though I'm not doing <a href="https://arxiv.org/abs/1602.01783">A3C</a>

the notebooks are hopefully self explanatory. you can run these notebook with jupyter binder though it is pretty slow. 

this is implemented in <a href="https://princetonuniversity.github.io/PsyNeuLink/">psyneulink</a> as <a href="https://princetonuniversity.github.io/PsyNeuLink/MemoryFunctions.html?highlight=dnd#psyneulink.core.components.functions.statefulfunctions.memoryfunctions.DND">statefulfunctions.memoryfunctions.DND</a>

<br>

References: 

- Ritter, S., Wang, J. X., Kurth-Nelson, Z., Jayakumar, S. M., Blundell, C., Pascanu, R., & Botvinick, M. (2018). Been There, Done That: Meta-Learning with Episodic Recall. arXiv [stat.ML]. Retrieved from http://arxiv.org/abs/1805.09692

- also see Blundell et al. 2016, Pritzel et al. 2017 and Kaiser et al 2017... 
