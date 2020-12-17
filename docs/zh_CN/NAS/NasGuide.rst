One-Shot NAS 算法
=======================

除了 `经典 NAS 算法 <./ClassicNas.rst>`_，还可以使用更先进的 One-Shot NAS 算法来从搜索空间中找到更好的模型。 There are lots of related works about one-shot NAS algorithms, such as `SMASH <https://arxiv.org/abs/1708.05344>`__\ , `ENAS <https://arxiv.org/abs/1802.03268>`__\ , `DARTS <https://arxiv.org/abs/1808.05377>`__\ , `FBNet <https://arxiv.org/abs/1812.03443>`__\ , `ProxylessNAS <https://arxiv.org/abs/1812.00332>`__\ , `SPOS <https://arxiv.org/abs/1904.00420>`__\ , `Single-Path NAS <https://arxiv.org/abs/1904.02877>`__\ ,  `Understanding One-shot <http://proceedings.mlr.press/v80/bender18a>`__ and `GDAS <https://arxiv.org/abs/1910.04465>`__. One-shot NAS algorithms usually build a supernet containing every candidate in the search space as its subnetwork, and in each step, a subnetwork or combination of several subnetworks is trained.

Currently, several one-shot NAS methods are supported on NNI. For example, ``DartsTrainer``\ , which uses SGD to train architecture weights and model weights iteratively, and ``ENASTrainer``\ , which `uses a controller to train the model <https://arxiv.org/abs/1802.03268>`__. New and more efficient NAS trainers keep emerging in research community and some will be implemented in future releases of NNI.

Search with One-shot NAS Algorithms
-----------------------------------

Each one-shot NAS algorithm implements a trainer, for which users can find usage details in the description of each algorithm. Here is a simple example, demonstrating how users can use ``EnasTrainer``.

.. code-block:: python

   ＃与传统模型训练完全相同
   model = Net()
   dataset_train = CIFAR10(root="./data", train=True, download=True, transform=train_transform)
   dataset_valid = CIFAR10(root="./data", train=False, download=True, transform=valid_transform)
   criterion = nn.CrossEntropyLoss()
   optimizer = torch.optim.SGD(model.parameters(), 0.05, momentum=0.9, weight_decay=1.0E-4)

   # 这里使用 NAS
   def top1_accuracy(output, target):
       # 这是ENAS算法要求的计算奖励的函数
       batch_size = target.size(0)
       _, predicted = torch.max(output.data, 1)
       return (predicted == target).sum().item() / batch_size

   def metrics_fn(output, target):
       # metrics 函数接收 output 和 target 并计算出 dict metrics
       return {"acc1": top1_accuracy(output, target)}

   from nni.algorithms.nas.pytorch import enas
   trainer = enas.EnasTrainer(model,
                              loss=criterion,
                              metrics=metrics_fn,
                              reward_function=top1_accuracy,
                              optimizer=optimizer,
                              batch_size=128
                              num_epochs=10,  # 10 epochs
                              dataset_train=dataset_train,
                              dataset_valid=dataset_valid,
                              log_frequency=10)  # 每 10s 打印一次 log
   trainer.train()  # training
   trainer.export(file="model_dir/final_architecture.json")  # 把最终的架构导出到文件

``model`` is the one with `user defined search space <./WriteSearchSpace.rst>`__. Then users should prepare training data and model evaluation metrics. To search from the defined search space, a one-shot algorithm is instantiated, called trainer (e.g., EnasTrainer). The trainer exposes a few arguments that you can customize. For example, the loss function, the metrics function, the optimizer, and the datasets. These should satisfy most usage requirements and we do our best to make sure our built-in trainers work on as many models, tasks, and datasets as possible.

**Note that** when using one-shot NAS algorithms, there is no need to start an NNI experiment. Users can directly run this Python script (i.e., ``train.py``\ ) through ``python3 train.py`` without ``nnictl``. After training, users can export the best one of the found models through ``trainer.export()``.

Each trainer in NNI has its targeted scenario and usage. Some trainers have the assumption that the task is a classification task; some trainers might have a different definition of "epoch" (e.g., an ENAS epoch = some child steps + some controller steps). Most trainers do not have support for distributed training: they won't wrap your model with ``DataParallel`` or ``DistributedDataParallel`` to do that. So after a few tryouts, if you want to actually use the trainers on your very customized applications, you might need to `customize your trainer <./Advanced.rst#extend-the-ability-of-one-shot-trainers>`__.

Furthermore, one-shot NAS can be visualized with our NAS UI. `See more details. <./Visualization.rst>`__

使用导出的架构重新训练
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

搜索阶段后，就该训练找到的架构了。 与很多开源 NAS 算法不同，它们为重新训练专门写了新的模型。 我们发现搜索模型和重新训练模型的过程非常相似，因而可直接将一样的模型代码用到最终模型上。 例如：

.. code-block:: python

   model = Net()
   apply_fixed_architecture(model, "model_dir/final_architecture.json")

此 JSON 是从 Mutable 键值到 Choice 的映射。 Choice 可为：


* string: 根据名称来指定候选项。
* number: 根据索引来指定候选项。
* string 数组: 根据名称来指定候选项。
* number 数组: 根据索引来指定候选项。
* boolean 数组: 可直接选定多项的数组。

例如：

.. code-block:: json

   {
       "LayerChoice1": "conv5x5",
       "LayerChoice2": 6,
       "InputChoice3": ["layer1", "layer3"],
       "InputChoice4": [1, 2],
       "InputChoice5": [false, true, false, false, true]
   }

应用后，模型会被固定，并准备好进行最终训练。 该模型作为单独的模型来工作，未使用的参数和模块已被剪除。

也可参考 `DARTS <./DARTS.rst>`__ 的重新训练代码。
