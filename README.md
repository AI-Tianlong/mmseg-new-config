# 【MMseg new config 迁移 社区任务】
Hello~ 各位社区大佬们。  
- 感谢参加此次 MMSEG new config 迁移社区任务，最终会根据大家迁移的config数量以及难度,给与合理的积分奖励哒~    
  - 注：积分可以用来换取精美的OpenMMLab周边哦~  
+ 目前，需要招募几位同学，优先把数据集的config迁移完毕，以供大家在模型迁移时使用。  
如有想法，可随时联系群内AI-Tianlong
---
**咳咳，开始详细说明此次 new config 贡献流程。(教程细致程度为面向从未有贡献经验的小伙伴们~**  
话不多说，我们赶快开始上手操作起来吧！
# 步骤1：为 mmsegmentation 贡献代码的准备工作   
**（有贡献经验的同学可忽略）**  
此步骤包含：
- 在Github 中 Fork mmsegmentation 至自己的仓库
- 以源码方式构建自己仓库中 Fork 的 mmsegmentation
- 为自己Fork的mmsegmentation 添加 upstream(上游仓库)
- 安装 pre-commit 代码格式检查工具

请参考，[在 mmsegmentation projects 中贡献一个标准格式的数据集教程](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/zh_cn/advanced_guides/contribute_dataset.md)，认真仔细核对完成参考教程中的 **`步骤1`及`步骤2`** 。  
# 步骤2：为 mmsegmentation 贡献 new config 配置文件
new config迁移的具体注意事项，可以参考：https://aicarrier.feishu.cn/docx/PkICdaJbpoOVjLxMFi2cYqPjnOc  
数据集、模型、schedule、defaulthook的new 配置文件，请看[configs_new](https://github.com/AI-Tianlong/mmseg-new-config/tree/main/configs_new)  

根据在微信群内认领的任务，创建属于自己的分支。  
例如，我在群内领了迁移potsdam数据集配置文件的任务，则我的分支可以命名为：  
```bash
cd mmsegmentation
git checkout dev-1.x  # 切换至dev-1.x分支
# git checkout -b {Github ID}/{任务相关的分支命名}
git checkout -b AI-Tianlong/support_potsdam_new_config
```
那么我的新分支，将命名为`AI-Tianlong/support_potsdam_new_config`。
在mmsegmentation文件树一侧，找到 config_new 文件夹，如没有，可先自己创立一个😢  
然后按照和`configs`下一模一样的子路径，创建new configs文件。  
如我要迁移`configs/_base_/datasets/potsdam.py`，那么我new config的位置应为：`configs_new/_base_/datasets/potsdam.py`。  
![image](https://github.com/AI-Tianlong/mmseg-new-config/assets/50650583/ce7d0a05-da8f-4ad9-a675-cbee18cc5419)

# 注意事项
任务领取以一个完整的模型config为单位，请在群内接龙，查看是否有重复领取：
- `configs_new/_base_/models/deeplabv3.py`
- `configs_new/deeplabv3下`
  - `xxxxxxxxxxxxxxxxxxxx.py`
  - `xxxxxxxxxxxxxxxxxxxx.py`
  - `xxxxxxxxxxxxxxxxxxxx.py`
  - `xxxxxxxxxxxxxxxxxxxx.py`
  - `xxxxxxxxxxxxxxxxxxxx.py` 
<img src='https://github.com/AI-Tianlong/mmseg-new-config/assets/50650583/c4f77d65-6cbe-4f86-8a0a-b391e7419c05' alt="微信群聊二维码" width="50%">
