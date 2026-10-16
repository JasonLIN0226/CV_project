# 标签生产来源补充核验（2026-10-13 22:52 EDT）

精确标签对齐仍未证明，未选择标签join；没有新增训练或推理。

从[官方项目网站](https://assembly-101.github.io/)新核验其固定版本HTML与作者公开托管的9页补充材料。原Git blob逐项匹配，补充PDF SHA256 `6e191ce9328a86bce6e52410d8f55f6b5f0443dfd7f56d2f51913524bf8eff30`。第2页标注界面已实际渲染并查看；三个静态相机画面和时间边界没有给出原RGB到annotation-frame0的生产起点。消费模型的特征采样说明也不能证明该映射。[完整证据](</Users/linxiangyu/Desktop/MSDS Semester1/CV/project/phase1_jepa_memory/runs/phase1_v6_revision/primary_sources/additional_official_asset_audit_20261014T0253.json>)。

官方Sample链接正常可访问，根目录5个公开录像文件夹。仅读取9043-a24…133010子目录元数据：12项均为MP4，没有带索引的原标注帧。该录像名不等于固定开发清单的9043-a24…131831；相同participant/toy前缀不能证明是同一录像，未预览、下载或解码样例视频，也未增加开发样本。

原始CVF补充链接返回403后没有重复该路径；使用的是官方作者仓库已经公开托管的文档。未发送访问请求或外部消息，临时网页已关闭。

下一项必需证据仍是权威抽帧生产脚本或带recording/camera/frame索引的原标注帧。现有可执行视觉计算及最终数值来源复核已全部结束；相关语义NLL/活动漂移/切换/供体恢复继续UNAVAILABLE。没有新认证待办或HPC分配。

## 新增维护者引用链核验

继续读取先前未展开的官方issues1/6/13评论。官方OWNER在issue6说明改用了RU-LSTM的视觉特征提取程序；论文作者在issue1说明采用TSM的2D骨干。实际核验所链接RU-LSTM固定commit e648c7ee40a1a71381bfaa7563f2e8a6935275ab及Git blob：程序只读取已经存在的JPG并按文件名保存特征，没有原MP4抽帧、取样相位或annotation-frame0生产过程。未运行源码或导入模型，未取回任何权重、特征、标签或原视频。这个新引用链仍不足以证明标签精确对齐。
