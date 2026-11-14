# 2026-11-13 主线整理审计

- `manifest-before.json`：清理前普通文件SHA256、大小与符号链接目标；主清单不包含Git元数据，两个第三方仓库的Git元数据另有已核验补充包，主项目.git仍保留本地。
- `archive-receipt.json`：SJTU1独立快照的逐文件核验结果。
- `cleanup-plan.json`：明确保留和移出的原始路径集合；不通过目录名通配删除。
- `cleanup-result.json`：核验后实际清理的计数与容量。
- 恢复命令及依赖边界见`../REPRODUCIBILITY.md`。

原报告和源文件受保护；服务器运行目录不修改。归档快照不与运行目录共享硬链接。本地删除前再次核对原字节，任何不匹配均停止清理。新文件和Git元数据不在清理范围。

`third-party-git-manifest.json`及`third-party-git-verification.json`对应SJTU1同归档目录的`third-party-git.tar`，保留两份第三方仓库的完整Git元数据；需要恢复第三方历史时与主快照中的源文件一起恢复。
