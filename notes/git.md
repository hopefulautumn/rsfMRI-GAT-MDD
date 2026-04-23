```
# 安装（如果没装）
git lfs install

# 跟踪 npz 文件
git lfs track "*.npz"

# 提交 .gitattributes
git add .gitattributes
git commit -m "track npz with lfs"
```

```
# 从历史中移除大文件
git rm --cached processed/rest_meta_mdd_fc.npz
git rm --cached backup/processed/rest_meta_mdd_fc.npz

git commit -m "remove large files"
```

```
git add processed/rest_meta_mdd_fc.npz
git commit -m "add with lfs"
git push origin main


```