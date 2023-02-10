# lung3d

## 安装依赖并启动

```bash
# 安装依赖
npm install

# dev
npm run electron:serve

# build
npm run electron:build
```

编译好的应用下载链接：https://cloud.tsinghua.edu.cn/f/ef1adb7bcafb4a8bb0d9/?dl=1



## 功能说明

本项目用于可视化肺部分割和肺结节病灶热力图。其中分割文件和热力图文件需要提前生成，以下链接可以下载示例：https://cloud.tsinghua.edu.cn/f/ef1adb7bcafb4a8bb0d9/?dl=1

解压后示例文件后，运行程序，点击Open Volume按钮打开其中的mhd文件即可

根据CT数据的特点，项目实现了三种不同的渲染方法，其中最大强度投影（Maximum Intensity Projection）效果最好

