---
title: '美化 PowerShell 终端：打造高效且赏心悦目的命令行环境'
description: '在现代开发中，命令行终端不仅是执行任务的工具，更是开发者日常工作的重要伙伴。一个美观且功能强大的终端环境可以显著提升工作效率和用户体验。本文将介绍如何通过安装和配置一系列工具，来美化你的 PowerShell 终端，使其既高效又赏心悦目。'
pubDate: '2025-09-12'
heroImage: '../../assets/blue.jpg'
---


> 作为一名开发者，命令行终端几乎每天都要用。以前我对 PowerShell 的印象就是“蓝底白字”，功能强但界面实在太朴素。直到有一天看到同事的终端界面炫酷又实用，才意识到：原来 PowerShell 也能像 macOS 下的 oh-my-zsh 一样美观高效！

这篇博客就记录一下我折腾 PowerShell 终端美化的全过程，以及踩过的坑和一些小心得。

---

## 为什么要美化 PowerShell？

一开始只是觉得默认终端太丑，后来发现美化后不仅颜值提升，效率也高了不少：

- Git 状态、路径、虚拟环境等信息一目了然
- 支持丰富的主题和插件，体验和 oh-my-zsh 很像
- 还能用上各种有趣的图标和配色

---

## 我的美化方案

其实核心就三样：

1. **Oh My Posh** —— 终端提示符主题引擎，类似 oh-my-zsh，但更现代、跨平台
2. **Nerd Fonts** —— 支持各种图标的特殊字体，不然主题会乱码
3. **功能模块** —— 类似插件，增强补全、图标等

---

## 实践步骤 & 踩坑记录

### 1. 字体一定要先装对！

一开始我没注意字体，结果主题装好后全是方块和乱码，查了半天才发现是 Nerd Font 没装对。

- 推荐 [Caskaydia Cove Nerd Font](https://www.nerdfonts.com/font-downloads)，兼容性最好。
- 下载后解压，右键所有字体文件“为所有用户安装”。
- Windows Terminal 里记得手动切换字体，否则还是不生效。

### 2. 安装 Oh My Posh

用 scoop 安装最方便：

```powershell
scoop install oh-my-posh
```

装完重启 PowerShell。

### 3. 配置 PowerShell Profile

这一步是让美化每次都自动生效。用记事本打开 profile：

```powershell
notepad $PROFILE
```

粘贴如下内容（主题名可换成你喜欢的）：

```powershell
# 初始化 Oh My Posh 主题引擎
oh-my-posh init pwsh --config '$env:POSH_THEMES_PATH/jandedobbeleer.omp.json' | Invoke-Expression
```

保存后，重启 PowerShell 或运行：

```powershell
. $PROFILE
```

这时应该能看到全新的提示符了！

### 4. 换主题玩花样

Oh My Posh 主题库非常丰富：[主题库地址](https://ohmyposh.dev/docs/themes#catppuccin)

挑喜欢的主题，把 profile 里的 `jandedobbeleer.omp.json` 换成对应主题名，保存并重载即可。

---

## 我的感受 & 总结

折腾完之后，PowerShell 终于有了现代终端的感觉，颜值和效率都提升了。最容易踩坑的就是字体，建议一定先搞定字体再装主题。

如果你也想让终端变得更好用、更好看，不妨试试这套方案！有问题欢迎留言交流～