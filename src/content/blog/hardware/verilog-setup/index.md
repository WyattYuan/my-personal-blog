---
title: 'Verilog环境配置'
description: ''
pubDate: '2025-11-14'
heroImage: '../../../../assets/blog-placeholder-1.jpg'
tags: ['Verilog', '环境配置', '开发工具', '硬件描述语言']
---
# Verilog环境配置

# vscode

## 用scoop安装插件

```powershell
scoop install universal-ctags

scoop install iverilog

```

check

```powershell
ctags --version
gtkwave --version
iverilog --version
```

## vscode扩展下载

参考

[Verilog 编程 - Digital Lab 2025](https://soc.ustc.edu.cn/Digital/history/2023/lab1/verilog_coding/)

[(12 封私信 / 34 条消息) vscode搭建Verilog HDL开发环境 - 知乎](https://zhuanlan.zhihu.com/p/586927434)

[(12 封私信 / 34 条消息) 一款轻量级verilog HDL开发方案（一）vscode+iverilog搭建开发环境 - 知乎](https://zhuanlan.zhihu.com/p/367612172)

## format

[VSCode配置Verilog开发环境 | Esing的小站](https://blog.esing.dev/2024/01/18/vscode-verilog-setup/)

```verilog
scoop install verible

verible-verilog-format --version
```

![image](./assets/image-20251110143728-riw9ym1.png)

```verilog
--indentation_spaces=4 --line_break_penalty=3 --named_port_alignment=align --port_declarations_alignment=align --port_declarations_indentation=indent --module_net_variable_alignment=align --assignment_statement_alignment=align --wrap_end_else_clauses=true --try_wrap_long_lines=true --column_limit=90 --named_parameter_alignment=align --named_parameter_indentation=indent --case_items_alignment=align --enum_assignment_statement_alignment=align --named_port_indentation=indent --formal_parameters_alignment=align --formal_parameters_indentation=wrap --class_member_variable_alignment=align --distribution_items_alignment=align --port_declarations_right_align_packed_dimensions=true
```

关键设置

![image](./assets/image-20251110143743-on3dlnm.png)

## lint

- xvlog（重型，最严格，但很慢）
- verilator（平时用最好，又快又严格）

  - ```verilog
    MinGW // 进入mingw终端

    pacman -Syu

    pacman -S mingw-w64-x86_64-verilator

    verilator --version

    // 记得将verilator添加到环境变量：...\msys2\current\mingw64\bin
    ```
- iverilog似乎无法跨模块

# logisim

[logisim-evolution](https://github.com/logisim-evolution/logisim-evolution)

[(13 封私信 / 44 条消息) 用VSCode编辑verilog代码、iverilog编译、自动例化、自动补全、自动格式化等常用插件 - 知乎](https://zhuanlan.zhihu.com/p/338497672)

# 安装vivado

官网，登录，下载2024.2

[Vivado 安装教程 - Digital Lab 2025](https://soc.ustc.edu.cn/Digital/2025/lab0/vivado/)

改text editor

## vscode

**选中Verilog configuration，然后将Linter修改为**​**​`xvlog`​**（较重型）
