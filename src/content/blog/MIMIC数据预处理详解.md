---
title: 'MIMIC-III 数据预处理详细讲解文档'
description: ''
pubDate: '2025-09-12'
heroImage: '../../assets/blog-placeholder-1.jpg'
tags: ['MIMIC-III', '数据预处理', '机器学习', '医疗数据']
---
# MIMIC-III 数据预处理详细讲解文档

## 目录
1. [概述](#概述)
2. [数据流分析](#数据流分析)
3. [Notebook 1: Admissions.ipynb - 患者入院数据处理](#notebook-1-admissionsipynb)
4. [Notebook 2: LabEvents.ipynb - 实验室检验数据处理](#notebook-2-labeventsipynb)
5. [Notebook 3: Outputs.ipynb - 输出事件数据处理](#notebook-3-outputsipynb)
6. [Notebook 4: Prescriptions.ipynb - 处方数据处理](#notebook-4-prescriptionsipynb)
7. [Notebook 5: DataMerging.ipynb - 数据合并与张量化](#notebook-5-datamergingipynb)
8. [Notebook 6: Extra_covariates.ipynb - 额外协变量创建](#notebook-6-extra_covariatesipynb)
9. [输出文件说明](#输出文件说明)
10. [总结](#总结)

---

## 概述

本文档详细说明了 MIMIC-III 临床数据库的预处理流程。整个处理过程将原始的 MIMIC-III 数据库转换为适合 GRU-ODE-Bayes 模型训练的时间序列数据集。

### 处理目标
- **任务**: 基于患者ICU前48小时的时间序列数据预测住院死亡率
- **数据类型**: 多变量不规则采样时间序列
- **最终输出**: 张量化的时间序列数据、死亡标签、协变量

### 关键处理步骤
1. 患者筛选与过滤
2. 数据清洗与异常值处理
3. 时间戳标准化
4. 时间分桶（Time Binning）
5. 数据归一化
6. 训练/验证/测试集划分

---

## 数据流分析

### 整体数据流向图

```
原始 MIMIC-III 数据库
├── ADMISSIONS.csv ─────────┐
├── PATIENTS.csv ───────────┤
├── INPUTEVENTS_MV.csv ─────┤
├── OUTPUTEVENTS.csv ───────┤
├── LABEVENTS.csv ──────────┤──> [数据清洗 & 筛选] ──> 中间处理文件
├── PRESCRIPTIONS.csv ──────┤                           ├── Admissions_processed.csv
├── DIAGNOSES_ICD.csv ──────┤                           ├── INPUTS_processed.csv
├── D_ITEMS.csv ────────────┤                           ├── OUTPUTS_processed.csv
└── D_LABITEMS.csv ─────────┘                           ├── LAB_processed.csv
                                                         └── PRESCRIPTIONS_processed.csv
                                                                   ↓
                                                         [DataMerging.ipynb]
                                                         数据合并、时间分桶、归一化
                                                                   ↓
                                                         ┌─────────┴─────────┐
                                                         ↓                   ↓
                                              张量分解数据集          LSTM/GRU-ODE 数据集
                                              ├── complete_tensor.csv       ├── LSTM_tensor_train.csv
                                              ├── complete_death_tags.csv   ├── LSTM_tensor_val.csv
                                              ├── complete_covariates.csv   ├── LSTM_tensor_test.csv
                                              ├── complete_tensor_train1-3  ├── LSTM_death_tags_*.csv
                                              ├── complete_tensor_val1-3    └── LSTM_covariates_*.csv
                                              └── complete_tensor_test
                                                         ↓
                                              [Extra_covariates.ipynb]
                                              添加额外协变量
                                              ├── complete_covariates_extended.csv
                                              ├── complete_durations_vec.csv
                                              └── mean_features.csv
```

### 关键数据转换流程

1. **患者筛选阶段**
   - 输入: ADMISSIONS.csv (58,976 入院记录)
   - 筛选条件:
     - 仅一次入院
     - ICU停留时间: 2-30天
     - 年龄 > 15岁
     - 有图表事件数据
     - 排除新生儿
   - 输出: ~22,000 患者

2. **特征选择阶段**
   - LAB: 39个实验室检验指标
   - INPUTS: 32个输入事件类型
   - OUTPUTS: 15个输出事件类型
   - PRESCRIPTIONS: 10种药物处方

3. **时间处理阶段**
   - 时间窗口: 入院后48小时
   - 时间分桶: 每小时60个bins（即每分钟1个bin）
   - 参考时间: 每次入院的最早事件时间

---

## Notebook 1: Admissions.ipynb

### 功能概述
处理患者入院信息和输入事件数据，建立数据集的患者基础。

### 输入文件
- `ADMISSIONS.csv`: 入院记录
- `PATIENTS.csv`: 患者基本信息
- `INPUTEVENTS_MV.csv`: MetaVision系统的输入事件（如药物、液体输注）
- `D_ITEMS.csv`: 项目字典

### 处理流程

#### 1. 患者筛选

```python
# 步骤1: 加载入院数据并合并患者出生日期
adm = pd.read_csv("ADMISSIONS.csv")
patients_df = pd.read_csv("PATIENTS.csv")
adm_dob = pd.merge(patients_df[["SUBJECT_ID", "DOBTIME"]], adm, on="SUBJECT_ID")
```

**筛选条件详解:**

##### 1.1 单次入院患者
```python
# 统计每个患者的入院次数
df = adm.groupby("SUBJECT_ID")["HADM_ID"].nunique()
# 只保留单次入院的患者
subj_ids = list(df[df==1].index)
adm_1 = adm_dob.loc[adm_dob["SUBJECT_ID"].isin(subj_ids)]
```
**原因**: 多次入院会引入复杂的患者历史依赖，简化模型训练。

##### 1.2 ICU停留时间筛选
```python
# 计算停留时长
adm_1["ELAPSED_TIME"] = adm_1["DISCHTIME"] - adm_1["ADMITTIME"]
adm_1["ELAPSED_DAYS"] = adm_1["ELAPSED_TIME"].dt.days
# 筛选 2-30 天
adm_2 = adm_1.loc[(adm_1["ELAPSED_DAYS"]<30) & (adm_1["ELAPSED_DAYS"]>2)]
```
**原因**: 
- 停留时间过短(<2天): 数据不足，无法捕获有意义的时间动态
- 停留时间过长(>30天): 可能是慢性病例，不符合急性ICU场景

##### 1.3 年龄筛选
```python
# 计算年龄（处理日期溢出问题）
age_years = (adm_2['ADMITTIME'].dt.year - adm_2['DOBTIME'].dt.year)
# 根据生日是否已过进行调整
age_years -= (~birthday_passed).astype(int)
# 筛选年龄 > 15岁
adm_2_15 = adm_2.loc[age_years > 15]
```
**注意**: 代码中处理了MIMIC数据中>89岁患者的出生日期被设置为300年后的隐私保护问题。

##### 1.4 其他筛选
```python
# 必须有图表事件数据
adm_2_15_chart = adm_2_15.loc[adm_2_15["HAS_CHARTEVENTS_DATA"]==1]
# 排除新生儿
adm_3 = adm_2_15_chart.loc[adm_2_15_chart["ADMISSION_TYPE"]!="NEWBORN"]
```

**创建死亡标签:**
```python
adm_1["DEATHTAG"] = 0
adm_1.loc[adm_1.DEATHTIME.notnull(), "DEATHTAG"] = 1
```

#### 2. 输入事件数据处理

##### 2.1 特征选择
从数千种输入项目中选择32个最常见和临床重要的输入类型：

```python
retained_list = [
    "Albumin 5%", "Dextrose 5%", "Lorazepam (Ativan)",
    "Calcium Gluconate", "Midazolam (Versed)", "Phenylephrine",
    "Furosemide (Lasix)", "Hydralazine", "Norepinephrine",
    "Magnesium Sulfate", "Nitroglycerin", "Insulin - Glargine",
    # ... 等32个
]
```

##### 2.2 单位标准化

**问题**: 同一药物可能有不同的单位记录（如 mg vs dose）

**解决方案**: 
```python
# 示例：Fentanyl 单位转换
# 将 mg 转换为 mcg
inputs_small_3.loc[
    (inputs_small_3["LABEL"]=="Fentanyl (Concentrate)") & 
    (inputs_small_3["AMOUNTUOM"]=="mg"), 
    "AMOUNT"
] *= 1000
inputs_small_3.loc[
    (inputs_small_3["LABEL"]=="Fentanyl (Concentrate)") & 
    (inputs_small_3["AMOUNTUOM"]=="mg"), 
    "AMOUNTUOM"
] = "mcg"
```

##### 2.3 异常值处理

**负值处理**:
```python
# 检查是否是开始/结束时间颠倒导致
# 如果是，交换时间并取绝对值
a = inputs_small_3.loc[inputs_small_3["AMOUNT"]<0, "STARTTIME"]
inputs_small_3.loc[inputs_small_3["AMOUNT"]<0, "STARTTIME"] = \
    inputs_small_3.loc[inputs_small_3["AMOUNT"]<0, "ENDTIME"]
inputs_small_3.loc[inputs_small_3["AMOUNT"]<0, "ENDTIME"] = a
inputs_small_3.loc[inputs_small_3["AMOUNT"]<0, "AMOUNT"] *= -1
```

**极端值移除**: 使用 4σ 规则
```python
amount_desc = inputs_small_4.groupby("LABEL")["AMOUNT"].describe()
for label in name_list:
    inputs_small_4 = inputs_small_4.drop(
        inputs_small_4.loc[
            (inputs_small_4["LABEL"]==label) & 
            (inputs_small_4["AMOUNT"] > (amount_desc.loc[label,"mean"] + 4*amount_desc.loc[label,"std"]))
        ].index
    ).copy()
```

##### 2.4 时间序列离散化

**目标**: 将持续输注转换为离散时间点的数据

```python
duration_split_hours = 0.5  # 每30分钟一个数据点

# 对于持续时间 > 0.5小时的输入
df_temp1["Repeat"] = np.ceil(df_temp1["DURATION"].dt.total_seconds()/to_sec_fact).astype(int)
# 复制行
df_new1 = df_temp1.reindex(df_temp1.index.repeat(df_temp1["Repeat"]))
# 创建时间序列
df_new1["CHARTTIME"] = df_new1.groupby(level=0)['STARTTIME'].transform(
    lambda x: pd.date_range(start=x.iat[0], freq='30min', periods=len(x))
)
# 分配剂量
df_new1["AMOUNT"] = df_new1["AMOUNT"] / df_new1["Repeat"]
```

**效果**: 
- 输入: [时间0: 开始, 持续2小时, 总量100ml]
- 输出: [时间0: 25ml, 时间30min: 25ml, 时间60min: 25ml, 时间90min: 25ml]

### 输出文件
- **`Admissions_processed.csv`**: 约22,000个筛选后的入院记录，包含DEATHTAG
- **`INPUTS_processed.csv`**: 清洗后的输入事件数据，每条记录包含:
  - HADM_ID, SUBJECT_ID
  - CHARTTIME (离散时间点)
  - LABEL (输入类型)
  - AMOUNT (剂量)

### 关键统计
- **初始患者数**: 46,520 (单次入院)
- **经过所有筛选后**: ~22,000 患者
- **输入事件特征数**: 32
- **死亡率**: 约12-15%

---

## Notebook 2: LabEvents.ipynb

### 功能概述
处理实验室检验数据，这是临床时间序列中最重要的部分。

### 输入文件
- `LABEVENTS.csv`: 实验室检验事件（数千万条记录）
- `D_LABITEMS.csv`: 实验室项目字典
- `Admissions_processed.csv`: 筛选后的入院记录

### 处理流程

#### 1. 特征选择

选择39个关键实验室指标（基于临床重要性和数据可用性）：

```python
subset = [
    # 代谢指标
    "Glucose", "Bicarbonate", "Calcium, Total", "Chloride", 
    "Potassium", "Sodium", "Magnesium", "Phosphate",
    
    # 肾功能
    "Creatinine", "Urea Nitrogen",
    
    # 肝功能
    "Albumin", "Alanine Aminotransferase (ALT)",
    "Alkaline Phosphatase", "Asparate Aminotransferase (AST)",
    "Bilirubin, Total",
    
    # 血液学
    "Hemoglobin", "Hematocrit", "Platelet Count",
    "Red Blood Cells", "White Blood Cells",
    "Neutrophils", "Lymphocytes", "Monocytes", 
    "Eosinophils", "Basophils",
    "MCH", "MCHC", "MCV", "RDW",
    
    # 凝血功能
    "PT", "PTT",
    
    # 血气分析
    "pH", "pO2", "pCO2", "Base Excess", "Lactate",
    
    # 其他
    "Anion Gap", "Calculated Total CO2", "Specific Gravity"
]
```

#### 2. 单位标准化

**问题**: 不同医院/时间可能使用不同单位

```python
# 示例：统一单位
lab3.loc[lab3["LABEL"]=="Calculated Total CO2", "VALUEUOM"] = "mEq/L"
lab3.loc[lab3["LABEL"]=="PT", "VALUEUOM"] = "sec"
lab3.loc[lab3["LABEL"]=="pCO2", "VALUEUOM"] = "mm Hg"
lab3.loc[lab3["LABEL"]=="pH", "VALUEUOM"] = "units"
lab3.loc[lab3["LABEL"]=="pO2", "VALUEUOM"] = "mm Hg"
```

#### 3. 数据清洗

##### 3.1 缺失值处理
```python
# Glucose 特殊处理：NEG（阴性）标记为-1
lab3.loc[
    (lab3["LABEL"]=="Glucose") & 
    (lab3["VALUENUM"].isnull()) & 
    (lab3["VALUE"]=="NEG"), 
    "VALUENUM"
] = -1

# 删除其他缺失值
lab3 = lab3.drop(lab3.loc[lab3["VALUENUM"].isnull()].index)
```

##### 3.2 异常值处理

**基于生理学合理范围**:
```python
# Anion Gap: 不能为负
lab3 = lab3.drop(lab3.loc[
    (lab3["VALUENUM"]<0) & (lab3["LABEL"]=="Anion Gap")
].index)

# Base Excess: -50 到 50 mEq/L
lab3 = lab3.drop(lab3.loc[
    (lab3["LABEL"]=="Base Excess") & (lab3["VALUENUM"]<-50)
].index)
lab3 = lab3.drop(lab3.loc[
    (lab3["LABEL"]=="Base Excess") & (lab3["VALUENUM"]>50)
].index)

# Hemoglobin: < 25 g/dL（正常约12-18）
lab3 = lab3.drop(lab3.loc[
    (lab3["LABEL"]=="Hemoglobin") & (lab3["VALUENUM"]>25)
].index)

# Potassium: < 30 mmol/L（正常3.5-5）
lab3 = lab3.drop(lab3.loc[
    (lab3["LABEL"]=="Potassium") & (lab3["VALUENUM"]>30)
].index)
```

#### 4. 时间处理

##### 4.1 参考时间设置
```python
# 每次入院的第一个检验时间作为参考点（t=0）
ref_time = lab4.groupby("HADM_ID")["CHARTTIME"].min()
lab5 = pd.merge(ref_time.to_frame(name="REF_TIME"), lab4, 
                left_index=True, right_on="HADM_ID")
lab5["TIME_STAMP"] = lab5["CHARTTIME"] - lab5["REF_TIME"]
```

##### 4.2 时间窗口截取
```python
# 只保留前48小时的数据
lab_short = lab_short.loc[lab_short["TIME_STAMP"] < timedelta(hours=48)]
```

#### 5. 时间分桶（Time Binning）

**目的**: 将不规则采样转换为规则网格

##### 5.1 分桶策略分析
```python
# 测试不同分桶因子的碰撞率
bins_num = range(1, 10)  # 每小时1-10个bins
for bin_k in bins_num:
    lab_short_binned["TIME_STAMP_Bin"] = round(
        lab_short_binned["TIME_STAMP"].dt.total_seconds() * bin_k / 3600
    ).astype(int)
    # 计算碰撞率（多个测量落入同一bin）
    hits_prop = lab_short_binned.duplicated(
        subset=["HADM_ID", "LABEL_CODE", "TIME_STAMP_Bin"]
    ).sum() / len(lab_short_binned.index)
```

**选择**: 每小时2个bins（即30分钟分辨率）
- 碰撞率低（<5%）
- 时间分辨率适中
- 数据稀疏度可接受

##### 5.2 应用分桶
```python
# 将秒转换为bin索引
lab_short["TIME_STAMP"] = round(
    lab_short["TIME_STAMP"].dt.total_seconds() * 2 / 3600
).astype(int)
# 结果：时间戳变为整数 [0, 1, 2, ..., 96]（48小时×2）
```

##### 5.3 重复值处理
对于落入同一bin的多个测量，在后续DataMerging阶段取平均值。

#### 6. 标签编码
```python
# 为每个实验室指标分配唯一整数ID
label_dict = dict(zip(
    list(lab5["LABEL"].unique()), 
    range(len(list(lab5["LABEL"].unique())))
))
lab5["LABEL_CODE"] = lab5["LABEL"].map(label_dict)
```

### 输出文件
- **`LAB_processed.csv`**: 清洗后的实验室数据
  - 字段: HADM_ID, SUBJECT_ID, CHARTTIME, LABEL, VALUENUM, VALUEUOM
  
- **`lab_events_short.csv`**: 时间分桶后的数据
  - 字段: HADM_ID, LABEL_CODE, TIME_STAMP (整数), VALUENUM, DEATHTAG

- **`death_tags.csv`**: 死亡标签
  - 字段: HADM_ID, DEATHTAG

### 数据特征
- **患者数**: ~22,000
- **实验室指标数**: 39
- **时间bins**: 96 (48小时 × 每小时2bins)
- **数据稀疏度**: ~2-3%（大部分时间-特征组合无测量）
- **平均每患者测量数**: 100-200条

### 时间分桶效果示意

```
原始数据（不规则采样）:
时间    | 0:05 | 0:37 | 1:12 | 2:48 | ...
Glucose | 120  | 115  | 110  | 95   | ...

分桶后（每小时2bins）:
Bin     | 0    | 1    | 2    | 5    | ...
Glucose | 120  | 115  | 110  | 95   | ...
```

---

## Notebook 3: Outputs.ipynb

### 功能概述
处理患者的输出事件数据（如尿液、引流液等）。

### 输入文件
- `OUTPUTEVENTS.csv`: 输出事件记录
- `D_ITEMS.csv`: 项目字典
- `Admissions_processed.csv`: 筛选后的入院记录

### 处理流程

#### 1. 数据加载与初步筛选

```python
outputs = pd.read_csv("OUTPUTEVENTS.csv")

# 检查错误标记（MIMIC中ISERROR标记数据录入错误）
assert(len(outputs.loc[outputs["ISERROR"].notnull()].index)==0)

# 只保留筛选后的患者
adm_ids = list(adm["HADM_ID"])
outputs = outputs.loc[outputs["HADM_ID"].isin(adm_ids)]
```

#### 2. 特征选择

选择15个重要的输出类型（基于临床重要性和数据频率）：

```python
outputs_label_list = [
    # 胃肠道输出
    'Gastric Gastric Tube',  # 胃管引流
    'Stool Out Stool',       # 粪便
    'TF Residual',           # 管饲残留
    
    # 泌尿系统输出
    'Foley',                 # 导尿管
    'Void',                  # 自主排尿
    'Urine Out Incontinent', # 失禁尿液
    'Condom Cath',           # 外置导尿
    
    # 透析/超滤
    'Ultrafiltrate Ultrafiltrate',  # 超滤液
    
    # 引流管输出
    'Chest Tube #1',         # 胸管1
    'Chest Tube #2',         # 胸管2
    'Jackson Pratt #1',      # JP引流管
    
    # 其他
    'Fecal Bag',             # 粪便袋
    'Ostomy (output)',       # 造口输出
    'OR EBL',                # 手术失血量
    'Pre-Admission'          # 入院前
]
```

#### 3. 单位验证

```python
# 检查所有输出是否使用相同单位（应该都是mL）
outputs_3.groupby("LABEL")["VALUEUOM"].value_counts()
```

**结果**: 所有输出单位统一为 mL，无需转换。

#### 4. 异常值处理

##### 4.1 统计方法：4σ规则
```python
# 计算每个输出类型的统计量
out_desc = outputs_3.groupby("LABEL")["VALUE"].describe()

# 移除超过均值+4σ的值
for label in name_list:
    outputs_3 = outputs_3.drop(
        outputs_3.loc[
            (outputs_3["LABEL"]==label) & 
            (outputs_3["VALUE"] > (out_desc.loc[label,"mean"] + 4*out_desc.loc[label,"std"]))
        ].index
    ).copy()
```

##### 4.2 基于临床知识的上限
```python
# Foley（导尿管）: < 5500 mL
# 正常尿量：1-2L/天，极端情况可达5L
outputs_3 = outputs_3.drop(
    outputs_3.loc[(outputs_3["LABEL"]=="Foley") & (outputs_3["VALUE"]>5500)].index
)

# OR EBL（手术失血量）: < 5000 mL
# 大量失血但超过5L通常是记录错误
outputs_3 = outputs_3.drop(
    outputs_3.loc[(outputs_3["LABEL"]=="OR EBL") & (outputs_3["VALUE"]>5000)].index
)

# Pre-Admission: 0-5000 mL
# 入院前的输出记录
outputs_3 = outputs_3.drop(
    outputs_3.loc[(outputs_3["LABEL"]=="Pre-Admission") & (outputs_3["VALUE"]<0)].index
)
outputs_3 = outputs_3.drop(
    outputs_3.loc[(outputs_3["LABEL"]=="Pre-Admission") & (outputs_3["VALUE"]>5000)].index
)

# Void（自主排尿）: >= 0
# 移除负值
outputs_3 = outputs_3.drop(
    outputs_3.loc[(outputs_3["LABEL"]=="Void") & (outputs_3["VALUE"]<0)].index
)
```

##### 4.3 缺失值处理
```python
# 删除值为空的记录
outputs_3.dropna(subset=["VALUE"], inplace=True)
```

#### 5. 时间处理特点

**与Inputs不同**: 
- Outputs数据已经是时间戳格式（CHARTTIME）
- 无需进行时间分割（不像持续输注需要离散化）
- 每条记录代表一个时间点的累积输出量

```python
# 输出数据直接使用CHARTTIME
# 无需额外的时间处理（将在DataMerging中统一处理）
```

### 输出文件
- **`OUTPUTS_processed.csv`**: 清洗后的输出事件数据
  - 字段: HADM_ID, SUBJECT_ID, CHARTTIME, LABEL, VALUE (单位: mL)

### 数据特征
- **患者数**: ~22,000
- **输出类型数**: 15
- **主要输出类型**:
  - Foley (导尿管): 最常见，几乎所有ICU患者
  - Stool: 常见
  - Chest Tube: 胸外科患者
- **典型值范围**:
  - Foley: 0-500 mL/次（记录周期不定）
  - Stool: 0-1000 mL/次
  - Chest Tube: 0-500 mL/次

### 临床意义

输出监测在ICU中的重要性：

1. **液体平衡**: 输入-输出 = 净液体平衡
   - 正平衡：可能导致水肿、心衰恶化
   - 负平衡：可能导致脱水、肾功能恶化

2. **肾功能评估**: 尿量是肾功能的直接指标
   - < 0.5 mL/kg/h: 少尿，急性肾损伤警示
   - < 100 mL/12h: 无尿，严重肾衰

3. **其他**: 引流液量反映感染、出血等并发症

### 数据质量考虑

**记录频率不规则**:
- 有些输出（如Foley）可能每小时记录
- 有些输出（如Stool）可能一天只记录一次或几次
- 这种不规则性将在时间分桶时处理

**测量累积性**:
- 每次记录可能是自上次记录以来的累积量
- 也可能是某时间段的总量
- 需要注意在合并时避免重复计数

---

## 数据处理的关键技术决策

### 1. 时间分桶策略

| 特征     | 选择                              | 原因                                 |
| -------- | --------------------------------- | ------------------------------------ |
| 分桶频率 | 每小时2bins (LAB) / 60bins (完整) | 平衡时间分辨率和数据稀疏度           |
| 参考时间 | 每次入院的第一个事件              | 标准化不同患者的时间轴               |
| 时间窗口 | 48小时                            | 急性期预测窗口，足够捕获早期病情变化 |

### 2. 异常值处理原则

1. **统计方法**: 4σ规则（适用于正态分布）
2. **临床知识**: 基于生理学合理范围
3. **组合使用**: 先统计后临床

### 3. 缺失数据策略

- **特征选择阶段**: 保留覆盖率高的特征
- **个体记录**: 直接删除缺失值（不插补）
- **时间序列**: 保持稀疏表示（在模型中处理）

### 4. 数据质量保证

```python
# 一致性检查示例
assert(len(lab5.loc[lab5["TIME_STAMP"]<timedelta(hours=0)].index)==0)  # 无负时间戳
assert(sum(complete_df.duplicated(subset=["HADM_ID","LABEL_CODE","TIME"])==True)==0)  # 无重复
```

---

## 后续处理预览

在 `DataMerging.ipynb` 中，这三个数据源将：

1. **合并**: LAB + INPUTS + OUTPUTS + PRESCRIPTIONS → 完整时间序列
2. **统一时间轴**: 所有数据映射到同一时间bins
3. **处理重复**: 同一bin内多个值 → LAB取平均，INPUTS/OUTPUTS求和
4. **归一化**: Z-score标准化 (μ=0, σ=1)
5. **张量化**: 转换为 (患者 × 时间 × 特征) 三维张量
6. **数据集分割**: 训练/验证/测试集

---

## 总结

### 三个Notebook的核心贡献

| Notebook   | 主要任务               | 输出                                             | 患者数 | 特征数         |
| ---------- | ---------------------- | ------------------------------------------------ | ------ | -------------- |
| Admissions | 患者筛选、输入事件处理 | Admissions_processed.csv<br>INPUTS_processed.csv | 22,000 | 32 (inputs)    |
| LabEvents  | 实验室检验处理         | LAB_processed.csv<br>lab_events_short.csv        | 22,000 | 39 (lab tests) |
| Outputs    | 输出事件处理           | OUTPUTS_processed.csv                            | 22,000 | 15 (outputs)   |

### 数据质量提升

| 阶段          | 记录数（估计）   | 说明                 |
| ------------- | ---------------- | -------------------- |
| 原始MIMIC-III | 数百万至数千万条 | 所有患者所有事件     |
| 患者筛选后    | ~数百万条        | 22,000患者，所有事件 |
| 特征选择后    | ~数十万条        | 22,000患者，86个特征 |
| 清洗后        | ~数十万条        | 移除异常值和缺失值   |
| 时间窗口截取  | ~50-100万条      | 仅前48小时           |

### 技术亮点

1. **系统化的数据清洗流程**
   - 单位标准化
   - 多层次异常值检测
   - 时间一致性验证

2. **临床知识驱动**
   - 特征选择基于临床重要性
   - 异常值范围基于生理学
   - 时间窗口符合临床实践

3. **时间处理创新**
   - 不规则采样 → 规则bins
   - 持续事件的离散化
   - 统一的时间参考系统

### 为机器学习准备的优势

1. **数据质量高**: 系统化清洗，异常值少
2. **特征工程**: 时间bins使得时间序列模型易于应用
3. **标签明确**: 死亡标签清晰，适合监督学习
4. **时间对齐**: 所有患者的时间序列对齐到相同参考点

---

## Notebook 4: Prescriptions.ipynb

### 功能概述
处理药物处方数据，记录患者在ICU期间的用药情况。

### 输入文件
- `PRESCRIPTIONS.csv`: 处方记录
- `Admissions_processed.csv`: 筛选后的入院记录

### 处理流程

#### 1. 数据加载与患者筛选

```python
presc = pd.read_csv("PRESCRIPTIONS.csv")

# 只保留筛选后的患者
adm_ids = list(adm["HADM_ID"])
presc = presc.loc[presc["HADM_ID"].isin(adm_ids)]
```

#### 2. 药物选择

基于论文中的药物列表，选择10种常用药物：

```python
drugs_list = [
    "Aspirin",                      # 阿司匹林 - 抗血小板
    "Bisacodyl",                    # 比沙可啶 - 泻药
    "Docusate Sodium",              # 多库酯钠 - 大便软化剂
    "D5W",                          # 5%葡萄糖水
    "Humulin-R Insulin",            # 人胰岛素
    "Potassium Chloride",           # 氯化钾 - 电解质补充
    "Magnesium Sulfate",            # 硫酸镁 - 电解质补充
    "Metoprolol Tartrate",          # 美托洛尔 - β受体阻滞剂
    "Sodium Chloride 0.9% Flush",   # 生理盐水冲洗
    "Pantoprazole"                  # 泮托拉唑 - 质子泵抑制剂
]
```

**选择依据**:
- 高频使用（覆盖大量患者）
- 临床重要性（常规ICU用药）
- 不同药理类别的代表

#### 3. 单位标准化

**挑战**: 处方数据的剂量和单位多样化

```python
# 删除无单位的记录
presc2 = presc2.drop(presc2.loc[presc2["DOSE_UNIT_RX"].isnull()].index)

# 统一mL表示（ml → mL）
presc2.loc[(presc2["DRUG"]=="D5W") & (presc2["DOSE_UNIT_RX"]=="ml"), "DOSE_UNIT_RX"] = "mL"
presc2.loc[(presc2["DRUG"]=="Sodium Chloride 0.9% Flush") & 
           (presc2["DOSE_UNIT_RX"]=="ml"), "DOSE_UNIT_RX"] = "mL"

# 删除单位不一致的记录
# 示例：Magnesium Sulfate只保留gm单位
presc2 = presc2.drop(
    presc2.loc[(presc2["DRUG"]=="Magnesium Sulfate") & 
               (presc2["DOSE_UNIT_RX"]!="gm")].index
)

# 其他药物类似处理
# Insulin → UNIT
# Potassium Chloride → mEq
# Aspirin, Bisacodyl, Pantoprazole → mg
```

**最终单位映射**:
| 药物                       | 单位 |
| -------------------------- | ---- |
| Aspirin                    | mg   |
| Bisacodyl                  | mg   |
| Docusate Sodium            | mg   |
| D5W                        | mL   |
| Humulin-R Insulin          | UNIT |
| Potassium Chloride         | mEq  |
| Magnesium Sulfate          | gm   |
| Metoprolol Tartrate        | mg   |
| Sodium Chloride 0.9% Flush | mL   |
| Pantoprazole               | mg   |

#### 4. 剂量值处理

**问题**: DOSE_VAL_RX字段是字符串类型，包含范围值

```python
# 原始数据示例
# "100"      → 100
# "50-100"   → 75 (取平均)
# "10-"      → 10 (缺失第二个值，使用第一个)

# 处理范围值（"xx-yy"格式）
range_df = presc2.loc[presc2["DOSE_VAL_RX"].str.contains("-", na=False)].copy()
range_df["First_digit"] = range_df["DOSE_VAL_RX"].str.split("-").str[0].astype(float)
range_df["Second_digit"] = range_df["DOSE_VAL_RX"].str.split("-").str[1]

# 处理空字符串（"10-"情况）
range_df.loc[range_df["Second_digit"]=="", 'Second_digit'] = \
    range_df.loc[range_df["Second_digit"]=="", 'First_digit']
range_df["Second_digit"] = range_df["Second_digit"].astype(float)

# 计算平均值
range_df["mean"] = (range_df["First_digit"] + range_df["Second_digit"]) / 2
range_df["DOSE_VAL_RX"] = range_df["mean"]

# 处理单一数值
presc3 = presc2.drop(presc2.loc[presc2["DOSE_VAL_RX"].str.contains("-", na=False)].index)
presc3["DOSE_VAL_RX"] = pd.to_numeric(presc3["DOSE_VAL_RX"], errors="coerce")
presc3.dropna(subset=["DOSE_VAL_RX"], inplace=True)

# 合并
presc2 = pd.concat([presc3, range_df], ignore_index=True)
```

#### 5. 异常值处理

##### 5.1 统计方法：4σ规则
```python
presc_desc = presc2.groupby("DRUG")["DOSE_VAL_RX"].describe()
name_list = list(presc_desc.loc[presc_desc["count"]!=0].index)

for label in name_list:
    presc2 = presc2.drop(
        presc2.loc[
            (presc2["DRUG"]==label) & 
            (presc2["DOSE_VAL_RX"] > (presc_desc.loc[label,"mean"] + 4*presc_desc.loc[label,"std"]))
        ].index
    ).copy()
```

**效果**: 移除极端异常剂量，如：
- Aspirin > 1000 mg（正常100-325 mg）
- Insulin > 200 UNIT（正常5-20 UNIT）

#### 6. 时间处理

```python
# 使用处方开始日期作为CHARTTIME
presc2['CHARTTIME'] = pd.to_datetime(presc2["STARTDATE"], format='%Y-%m-%d %H:%M:%S')

# 为避免与其他表的标签冲突，添加"Drug"后缀
presc2["DRUG"] = presc2["DRUG"] + " Drug"
```

**时间字段说明**:
- `STARTDATE`: 处方开始日期
- `ENDDATE`: 处方结束日期（可能为空）
- 用于后续DataMerging的时间对齐

#### 7. 处方持续时间分析

```python
# 分析处方持续时间
presc2['STARTDATE'] = pd.to_datetime(presc2["STARTDATE"])
presc2['ENDDATE'] = pd.to_datetime(presc2["ENDDATE"])
presc2["ELAPSED_TIME"] = presc2["ENDDATE"] - presc2["STARTDATE"]
presc2["ELAPSED_DAYS"] = presc2["ELAPSED_TIME"].dt.days
```

**观察**:
- 大部分处方持续1-7天
- 部分处方无结束日期（长期用药或数据缺失）
- 在DataMerging中只使用开始时间

### 输出文件
- **`PRESCRIPTIONS_processed.csv`**: 清洗后的处方数据
  - 字段: HADM_ID, SUBJECT_ID, CHARTTIME, DRUG (带"Drug"后缀), DOSE_VAL_RX (数值型)

### 数据特征
- **患者数**: ~22,000
- **药物类型数**: 10
- **记录特点**:
  - 处方频率低于实验室检验
  - 某些药物（如Insulin）可能一天多次调整
  - 某些药物（如Aspirin）可能整个住院期间保持不变

### 临床意义

处方数据的重要性：

1. **治疗反应**: 药物使用变化反映病情变化
   - Insulin剂量↑ → 血糖控制困难
   - β受体阻滞剂 → 心血管疾病管理

2. **并发症指示**:
   - 抗生素使用 → 感染
   - 电解质补充 → 代谢紊乱
   - 质子泵抑制剂 → 消化道保护

3. **疾病严重程度**:
   - 用药种类↑ → 病情复杂
   - 剂量变化频繁 → 病情不稳定

### 数据质量考虑

**与Inputs的区别**:
| 特征       | Inputs (INPUTEVENTS) | Prescriptions (PRESCRIPTIONS) |
| ---------- | -------------------- | ----------------------------- |
| 记录内容   | 实际给药记录         | 处方医嘱                      |
| 时间精度   | 精确到分钟           | 通常到小时或天                |
| 剂量信息   | 实际输入量           | 处方剂量（可能与实际不同）    |
| 覆盖范围   | 主要是静脉用药       | 包括口服、静脉等所有途径      |
| 数据完整性 | 较高                 | 可能有缺失                    |

**注意事项**:
- Prescriptions记录的是医嘱，不一定是实际执行
- 与Inputs可能有重叠（同一药物不同记录系统）
- 在合并时需要注意去重

---

## Notebook 5: DataMerging.ipynb

### 功能概述
这是整个预处理流程的核心，将所有清洗后的数据源合并，进行时间对齐、归一化和数据集划分。

### 输入文件
- `LAB_processed.csv`: 实验室检验数据（39个特征）
- `INPUTS_processed.csv`: 输入事件数据（32个特征）
- `OUTPUTS_processed.csv`: 输出事件数据（15个特征）
- `PRESCRIPTIONS_processed.csv`: 处方数据（10个特征）
- `Admissions_processed.csv`: 入院信息和死亡标签
- `DIAGNOSES_ICD.csv`: ICD9诊断代码

### 处理流程

#### 阶段1: 数据合并

##### 1.1 列名标准化

```python
# 将所有数据源的值列统一为VALUENUM
inputs_df["VALUENUM"] = inputs_df["AMOUNT"]
outputs_df["VALUENUM"] = outputs_df["VALUE"]
presc_df["VALUENUM"] = presc_df["DOSE_VAL_RX"]

# 将所有标签列统一为LABEL
presc_df["LABEL"] = presc_df["DRUG"]

# 添加数据来源标签
inputs_df["Origin"] = "Inputs"
lab_df["Origin"] = "Lab"
outputs_df["Origin"] = "Outputs"
presc_df["Origin"] = "Prescriptions"
```

##### 1.2 合并数据

```python
# 纵向合并四个数据源
merged_df = pd.concat([inputs_df, lab_df, outputs_df, presc_df], ignore_index=True)

# 验证标签唯一性（确保没有重复名称）
assert(merged_df["LABEL"].nunique() == 
       (inputs_df["LABEL"].nunique() + 
        lab_df["LABEL"].nunique() + 
        outputs_df["LABEL"].nunique() + 
        presc_df["LABEL"].nunique()))
```

**合并结果**: 96种时间序列特征（39+32+15+10）

#### 阶段2: 时间对齐

##### 2.1 确定参考时间

```python
# 将时间字符串转换为datetime
merged_df['CHARTTIME'] = pd.to_datetime(merged_df["CHARTTIME"], format='mixed')

# 每个入院ID的最早事件时间作为t=0
ref_time = merged_df.groupby("HADM_ID")["CHARTTIME"].min()

# 计算相对时间戳
merged_df_1 = pd.merge(ref_time.to_frame(name="REF_TIME"), 
                       merged_df, 
                       left_index=True, 
                       right_on="HADM_ID")
merged_df_1["TIME_STAMP"] = merged_df_1["CHARTTIME"] - merged_df_1["REF_TIME"]

# 验证：确保没有负时间戳
assert(len(merged_df_1.loc[merged_df_1["TIME_STAMP"]<timedelta(hours=0)].index)==0)
```

**效果**: 所有患者的时间轴对齐到相同起点

##### 2.2 标签编码

```python
# 为每个特征分配唯一整数ID
label_dict = dict(zip(
    list(merged_df_1["LABEL"].unique()),
    range(len(list(merged_df_1["LABEL"].unique())))
))
merged_df_1["LABEL_CODE"] = merged_df_1["LABEL"].map(label_dict)

# 保存标签字典
label_dict_df.to_csv(outfile_path + "label_dict.csv")
```

##### 2.3 时间窗口截取

```python
# 只保留前48小时的数据
merged_df_short = merged_df_short.loc[
    merged_df_short["TIME_STAMP"] < timedelta(hours=48)
]

print(f"Number of patients: {merged_df_short['HADM_ID'].nunique()}")
```

#### 阶段3: 时间分桶

##### 3.1 选择分桶策略

```python
# 测试不同分桶因子的碰撞率
bins_num = range(1, 60)
hits_vec = []

for bin_k in bins_num:
    # 计算时间bin
    merged_df_short_binned["TIME_STAMP_Bin"] = round(
        merged_df_short_binned["TIME_STAMP"].dt.total_seconds() * bin_k / 3600
    ).astype(int)
    
    # 计算碰撞率（多个测量落入同一bin的比例）
    hits_prop = merged_df_short_binned.duplicated(
        subset=["HADM_ID", "LABEL_CODE", "TIME_STAMP_Bin"]
    ).sum() / len(merged_df_short_binned.index)
    hits_vec.append(hits_prop)

# 绘制碰撞率曲线
plt.plot(bins_num, hits_vec)
plt.xlabel("Number of bins/hour")
plt.ylabel("% of hits")
```

**选择**: 60 bins/hour（即每分钟1个bin）
- 碰撞率适中（约5-10%）
- 时间分辨率高
- 48小时 = 2880个bins

##### 3.2 应用分桶并处理重复

```python
bin_k = 60
merged_df_short["TIME"] = round(
    merged_df_short["TIME_STAMP"].dt.total_seconds() * bin_k / 3600
).astype(int)
```

**关键**: 不同来源的数据采用不同的聚合策略

##### 3.3 按数据来源聚合

```python
# Lab数据：取平均值（同一时间多次测量）
lab_subset = merged_df_short.loc[merged_df_short["Origin"]=="Lab"]
lab_subset["KEY_ID"] = (lab_subset["HADM_ID"].astype(str) + "/" + 
                        lab_subset["TIME"].astype(str) + "/" + 
                        lab_subset["LABEL_CODE"].astype(str))
lab_subset_s = lab_subset.groupby("KEY_ID")["VALUENUM"].mean()

# Inputs数据：求和（累积输入量）
input_subset = merged_df_short.loc[merged_df_short["Origin"]=="Inputs"]
input_subset["KEY_ID"] = ...  # 同上
input_subset_s = input_subset.groupby("KEY_ID")["VALUENUM"].sum()

# Outputs数据：求和（累积输出量）
output_subset = merged_df_short.loc[merged_df_short["Origin"]=="Outputs"]
output_subset_s = output_subset.groupby("KEY_ID")["VALUENUM"].sum()

# Prescriptions数据：求和（累积剂量）
presc_subset = merged_df_short.loc[merged_df_short["Origin"]=="Prescriptions"]
presc_subset_s = presc_subset.groupby("KEY_ID")["VALUENUM"].sum()
```

**聚合策略总结**:
| 数据源        | 聚合方法      | 理由               |
| ------------- | ------------- | ------------------ |
| Lab           | 平均值 (mean) | 多次测量应取代表值 |
| Inputs        | 求和 (sum)    | 累积输入量         |
| Outputs       | 求和 (sum)    | 累积输出量         |
| Prescriptions | 求和 (sum)    | 累积剂量           |

##### 3.4 合并并去重

```python
# 去除重复记录
lab_s = lab_s.drop_duplicates(subset=["HADM_ID", "LABEL_CODE", "TIME"])
input_s = input_s.drop_duplicates(subset=["HADM_ID", "LABEL_CODE", "TIME"])
output_s = output_s.drop_duplicates(subset=["HADM_ID", "LABEL_CODE", "TIME"])
presc_s = presc_s.drop_duplicates(subset=["HADM_ID", "LABEL_CODE", "TIME"])

# 合并所有数据
complete_df = pd.concat([lab_s, input_s, output_s, presc_s], ignore_index=True)

# 验证：确保没有重复
assert(sum(complete_df.duplicated(subset=["HADM_ID","LABEL_CODE","TIME"])==True)==0)
```

##### 3.5 患者筛选

```python
# 移除观测次数少于50的患者（数据太少）
id_counts = complete_df.groupby("HADM_ID").count()
id_list = list(id_counts.loc[id_counts["TIME"]<50].index)
complete_df = complete_df.drop(
    complete_df.loc[complete_df["HADM_ID"].isin(id_list)].index
)
```

#### 阶段4: 创建唯一患者ID

```python
# 为每个HADM_ID分配唯一整数ID
unique_ids = np.arange(complete_df["HADM_ID"].nunique())
np.random.shuffle(unique_ids)  # 随机打乱
d = dict(zip(complete_df["HADM_ID"].unique(), unique_ids))

complete_df["UNIQUE_ID"] = complete_df["HADM_ID"].map(d)
```

**目的**: 
- 保护患者隐私
- 简化索引
- 便于数据集划分

#### 阶段5: 添加协变量（ICD9诊断代码）

```python
# 加载诊断代码
ICD_diag = pd.read_csv(file_path + "DIAGNOSES_ICD.csv")
main_diag = ICD_diag.loc[ICD_diag["SEQ_NUM"]==1]  # 只选主诊断

# 合并
complete_tensor = pd.merge(complete_df, 
                          main_diag[["HADM_ID","ICD9_CODE"]], 
                          on="HADM_ID")

# 只保留前3位数字（疾病大类）
complete_tensor["ICD9_short"] = complete_tensor["ICD9_CODE"].astype(str).str[:3]

# One-hot编码
hot_encodings = pd.get_dummies(complete_tensor["ICD9_short"])
complete_tensor[hot_encodings.columns] = hot_encodings
```

**ICD9编码示例**:
- `250` → 糖尿病
- `410` → 急性心肌梗死
- `428` → 心力衰竭
- `486` → 肺炎

#### 阶段6: 数据归一化

```python
# Z-score标准化：(x - μ) / σ
d_mean = dict(complete_tensor_nocov.groupby("LABEL_CODE")["VALUENUM"].mean())
d_std = dict(complete_tensor_nocov.groupby("LABEL_CODE")["VALUENUM"].std())

complete_tensor_nocov["MEAN"] = complete_tensor_nocov["LABEL_CODE"].map(d_mean)
complete_tensor_nocov["STD"] = complete_tensor_nocov["LABEL_CODE"].map(d_std)
complete_tensor_nocov["VALUENORM"] = (
    (complete_tensor_nocov["VALUENUM"] - complete_tensor_nocov["MEAN"]) / 
    complete_tensor_nocov["STD"]
)
```

**归一化的重要性**:
- 不同特征的量纲不同（如Glucose: 70-110 mg/dL vs WBC: 4-11 K/μL）
- 神经网络对输入范围敏感
- 标准化后所有特征均值为0，标准差为1

#### 阶段7: 数据集划分

##### 7.1 随机划分（用于张量分解）

```python
# 90% 训练+验证，10% 测试
df_train, df_test = train_test_split(complete_tensor_nocov, test_size=0.1)

# 创建3个训练/验证fold（用于交叉验证）
df_train1, df_val1 = train_test_split(df_train, test_size=0.2)  # Fold 1
df_train2, df_val2 = train_test_split(df_train, test_size=0.2)  # Fold 2
df_train3, df_val3 = train_test_split(df_train, test_size=0.2)  # Fold 3
```

**验证一致性**:
```python
# 确保验证集中的所有患者和特征在训练集中都出现过
assert(len(df_val1.loc[~df_val1["UNIQUE_ID"].isin(df_train1["UNIQUE_ID"])].index)==0)
assert(len(df_val1.loc[~df_val1["LABEL_CODE"].isin(df_train1["LABEL_CODE"])].index)==0)
```

**为什么3个fold**:
- 交叉验证提高模型稳定性
- 每个fold独立随机划分
- 最终结果取平均

##### 7.2 患者级划分（用于LSTM/GRU-ODE）

```python
test_prop = 0.1
val_prop = 0.2

sorted_unique_ids = np.sort(unique_ids)
train_unique_ids = sorted_unique_ids[:int((1-test_prop)*(1-val_prop)*len(unique_ids))]
val_unique_ids = sorted_unique_ids[int((1-test_prop)*(1-val_prop)*len(unique_ids)):
                                   int((1-test_prop)*len(unique_ids))]
test_unique_ids = sorted_unique_ids[int((1-test_prop)*len(unique_ids)):]
```

**区别**:
| 划分方式     | 随机划分                   | 患者级划分 |
| ------------ | -------------------------- | ---------- |
| 划分单位     | 观测记录                   | 患者       |
| 用途         | 张量分解                   | 序列模型   |
| 特点         | 同一患者可能在训练和测试集 | 患者不重叠 |
| 数据泄露风险 | 较高                       | 低         |

#### 阶段8: 创建死亡标签

```python
admissions = pd.read_csv(file_path + "Admissions_processed.csv")
death_tags_s = admissions.groupby("HADM_ID")["DEATHTAG"].unique().astype(int)
death_tags_df = death_tags_s.to_frame().reset_index()
death_tags_df["UNIQUE_ID"] = death_tags_df["HADM_ID"].map(d)
death_tags_df.sort_values(by="UNIQUE_ID", inplace=True)
```

#### 阶段9: 创建协变量矩阵

```python
# 每个患者一行，ICD9编码作为列
covariates = complete_tensor.groupby("UNIQUE_ID").nth(0)[list(hot_encodings.columns)]
```

**协变量矩阵结构**:
```
| UNIQUE_ID | 250 | 410 | 428 | ... | age | EMERGENCY_ADMISSION |
| --------- | --- | --- | --- | --- | --- | ------------------- |
| 0         | 1   | 0   | 1   | ... | 65  | 1                   |
| 1         | 0   | 1   | 0   | ... | 72  | 1                   |
| ...       | ... | ... | ... | ... | ... | ...                 |
```

### 输出文件

#### 张量分解数据集
```
complete_tensor.csv              # 完整数据
complete_tensor_train1-3.csv     # 3个训练fold
complete_tensor_val1-3.csv       # 3个验证fold
complete_tensor_test.csv         # 测试集
complete_covariates.csv          # ICD9协变量
complete_death_tags.csv          # 死亡标签
label_dict.csv                   # 特征名称-编码映射
UNIQUE_ID_dict.csv               # HADM_ID-UNIQUE_ID映射
```

#### LSTM/GRU-ODE数据集
```
LSTM_tensor_train.csv            # 训练数据（患者级划分）
LSTM_tensor_val.csv              # 验证数据
LSTM_tensor_test.csv             # 测试数据
LSTM_death_tags_train/val/test.csv  # 对应的标签
LSTM_covariates_train/val/test.csv  # 对应的协变量
mean_features.csv                # 每个特征的均值（用于插补）
```

### 数据结构示意

#### 张量数据格式
```csv
UNIQUE_ID,LABEL_CODE,TIME_STAMP,VALUENUM,MEAN,STD,VALUENORM
0,5,10,120.5,100.2,15.3,1.33
0,5,25,115.2,100.2,15.3,0.98
0,12,10,7.5,7.2,1.5,0.20
...
```

#### 特征:
- `UNIQUE_ID`: 患者唯一标识 (0-21999)
- `LABEL_CODE`: 特征编码 (0-95，对应96个特征)
- `TIME_STAMP`: 时间bin (0-2879，对应48小时×60bins/h)
- `VALUENUM`: 原始值
- `VALUENORM`: 归一化值

### 数据稀疏性分析

```python
# 理论最大数据点数
max_possible = n_patients × n_features × n_timebins
             = 22,000 × 96 × 2880
             = 6,082,560,000

# 实际数据点数
actual = len(complete_df)
       ≈ 5,000,000

# 稀疏度
sparsity = actual / max_possible
         ≈ 0.08%  (即99.92%的数据缺失)
```

**这是正常的**:
- 不是所有检验每分钟都做
- 不是所有药物每个患者都用
- 时间序列本质上是稀疏的

### SAPSII基准模型

代码中包含SAPSII（Simplified Acute Physiology Score II）计算：

```python
saps["SUM_score"] = saps[[
    "hr_score", "sysbp_score", "temp_score", 
    "pao2fio2_score", "uo_score", "bun_score",
    "wbc_score", "potassium_score", "sodium_score",
    "bicarbonate_score", "bilirubin_score", "gcs_score"
]].sum(axis=1)

# SAPSII死亡率预测公式
saps["X"] = -7.7631 + 0.0737 * saps["SUM_score"] + 0.9971 * np.log(saps["SUM_score"] + 1)
saps["PROB"] = np.exp(saps["X"]) / (1 + np.exp(saps["X"]))

# 计算AUC
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_true, y_pred)
```

**SAPSII的意义**:
- 标准临床评分系统
- 作为机器学习模型的基准
- 通常AUC约0.75-0.85

---

## Notebook 6: Extra_covariates.ipynb

### 功能概述
添加额外的协变量，这些协变量不是时间序列，而是患者级别的静态特征。

### 输入文件
- `UNIQUE_ID_dict.csv`: 患者ID映射
- `Admissions_processed.csv`: 入院信息
- `complete_covariates.csv`: ICD9协变量矩阵

### 处理流程

#### 1. 住院时长计算

```python
# 加载数据
unique_id_dict = pd.read_csv(outfile_path + "UNIQUE_ID_dict.csv")
admissions = pd.read_csv(file_path + "Admissions_processed.csv")
adm = admissions.loc[admissions["HADM_ID"].isin(hadm_ids)]

# 计算住院时长（天）
adm["admit"] = pd.to_datetime(adm["ADMITTIME"])
adm["disc"] = pd.to_datetime(adm["DISCHTIME"])
adm["duration"] = adm["disc"] - adm["admit"]
adm["Value"] = adm["duration"].dt.days

# 映射到UNIQUE_ID
d = unique_id_dict.set_index("HADM_ID").to_dict()["UNIQUE_ID"]
adm["UNIQUE_ID"] = adm["HADM_ID"].map(d)

# 保存
durations_df = adm[["UNIQUE_ID", "Value"]].copy()
durations_df.sort_values(by="UNIQUE_ID", inplace=True)
durations_df.to_csv(outfile_path + "complete_durations_vec.csv")
```

**临床意义**:
- 住院时长是疾病严重程度的指标
- 可作为回归任务的标签
- 与死亡率相关（但非因果关系）

#### 2. 年龄计算

```python
# 计算入院时年龄
adm["birth_date"] = pd.to_datetime(adm["DOBTIME"])
adm["admit_date"] = pd.to_datetime(adm["ADMITTIME"])
adm["age"] = (adm["admit_date"] - adm["birth_date"]).dt.days // 365
```

**注意事项**:
- MIMIC中>89岁患者的出生日期被修改（隐私保护）
- 可能出现>100岁的年龄（实际是>89岁）
- 年龄是死亡率的强预测因子

#### 3. 入院类型编码

```python
# 创建入院类型的哑变量
adm["EMERGENCY_ADMISSION"] = 1 * (adm["ADMISSION_TYPE"]=="EMERGENCY")
adm["ELECTIVE_ADMISSION"] = 1 * (adm["ADMISSION_TYPE"]=="ELECTIVE")
adm["URGENT_ADMISSION"] = 1 * (adm["ADMISSION_TYPE"]=="URGENT")
```

**入院类型说明**:
| 类型      | 含义     | 预后     |
| --------- | -------- | -------- |
| EMERGENCY | 急诊入院 | 通常较差 |
| URGENT    | 紧急入院 | 中等     |
| ELECTIVE  | 择期入院 | 通常较好 |

**临床意义**:
- 急诊入院的患者通常病情更重
- 择期手术患者经过充分准备
- 是死亡率的独立预测因子

#### 4. 合并协变量

```python
# 加载现有协变量（ICD9编码）
cov = pd.read_csv(infile_path + "complete_covariates.csv")

# 合并新协变量
cov = cov.merge(
    adm[["age", "EMERGENCY_ADMISSION", "ELECTIVE_ADMISSION", 
         "URGENT_ADMISSION", "UNIQUE_ID"]], 
    on="UNIQUE_ID"
)

# 保存扩展协变量
cov.to_csv(infile_path + "complete_covariates_extended.csv")
```

### 输出文件

- **`complete_covariates_extended.csv`**: 扩展协变量矩阵
  - ICD9编码（hundreds列）+ 年龄 + 入院类型（3列）
  
- **`complete_durations_vec.csv`**: 住院时长向量
  - 用于潜在的回归任务
  
- **`mean_features.csv`**: 特征均值向量
  - 用于缺失值插补

### 协变量矩阵最终结构

```
| UNIQUE_ID | ICD9_250 | ICD9_410 | ... | age | EMERGENCY | ELECTIVE | URGENT |
| --------- | -------- | -------- | --- | --- | --------- | -------- | ------ |
| 0         | 1        | 0        | ... | 65  | 1         | 0        | 0      |
| 1         | 0        | 1        | ... | 72  | 1         | 0        | 0      |
| 2         | 1        | 1        | ... | 58  | 0         | 1        | 0      |
| ...       | ...      | ...      | ... | ... | ...       | ...      | ...    |
```

**维度**: 
- 行数: ~22,000（患者数）
- 列数: ~300-500（ICD9编码数量）+ 4（age + 3个入院类型）

### 协变量的使用

在GRU-ODE-Bayes模型中：

1. **直接使用**:
   ```python
   # 协变量作为模型输入的一部分
   model_input = {
       'time_series': X,  # (batch, time, features)
       'covariates': C    # (batch, cov_dim)
   }
   ```

2. **嵌入**:
   ```python
   # 通过神经网络映射到低维空间
   cov_embedding = nn.Linear(cov_dim, embed_dim)(covariates)
   ```

3. **注意力机制**:
   ```python
   # 动态调整时间序列特征的权重
   attention_weights = softmax(time_series @ cov_embedding.T)
   ```

### K-Fold交叉验证准备

代码中包含K-Fold设置（虽然未完全实现）：

```python
from sklearn.model_selection import KFold

k = KFold(n_splits=3, random_state=22)
for train, val in k.split(np.arange(n_patients)):
    # 创建fold
    pass
```

**用途**:
- 更稳健的模型评估
- 减少随机性影响
- 与DataMerging中的3个fold对应

---

## 输出文件说明

### 完整文件清单

#### 中间处理文件（单一数据源）
```
Admissions_processed.csv         # 筛选后的入院记录 + 死亡标签
INPUTS_processed.csv             # 清洗后的输入事件（32特征）
OUTPUTS_processed.csv            # 清洗后的输出事件（15特征）
LAB_processed.csv                # 清洗后的实验室检验（39特征）
PRESCRIPTIONS_processed.csv      # 清洗后的处方（10特征）
```

#### 合并后的张量数据（随机划分）
```
complete_tensor.csv              # 完整合并数据
complete_tensor_train1.csv       # 训练集 Fold 1
complete_tensor_val1.csv         # 验证集 Fold 1
complete_tensor_train2.csv       # 训练集 Fold 2
complete_tensor_val2.csv         # 验证集 Fold 2
complete_tensor_train3.csv       # 训练集 Fold 3
complete_tensor_val3.csv         # 验证集 Fold 3
complete_tensor_test.csv         # 测试集
```

#### LSTM/GRU-ODE数据（患者级划分）
```
LSTM_tensor_train.csv            # 时间序列-训练
LSTM_tensor_val.csv              # 时间序列-验证
LSTM_tensor_test.csv             # 时间序列-测试
LSTM_death_tags_train.csv        # 标签-训练
LSTM_death_tags_val.csv          # 标签-验证
LSTM_death_tags_test.csv         # 标签-测试
LSTM_covariates_train.csv        # 协变量-训练
LSTM_covariates_val.csv          # 协变量-验证
LSTM_covariates_test.csv         # 协变量-测试
```

#### 协变量和元数据
```
complete_covariates.csv          # ICD9协变量矩阵
complete_covariates_extended.csv # 扩展协变量（+年龄+入院类型）
complete_death_tags.csv          # 死亡标签
complete_durations_vec.csv       # 住院时长
mean_features.csv                # 特征均值（用于插补）
label_dict.csv                   # 特征编码字典
UNIQUE_ID_dict.csv               # 患者ID映射
sapsii_processed.csv             # SAPSII评分
```

### 文件大小估计

| 文件类型         | 估计大小   | 说明                 |
| ---------------- | ---------- | -------------------- |
| 中间处理文件     | 500MB-2GB  | 取决于患者数和特征数 |
| 完整张量         | 200-500MB  | 稀疏表示             |
| 训练/验证/测试集 | 各50-200MB | 根据划分比例         |
| 协变量矩阵       | 10-50MB    | 患者数 × 协变量维度  |
| 元数据           | <5MB       | 字典和映射           |

### 数据加载示例

```python
import pandas as pd

# 加载训练数据
train_data = pd.read_csv("complete_tensor_train1.csv")
train_labels = pd.read_csv("complete_death_tags.csv")
train_covs = pd.read_csv("complete_covariates_extended.csv")
label_dict = pd.read_csv("label_dict.csv")

# 查看数据结构
print(f"训练数据: {train_data.shape}")
print(f"特征数: {label_dict.shape[0]}")
print(f"患者数: {train_labels.shape[0]}")
print(f"协变量维度: {train_covs.shape[1]}")
```

---

## 总结

### 完整处理流程回顾

```
原始MIMIC-III (数千万条记录)
    ↓
[1. Admissions] 患者筛选 (22,000患者) + Inputs处理 (32特征)
    ↓
[2. LabEvents] 实验室检验处理 (39特征)
    ↓
[3. Outputs] 输出事件处理 (15特征)
    ↓
[4. Prescriptions] 处方处理 (10特征)
    ↓
[5. DataMerging] 
   - 合并4个数据源 (96特征)
   - 时间对齐 (参考时间 = t0)
   - 时间分桶 (60 bins/hour)
   - 归一化 (Z-score)
   - 添加ICD9协变量
   - 数据集划分
    ↓
[6. Extra_covariates]
   - 添加年龄
   - 添加入院类型
   - 计算住院时长
    ↓
最终数据集：
├── 时间序列数据 (稀疏张量)
├── 死亡标签 (二分类)
├── 协变量矩阵 (ICD9 + 人口学)
└── 元数据 (特征字典等)
```

### 关键统计数据

| 指标           | 数值                                                    |
| -------------- | ------------------------------------------------------- |
| 最终患者数     | ~22,000                                                 |
| 时间序列特征数 | 96 (39 Lab + 32 Inputs + 15 Outputs + 10 Prescriptions) |
| 时间窗口       | 48小时                                                  |
| 时间分辨率     | 1分钟/bin (60 bins/hour)                                |
| 时间bins总数   | 2,880                                                   |
| ICD9协变量     | ~300-500个诊断类别                                      |
| 额外协变量     | 4 (年龄 + 3个入院类型)                                  |
| 死亡率         | ~12-15%                                                 |
| 数据稀疏度     | ~99.9% (正常现象)                                       |

### 六个Notebook的分工总结

| Notebook            | 输入                                        | 输出                                                     | 关键处理                                                 | 患者数  | 特征数    |
| ------------------- | ------------------------------------------- | -------------------------------------------------------- | -------------------------------------------------------- | ------- | --------- |
| 1. Admissions       | ADMISSIONS, PATIENTS, INPUTEVENTS_MV        | Admissions_processed<br>INPUTS_processed                 | 患者筛选<br>单位标准化<br>时间离散化                     | 22,000  | 32        |
| 2. LabEvents        | LABEVENTS, D_LABITEMS                       | LAB_processed                                            | 单位标准化<br>异常值处理<br>初步时间分桶                 | 22,000  | 39        |
| 3. Outputs          | OUTPUTEVENTS, D_ITEMS                       | OUTPUTS_processed                                        | 异常值处理<br>临床范围验证                               | 22,000  | 15        |
| 4. Prescriptions    | PRESCRIPTIONS                               | PRESCRIPTIONS_processed                                  | 剂量值解析<br>范围值处理<br>单位标准化                   | 22,000  | 10        |
| 5. DataMerging      | 上述4个processed文件<br>DIAGNOSES_ICD       | complete_tensor系列<br>LSTM_tensor系列<br>协变量<br>标签 | 数据合并<br>时间对齐<br>精细分桶<br>归一化<br>数据集划分 | ~21,000 | 96 + ICD9 |
| 6. Extra_covariates | complete_covariates<br>Admissions_processed | complete_covariates_extended<br>complete_durations_vec   | 年龄计算<br>入院类型编码<br>住院时长                     | ~21,000 | +4        |

### 数据质量保证措施

1. **多层次筛选**:
   - 患者级别：单次入院、年龄、停留时间
   - 特征级别：覆盖率、临床重要性
   - 记录级别：异常值、缺失值

2. **异常值检测**:
   - 统计方法：4σ规则
   - 临床知识：生理学合理范围
   - 数据一致性：时间戳验证

3. **数据验证**:
   ```python
   assert(无负时间戳)
   assert(无重复记录)
   assert(验证集患者在训练集中)
   assert(验证集特征在训练集中)
   ```

4. **标准化处理**:
   - 单位统一
   - Z-score归一化
   - 标签编码一致性

### 技术创新点

1. **时间处理**:
   - 多粒度分桶策略（2 bins/hour vs 60 bins/hour）
   - 碰撞率分析选择最优分桶
   - 持续事件的智能离散化

2. **聚合策略**:
   - 不同数据源采用不同聚合方法
   - Lab取平均，Inputs/Outputs/Prescriptions求和
   - 保留数据的物理意义

3. **双重划分**:
   - 随机划分：用于张量分解
   - 患者级划分：用于序列模型
   - 避免数据泄露

4. **协变量整合**:
   - 时间序列特征 + 静态协变量
   - ICD9诊断 + 人口学特征
   - 为多模态学习提供基础

### 为GRU-ODE-Bayes准备的数据特点

1. **不规则采样**:
   - 保留原始采样时间信息
   - 适合ODE-based模型

2. **稀疏表示**:
   - 只存储观测值
   - 节省内存，加快训练

3. **标准化接口**:
   - 统一的数据格式
   - 清晰的特征编码
   - 完整的元数据

4. **多任务支持**:
   - 分类：死亡预测
   - 回归：住院时长预测
   - 插补：缺失值填充

### 潜在改进方向

1. **数据增强**:
   - 时间平移
   - 噪声注入
   - 合成少数类样本

2. **特征工程**:
   - 趋势特征（增加/减少）
   - 交互特征（Lab × 药物）
   - 时间特征（小时、星期）

3. **不平衡处理**:
   - 死亡率~12%（不平衡）
   - 可使用SMOTE、加权损失等

4. **外部验证**:
   - 其他医院数据
   - 不同时间段数据
   - 提高泛化能力

### 使用建议

1. **模型选择**:
   - 稀疏数据 → GRU-ODE, Latent ODE
   - 规则采样 → LSTM, Transformer
   - 协变量丰富 → 多模态模型

2. **评估指标**:
   - AUC-ROC（主要）
   - AUC-PR（不平衡数据）
   - 校准曲线（预测概率）
   - 与SAPSII比较（基准）

3. **计算资源**:
   - 内存：16GB+ 推荐
   - GPU：训练深度模型
   - 时间：完整流程数小时

### 最后的话

这个预处理流程展示了：
- **系统性**: 从原始数据到模型就绪的完整pipeline
- **可重复性**: 详细的步骤和参数
- **临床导向**: 处处体现医学知识
- **工程实践**: 数据验证、异常处理、版本管理

通过这6个notebook，我们将混乱的临床数据库转化为结构化的机器学习数据集，为后续的深度学习模型训练奠定了坚实基础。

---

**文档版本**: v2.0  
**最后更新**: 2025-10-19  
**适用于**: MIMIC-III v1.4, GRU-ODE-Bayes 项目  
**作者**: AI Assistant  
**审阅**: 建议临床专家审阅
