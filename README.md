<p align="center">
  <img src="app.png" width="128"/>
</p>

# targetDetect 技术文档

## 目录

- [1. 文档目的](#1-文档目的)
- [2. 项目结构](#2-项目结构)
- [3. 系统架构](#3-系统架构)
- [4. 主流程说明](#4-主流程说明)
- [5. 模块说明](#5-模块说明)
  - [5.1 `main.py`](#51-mainpy)
  - [5.2 `common/config.py`](#52-commonconfigpy)
    - [摄像头与校正图](#摄像头与校正图)
    - [靶纸检测参数](#靶纸检测)
    - [红环检测参数](#红环检测)
    - [角点跟踪参数](#角点跟踪)
    - [原生加速后端](#原生加速后端)
  - [5.3 `vision/lighting.py`](#53-visionlightingpy)
  - [5.4 `vision/target_frame.py`](#54-visiontarget_framepy)
    - [角点排序](#角点排序)
    - [靶纸框重检](#靶纸框重检)
    - [角点光流跟踪](#角点光流跟踪)
    - [透视变换](#透视变换)
    - [跟踪 ROI](#跟踪-roi)
  - [5.5 `vision/target_center.py`](#55-visiontarget_centerpy)
    - [红色掩膜提取](#红色掩膜提取)
    - [中心拟合](#中心拟合)
    - [下采样](#下采样)
  - [5.6 `vision/kalman.py`](#56-visionkalmanpy)
  - [5.7 `control/serial_sender.py`](#57-controlserial_senderpy)
  - [5.8 屏幕数据展示](#58-屏幕数据展示)
  - [5.9 `control/gimbal_pid.py`](#59-controlgimbal_pidpy)
    - [IMU 姿态反馈](#imu-姿态反馈)
  - [5.10 `control/camera_attitude.py`](#510-controlcamera_attitudepy)
- [6. 坐标系统](#6-坐标系统)
- [7. 状态机](#7-状态机)
  - [靶纸框状态](#靶纸框状态)
  - [靶心状态](#靶心状态)
- [8. 耗时统计](#8-耗时统计)
- [9. 性能优化原则](#9-性能优化原则)
- [10. 关键算法约束](#10-关键算法约束)
  - [靶纸框约束](#靶纸框约束)
  - [角点跟踪约束](#角点跟踪约束)
  - [靶心约束](#靶心约束)
- [11. 部署与运行](#11-部署与运行)
  - [应用打包](#应用打包)
  - [GPIO 与补光灯](#gpio-与补光灯)
  - [摄像头](#摄像头)
- [12. 调参指南](#12-调参指南)
  - [靶纸检测失败](#靶纸检测失败)
  - [靶心低置信](#靶心低置信)
  - [边框乱动](#边框乱动)
  - [帧率下降](#帧率下降)
- [13. 故障排查](#13-故障排查)
  - [没有检测到靶框](#没有检测到靶框)
  - [靶心不准](#靶心不准)
  - [黄点不准](#黄点不准)
  - [补光灯一直变化](#补光灯一直变化)
- [14. 建议的后续维护方向](#14-建议的后续维护方向)
- [15. 新人快速上手顺序](#15-新人快速上手顺序)

## 1. 文档目的

本文档说明打靶目标检测项目的整体架构、每个模块的职责、核心算法、运行流程、参数配置及维护方法。目标读者包括首次接触项目的新开发者、部署人员和后续维护人员。

项目运行在 MaixCAM Pro 上，使用摄像头读取图像、OpenCV 检测靶纸和靶心、LK 光流跟踪靶纸角点，并将结果叠加到屏幕。

## 2. 项目结构

源码按功能分为三个包（每个包含 `__init__.py`）：

- `common/`：公共配置与日志输出
- `vision/`：图像处理与目标检测（补光、靶纸框、靶心、卡尔曼预测）
- `control/`：控制与通信（串口发送、云台 PID、相机姿态）

| 文件 | 主要职责 |
|---|---|
| `main.py` | Python 程序入口，调用 `app.main()` |
| `app.py` | 主循环、模块调度、状态展示、相机和屏幕 I/O、耗时统计 |
| `common/config.py` | 摄像头、几何、颜色、跟踪、置信度及补光参数 |
| `vision/lighting.py` | 根据画面亮度控制补光灯 |
| `vision/target_frame.py` | 靶纸框检测、角点排序、四角光流跟踪、透视变换和跟踪 ROI |
| `vision/target_center.py` | 红色同心环提取、靶心拟合和置信度判断 |
| `vision/native_backend.py` | C++ 动态库加载与 LK/透视/圆环拟合桥接 |
| `vision/kalman.py` | 目标丢失或低置信时的短时速度预测 |
| `control/serial_sender.py` | 串口初始化、结构化靶子数据和传输异常处理 |
| `control/gimbal_pid.py` | 二轴云台闭环 PID、积分限幅、输出限幅及转动方向打印 |
| `control/camera_attitude.py` | MaixCAM-Pro QMI8658 姿态读取、陀螺仪校准和 AHRS 欧拉角反馈 |
| `common/console_output.py` | 统一控制台状态日志格式 |
| `vision/native/targetdetect_accel.cpp` | C++/NEON 加速实现（LK、透视、圆环拟合） |
| `vision/native/build_with_maixcdk.sh` | MaixCDK 一键编译并同步 `.so`/`dl_lib` |
| `vision/native/maixcdk_project/` | MaixCDK 编译工程（组件化） |
| `app.yaml` | MaixApp 应用元数据及打包文件清单 |
| `app.png` | MaixApp 应用图标 |

## 3. 系统架构

系统由以下阶段组成：

```text
摄像头
  -> 图像读取和格式转换
  -> 场景亮度检测与补光
  -> 靶纸框检测 / 框间光流跟踪
  -> 四角透视校正
  -> 红色靶心提取与拟合
  -> 坐标回投影到原始画面
  -> 高置信输出或短时预测
  -> 二轴云台视觉外环和 IMU 姿态内环
  -> 屏幕叠加显示和耗时统计
```

核心原则：

1. 四角用于确定靶纸所在位置和透视几何。
2. 红色同心环用于确定靶心坐标。
3. 四角跟踪用于降低每帧重新识别四角的成本。
4. 靶心绿点表示本帧可置信测量；卡尔曼结果不替代绿点。
5. 卡尔曼主要用于靶心短时丢失或低置信时的预测。
6. 云台控制只使用 `TARGET LOCKED` 的高置信靶心，不使用低置信或卡尔曼预测点。

## 4. 主流程说明

入口为 `main.py`，它只调用 `app.main()`。实际运行逻辑全部位于 `app.py`。

主循环顺序：

1. 使用 `camera.Camera()` 读取 `640x480` 的 `BGR888` 图像。
2. 使用 `image.image2cv(..., copy=False)` 将 Maix 图像转换为 NumPy/OpenCV 数组。
3. `update_fill_light()` 检查画面中心区域亮度并控制补光灯。
4. 在现有跟踪 ROI 内运行靶纸检测，或使用角点光流跟踪。
5. 通过透视矩阵将靶纸校正为 `210x297` 像素的标准图。
6. 提取红色掩膜并拟合靶心。
7. 将校正图坐标通过逆透视矩阵投影回摄像头图像。
8. 高置信测量直接绘制绿色靶心；低置信或丢失时绘制橙色预测点。
9. 云台 PID 暂时停用（保留在 `control/gimbal_pid.py` 和配置中）；仅运行靶心解算、状态展示和串口坐标输出。
10. 绘制简化状态栏并调用 `display.Display.show()`。
11. 每秒输出一次模块耗时快照。

## 5. 模块说明

### 5.1 `main.py`

该文件是最薄的入口层：

- 导入 `app.main`
- 当脚本作为主程序运行时启动应用

通常不应在此文件中添加算法逻辑。

### 5.2 `common/config.py`

这是全局参数中心。常用参数分为以下几类。

#### 摄像头与校正图

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `CAMERA_WIDTH` | 640 | 摄像头宽度 |
| `CAMERA_HEIGHT` | 480 | 摄像头高度 |
| `CAMERA_FPS` | 30 | 摄像头目标帧率 |
| `RECT_WIDTH` | 210 | 校正图宽度 |
| `RECT_HEIGHT` | 297 | 校正图高度 |
| `PIXELS_PER_MM` | 1.0 | 校正图比例 |

校正图近似采用 1 像素对应 1 毫米，避免引入过大的中间图像。

#### 靶纸检测

| 参数 | 含义 |
|---|---|
| `FRAME_DETECT_SCALE` | 靶纸重检时的缩放比例 |
| `FRAME_ASPECT_TOLERANCE` | 候选框相对 A4 长宽比的最大误差 |
| `FRAME_HSV_LOW/HIGH` | 黑色靶框的 HSV 范围 |
| `FRAME_CLOSE_KERNEL` | 黑色掩膜形态学闭运算核 |

靶纸检测优先使用 HSV 黑色掩膜，失败时再使用灰度 Otsu 反色掩膜。

#### 红环检测

| 参数 | 含义 |
|---|---|
| `RING_RADII_PX` | 五个理论环半径：20、40、60、80、100 px |
| `RED_HSV_LOW1/HIGH1` | 红色低色相范围 |
| `RED_HSV_LOW2/HIGH2` | 红色高色相范围 |
| `RED_LOW_LIGHT_*` | 暗红候选和低照度补充阈值 |
| `RED_CLOSE_KERNEL` | 红色掩膜形态学闭运算核 |
| `FIT_RMS_WARNING_THRESHOLD` | 低置信 RMS 阈值 |

#### 角点跟踪

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `CORNER_TRACK_SCALE` | 0.25 | LK 使用的灰度图比例 |
| `CORNER_TRACK_MAX_FEATURES` | 24 | 每次最多提取特征数 |
| `CORNER_TRACK_MIN_FEATURES` | 12 | 可接受跟踪所需特征数 |
| `CORNER_DETECT_INTERVAL` | 15 | 周期重检角点的帧间隔 |
| `CORNER_TRACK_MAX_FRAMES` | 90 | 一次跟踪生命周期上限 |
| `CORNER_TRACK_MAX_PREDICTED_FRAMES` | 12 | LK 失败时最多外推帧数 |
| `CORNER_TRACK_MAX_SHIFT` | 18 | 重检角点允许的最大位移 |
| `CORNER_TRACK_AREA_MIN_RATIO` | 0.85 | 重检候选允许的最小面积比 |
| `CORNER_TRACK_AREA_MAX_RATIO` | 1.15 | 重检候选允许的最大面积比 |
| `CORNER_TRACK_REJECTED_FRAMES` | 2 | 异常候选连续出现多少次后才接受 |
| `CORNER_TRACK_SMOOTH_ALPHA` | 0.85 | 正常重检候选的新位置权重 |

#### 原生加速后端

| 参数 | 默认值 | 含义 |
|---|---|---|
| `TARGET_ACCEL_BACKEND` | `python` | 联调默认使用 Python；可改为 `auto/npu` 启用原生加速 |
| `TARGET_ACCEL_LIB_PATH` | `vision/native/libtargetdetect_accel.so` | 原生动态库路径 |
| `TARGET_ACCEL_USE_CPP_LK` | `False` | 角点 LK + RANSAC + 投影是否走 C++ |
| `TARGET_ACCEL_USE_CPP_PERSPECTIVE` | `False` | 透视矩阵、warp 和逆投影是否走 C++ |
| `TARGET_ACCEL_USE_CPP_RING_FIT` | `False` | 圆环拟合是否走 C++ |

联调和量产回退版本默认使用 Python；`TARGET_ACCEL_BACKEND="python"` 会明确禁用原生库加载，避免未完成的 C++ 加速影响联调。确需测试原生加速时再改为 `auto` 并编译 `.so`；加载失败也会自动回退到原 Python 算法。`npu` 为流水线预留模式，当前会在无可用 NPU stage 时自动回退。

### 5.3 `vision/lighting.py`

`update_fill_light()` 计算画面中心区域的亮度：

- 中心区域占画面的中间 1/2 宽度和高度。
- 使用加权灰度近似亮度：
  `0.114×B + 0.587×G + 0.299×R`
- 当补光未开启且亮度低于 `FILL_LIGHT_ON_THRESHOLD` 时开启补光。
- 补光灯一旦开启不会根据亮度自动关闭；需保持低亮场景稳定。
- 补光灯已锁存开启后跳过后续亮度计算，减少每帧像素扫描。
- 程序退出、异常退出或 atexit 流程都会调用补光灯关断。

补光不关闭的设计是为了避免亮度跨阈值时灯光高频开关。

### 5.4 `vision/target_frame.py`

该模块负责靶纸几何定位。

#### 角点排序

`order_points()` 将任意四边形转换为固定顺序：

1. 左上
2. 右上
3. 右下
4. 左下

排序依赖角度和四角到中心的几何关系，而不是依赖固定像素位置。

#### 靶纸框重检

`detect_target_corners()`：

1. 将输入缩放到约 20% 分辨率。
2. 使用黑色 HSV 阈值提取掩膜。
3. 闭运算并查找轮廓。
4. 排除边界轮廓和面积小于画面 12%的轮廓。
5. 使用多个 epsilon 比例近似轮廓为四边形。
6. 拒绝长宽比与 A4 纸差异超过 25% 的候选。
7. HSV 检测失败时尝试灰度 Otsu 检测。
8. 若存在上一参考框，只选择面积、中心和角点位置一致的候选。

长宽比筛选只排除与靶纸比例明显不符的矩形；倾斜拍摄时允许更大的比例变化。候选框仍需通过靶心红环几何验证，才会报告靶心。

#### 角点光流跟踪

`TargetCornerTracker` 使用金字塔 Lucas-Kanade 光流：

1. 靶框四角内提取最多 24 个 `goodFeaturesToTrack` 特征。
2. 将特征从上一帧灰度图跟踪到当前灰度图。
3. 使用 RANSAC 计算特征前后对应关系的单应矩阵。
4. 内点比例低于 75%时拒绝该帧跟踪。
5. 将上一时刻四角通过同态变换投影到当前帧。
6. 对面积比、角点位移、凸性和异常形变做约束。
7. 正常候选使用新旧角点平滑；异常候选需连续一致后才切换。
8. LK 失败时利用角点速度进行有限帧外推。

外推最多 12 帧，避免角点在靶纸移出画面后永久漂移。

当 `TARGET_ACCEL_USE_CPP_LK=True` 且原生库可用时，上述 LK/RANSAC/四角投影链路会由 `vision/native/targetdetect_accel.cpp` 执行，Python 仅保留状态管理与兜底。

#### 透视变换

`make_homography()` 根据相机四角生成校正到标准靶纸图的单应矩阵。

`rectify_image()` 将图像变换到 `210x297`。

`rectified_to_camera()` 对靶心坐标应用逆单应矩阵，投影回原始 `640x480` 摄像头坐标。

当 `TARGET_ACCEL_USE_CPP_PERSPECTIVE=True` 且原生库可用时，`make_homography/rectify_image/rectified_to_camera` 由 C++ 实现执行。

#### 跟踪 ROI

`get_tracking_roi()` 根据靶框和靶心位置生成局部搜索区域：

- 最小宽度为 240 px。
- 最小高度为 240 px。
- ROI 边界限制在原图范围内。
- 边距系数为 `0.25`。

正常情况下应优先在 ROI 内重检靶框，以减少计算量。

### 5.5 `vision/target_center.py`

该模块负责靶心定位。完整流程分为快速分割、快速拟合、兜底拟合和暗红候选四部分。

当 `TARGET_ACCEL_USE_CPP_RING_FIT=True` 且原生库可用时，轮廓筛选、联合圆拟合和 RMS 计算优先走 C++，失败时自动回退到 Python 兜底流程。

#### 红色掩膜提取

常规提取使用红通道差值：

```text
red_excess = red - max(green, blue)
```

红通道差值超过阈值 8 的像素进入红色掩膜，再做一次闭运算，以保持主检测路径轻量并兼容褪色红环。HSV 只在主拟合失败或低置信时用于补充暗红候选。

暗红区域只有在常规拟合失败或 RMS 低置信时才计算：

- 将图像转 HSV。
- 提取低饱和度、低亮度的暗红候选。
- 与常规掩膜合并。
- 再次拟合并比较 RMS。

#### 中心拟合

靶心由不同理论半径（20、40、60、80、100 px）的同心红环支持。快速轮廓拟合允许至少三个不同半径，避免因个别红环模糊或遮挡而转入较慢兜底；同时检查各环圆心一致性：

1. 查找掩膜轮廓。
2. 先用轮廓外接框的平均半径排除明显不匹配的轮廓，再对至少 5 点的剩余轮廓拟合椭圆。
3. 根据椭圆平均半径匹配 20、40、60、80、100 px 理论环。
4. 检查角度覆盖是否达到至少 6 个扇区。
5. 检查半径误差是否符合阈值。
6. 至少三个不同半径存在有效候选且各环圆心偏差不超过 8 px。
7. 候选筛选最多使用 16 个均匀间隔边界点计算覆盖度和评分，并用扇区比较代替逐点反三角函数；最终拟合仍使用完整边界点。所有有效环通过共享圆心、独立半径的联合最小二乘求解。
8. 按每个环的中位径向误差计算鲁棒 RMS，局部激光痕迹不会单独主导置信度；超过阈值标记低置信。

如果轮廓拟合找到的有效半径少于三个，则尝试像素分类兜底；兜底继续检查五个理论半径处的像素和角度覆盖，否则返回“未找到”：

- 将像素按理论环径向分组。
- 对每个环采样最多 80 点。
- 用椭圆拟合每环。
- 使用中位数计算共有圆心。
- 检查角度覆盖、半径误差和画面边界。

只有快速轮廓拟合不足时才进入兜底，避免每帧进行高成本拟合。

#### 下采样

靶心检测先将校正图缩放到约 40% 分辨率。五环半径、角度容差和结果坐标会通过 `scale_x`、`scale_y` 转回标准校正图或原始图坐标。

该设计降低 CPU 负担，但会使低分辨率轮廓边缘略微影响拟合精度。

### 5.6 `vision/kalman.py`

当前实现是二维常速度状态：

```text
状态：[x, y, vx, vy]
```

严格来说，当前可信靶心不再进行卡尔曼平滑：

- `fit_status == "OK"` 时绿点直接使用测量坐标。
- `update()` 只根据连续绿点更新速度状态，不改变位置输出。
- 靶心低置信时显示 `LOW CONF`。
- 靶纸角点有效且曾经锁定靶心但本帧靶心丢失时，才允许 `predict()` 短时外推。
- 靶纸角点不存在时立即显示 `TARGET NOT FOUND`，并复位卡尔曼状态。
- 尚未首次锁定靶心时不进入预测状态。
- 外推丢帧达到一定次数后速度衰减并最终复位。

卡尔曼存在的目的不是替代精准绿点，而是：

- 为短时遮挡提供预测点；
- 保存目标运动速度；
- 仅在靶纸仍在画面内且靶心短时丢失时预测；
- 靶纸移出画面时显示未找到，而不是继续预测；
- 防止异常跳变污染速度。

### 5.7 `control/serial_sender.py`

该模块负责实时串口遥测，不参与图像算法和屏幕绘制。

默认配置：

| 参数 | 默认值 |
|---|---|
| 串口设备 | `/dev/ttyS1` |
| 波特率 | 9600 |
| TX 引脚 | A19 |
| RX 引脚 | A18 |
| 最小发送间隔 | 2 秒；实际间隔会按消息长度和波特率自动增加 |
| 稳定状态心跳 | 5 秒 |

每条串口消息发送原始相机坐标系中的 `x`、`y`：包头为 ASCII 字符 `@`，两个坐标使用逗号分隔，包尾为真实的 CRLF（`\r\n` 两个控制字符，十六进制为 `0D 0A`），不是字面量反斜杠文本。

```text
@320.50,240.25\r\n
```

坐标按原始图像坐标系的 `640×480` 范围解释，仅发送有限的中心点。靶心丢失或中心坐标不是有限值时发送 `@-1.00,-1.00\r\n`。坐标按每坐标保留两位小数进行舍入后比较和发送，因此亚 0.01 px 的微小变化不触发状态发送。任意两次发送至少间隔 2 秒，坐标不变化时每 5 秒发送一次心跳。发送使用 8N1 字节传输速率估算并预留余量。串口写入失败后首次等待 30 秒再试，之后指数退避，最长间隔 300 秒，避免底层错误持续刷屏。

### 5.8 屏幕数据展示

屏幕展示已经简化为上下状态栏：

屏幕默认每 3 帧更新一次；图像检测、跟踪和串口处理仍逐帧运行。该设置用于减少同步显示提交对主循环的阻塞，不改变识别频率或拟合精度。

顶部状态栏：

- 左侧彩色圆点和目标状态。
- 右侧实时靶心坐标。

状态含义：

| 状态 | 颜色 | 含义 |
|---|---|---|
| `TARGET LOCKED` | 绿色 | 本帧绿点高置信 |
| `LOW CONF` | 橙色 | 靶纸角点有效，靶心测量存在但拟合置信度不足 |
| `PREDICTING` | 橙色 | 靶纸角点有效，但本帧无法得到靶心，使用卡尔曼预测点 |
| `TARGET NOT FOUND` | 红色 | 靶纸角点不存在，靶心不可用 |

底部状态栏：

- `FRAME LOCKED`：周期重检或最新四角有效。
- `FRAME TRACKED`：由 LK 光流维持四角。
- `FRAME LOST`：角点完全丢失。
- `RMS`：红环拟合残差，单位 px；绿色为 `OK`，橙色为低置信，`--` 表示无拟合结果。
- `LIGHT ON/OFF`：补光灯状态。
- `FPS`：实际主循环帧率。

### 5.9 `control/gimbal_pid.py`

该模块负责二轴舵机云台的闭环姿态计算，不包含 PWM、舵机脉宽或硬件总线代码。云台控制只使用高置信的 `TARGET LOCKED` 靶心坐标；`LOW CONF`、`PREDICTING` 和 `TARGET NOT FOUND` 不参与云台跟踪。

控制器以相机原始坐标的画面中心为反馈目标：

```text
水平误差 ex = 靶心 x - 画面中心 x
俯仰误差 ey = 靶心 y - 画面中心 y
水平控制量 = Kp×ex + Ki×Σ(ex×dt) + Kd×d(ex)/dt
俯仰控制量 = Kp×ey + Ki×Σ(ey×dt) + Kd×d(ey)/dt
```

每个轴都包含以下保护：

- 静差死区：误差绝对值小于 `GIMBAL_DEADBAND_PX` 时清除积分项并保持姿态。
- 积分限幅：积分项限制在角度范围内，并在饱和或误差反向时停止继续累积，防止积分饱和。
- 微分低通：对误差变化率进行指数滤波，抑制噪声和微分尖峰。
- 输出限幅：每帧水平或俯仰单轴修正角度限制在配置范围内。
- 角度限幅：最终舵机角度限制在水平、俯仰各自的 0°～180°范围内。
- 丢失保护：靶心丢失时立即清除两个轴的积分项、微分历史和目标状态，保持最后姿态。

主循环中，`target_status == "TARGET LOCKED"` 时调用 `gimbal_controller.update(target_point, dt)`；其他状态调用 `mark_target_lost()`。模块按控制周期打印下一步转动方向，例如：

```text
[GIMBAL] status=TRACKING pan_error_px=32.50 pan_action=move_right_1.14deg tilt_error_px=-12.20 tilt_action=move_up_0.43deg pan_attitude_status=moving pan_deg=91.14 tilt_deg=89.57
[GIMBAL] status=TARGET_LOST pan_deg=91.14 tilt_deg=89.57
```

打印中的 `hold, pan lower/upper limit` 和 `hold, tilt lower/upper limit` 表示内部目标角到达软件限位，并不证明实际舵机已到达机械限位。PID 此时不会继续向限位外累积积分项；误差反向后才恢复积分和正常跟踪。

方向含义由 `GIMBAL_PAN_DIRECTION` 和 `GIMBAL_TILT_DIRECTION` 配置：

| 参数 | 含义 |
|---|---|
| `GIMBAL_PAN_DIRECTION=1` | 水平舵机角度增大时云台向右转 |
| `GIMBAL_PAN_DIRECTION=-1` | 水平舵机角度增大时云台向左转 |
| `GIMBAL_TILT_DIRECTION=1` | 俯仰舵机角度增大时云台向下转 |
| `GIMBAL_TILT_DIRECTION=-1` | 俯仰舵机角度增大时云台向上转 |

主要参数位于 `common/config.py`：画面中心、水平和俯仰的 Kp/Ki/Kd、像素死区、角度积分及输出限幅、微分滤波系数、舵机角度范围和打印间隔。

#### IMU 姿态反馈

`gimbal_pid.py` 支持视觉外环与 IMU 姿态内环串级控制：

```text
靶心像素误差
  -> 视觉位置 PID
  -> 水平、俯仰目标舵机角度
  -> IMU 姿态相对误差和角速度反馈
  -> 姿态内环修正角增量
  -> 打印实际舵机运动指令和到位状态
```

外环仍以相机画面中心为目标，像素误差产生目标舵机角度。内环以 IMU 相对姿态和角速度为反馈，计算姿态误差；误差小于角度死区且角速度低于速度死区时标记为 `positioned`。到达软件限位时标记为相应的限位状态，并停止继续外推和积分。

需要注意：

- MaixCAM-Pro 内置 QMI8658 是六轴传感器，不含磁力计。
- 水平 yaw 姿态没有绝对航向参考，只能按首次姿态建立相对反馈，可能因陀螺仪漂移产生长期偏差。
- 俯仰和横滚可使用重力方向的加速度计长期修正；云台需要在安装后确认欧拉角轴对应关系。
- 姿态反馈测量的是相机相对姿态，不等于舵机编码器反馈；需标定轴映射和方向，实际到位情况仍需实机验证。

主循环在每一帧都读取一次板载姿态，无论当前靶心是高置信、低置信、预测还是丢失；这样启动后即使最初没有靶心也能进入 `RUNNING` 状态并建立姿态参考。有效反馈只在 `TARGET LOCKED` 时传入云台 PID，靶心丢失时仍只复位 PID 并保持姿态。
当板载 IMU 初始化或读取失败时，控制器只保留视觉外环，不会停止视觉跟踪，也不会循环报错。
相机姿态模块只在 MaixCAM-Pro 固件环境中直接导入官方的 `maix.ahrs` 和 `maix.ext_dev.imu`，不再尝试兼容旧路径或其他 Maix 设备。

### 5.10 `control/camera_attitude.py`

该模块封装 MaixCAM-Pro 的板载姿态读取：

- 使用 `maix.ext_dev.imu.IMU("default", mode=DUAL)` 初始化 QMI8658。
- 使用陀螺仪校准，首次校准会采集约 10000 个样本并保存到 MaixApp 共享目录。
- 使用 `maix.ahrs.MahonyAHRS(kp, ki)` 融合加速度计和陀螺仪。
- 按官方接口读取弧度每秒的陀螺仪数据供 AHRS 使用，反馈给 PID 的角速度转换为度每秒。
- 输出 `pitch_deg`、`roll_deg`、`yaw_deg`、相对水平/俯仰姿态和角速度。
- 横移轴默认映射到 yaw，俯仰轴默认映射到 pitch。
- 可通过 `GIMBAL_IMU_PAN_SIGN` 和 `GIMBAL_IMU_TILT_SIGN` 修正传感器坐标与云台运动方向。

启动日志按以下状态输出，方便确认姿态 IMU 是否已经启动：

```text
[IMU] status=INITIALIZING step=loading_maix_imu
[IMU] status=READY sensor=QMI8658 ahrs=MahonyAHRS calibration=loaded
[IMU] status=RUNNING pitch_deg=5.00 roll_deg=-2.00 yaw_deg=30.00 pan_relative_deg=0.00 tilt_relative_deg=0.00 temperature_c=25.00
```

- 首次读取成功时输出一次完整姿态读数，后续帧不重复。
- 首次读取失败时只输出一次 `READ_ERROR`，同时继续使用视觉 PID。
- 初始化失败时输出 `UNAVAILABLE` 和回退原因，不影响视觉跟踪主流程。

推荐的启动步骤：

1. 将云台置于中间机械姿态并保持静止。
2. 首次运行时完成陀螺仪校准。
3. 靠近画面中心放置靶心，观察水平、俯仰轴的实际到位方向。
4. 根据相机安装方向调整 `GIMBAL_IMU_PAN_AXIS`、`GIMBAL_IMU_TILT_AXIS` 和符号。
5. 舵机出现方向反转时只修改方向配置，不修改视觉靶心坐标。

## 6. 坐标系统

项目有三套坐标：

| 坐标系 | 范围 | 用途 |
|---|---|---|
| 原始相机坐标 | 640x480 | 屏幕显示、云台闭环控制、跟踪 ROI |
| 校正靶纸坐标 | 210x297 | 红环拟合和靶心计算 |
| 角点跟踪小图坐标 | 160x120 | LK 特征跟踪 |

坐标转换包括：

- 全图到角点跟踪小图：乘 `CORNER_TRACK_SCALE`。
- 全图到靶心检测小图：乘 `CENTER_DETECT_SCALE`。
- 校正图到相机图：逆透视矩阵。
- 局部 ROI 到全图：给角点加 ROI 左上偏移。

修改分辨率参数时必须同步检查所有坐标还原逻辑。

## 7. 状态机

### 靶纸框状态

```text
初始
  -> 无跟踪状态
  -> 周期重检靶纸
  -> 靶框重检成功：初始化光流
  -> LK 正常：TRACKED
  -> LK 失败：PREDICT
  -> 超过 12 帧或状态失效：LOST
  -> LOST 后重新全图重检
```

实际状态按角点有效性确定：

- `FRAME LOCKED`：本帧得到重检或有效角点。
- `FRAME TRACKED`：角点来自 LK。
- `FRAME LOST`：角点不可用。

异常候选会先进入待验证状态，连续一致两次后才更新角点。

### 靶心状态

```text
高置信绿环拟合
  -> 直接输出绿点

拟合存在但 RMS 过高
  -> 尝试暗红掩膜
  -> 成功且更优则输出
  -> 否则标记 LOW CONF

靶纸角点有效，但曾锁定且靶心短时丢失
  -> 卡尔曼预测并标记 PREDICTING

靶纸角点不存在或项目尚未首次锁定靶心
  -> TARGET NOT FOUND
```

云台状态与靶心状态并行：

- `TARGET LOCKED`：执行水平、俯仰 PID 并打印目标舵机角度变化。
- `LOW CONF`、`PREDICTING` 或 `TARGET NOT FOUND`：复位 PID 积分项并保持当前舵机姿态。

## 8. 耗时统计

主循环每秒打印一次结构化耗时快照：

```text
[TIMING] status=SNAPSHOT frame=649 corner_mode=TRACK fit_status=OK fit_rms=1.90 display_updated=false corner_features=20 camera_io_ms=11.49 target_frame_ms=74.99 target_center_ms=80.48 imu_ms=1.00 gimbal_ms=0.20 annotations_ms=0.00 status_overlay_ms=0.00 display_ms=0.00 total_ms=170.00 fps=5.88
```

应用状态日志统一使用以下格式：

```text
[模块名] status=状态 key=value key=value
```

其中 `APP` 表示应用启动，`IMU` 表示姿态传感器状态，`GIMBAL` 表示云台跟踪或丢失状态，`UART` 表示串口状态，`TIMING` 表示每秒耗时快照。字符串中的空格会转换为下划线，浮点值统一保留两位小数，空值显示为 `--`。

字段含义：

| 字段 | 含义 |
|---|---|
| `camera_io_ms` | 相机读取及图像格式转换 |
| `target_frame_ms` | 靶纸重检、LK、透视变换和 ROI |
| `target_center_ms` | 红环提取和靶心拟合 |
| `imu_ms` | IMU 读取、姿态解算和首次状态输出 |
| `gimbal_ms` | PID 更新或目标丢失处理，包含可能的云台日志输出 |
| `annotations_ms` | 靶框和靶心标记绘制 |
| `status_overlay_ms` | 实时状态栏绘制 |
| `display_ms` | `disp.show()` 显示提交耗时 |
| `display_updated` | 本帧是否提交了屏幕刷新 |
| `total_ms` | 从循环开始到输出本次快照之前的耗时 |
| `fps` | 单帧倒数 |

快照每秒输出一次，反映当时的一帧而非平均值，且不包含快照自身的打印耗时。非显示帧不绘制标记和状态栏，视觉检测、IMU 和 PID 仍逐帧运行。

常见诊断：

- `corner_mode=LOST` 且持续很久：角点重检或黑框颜色阈值存在问题。
- `corner_features` 持续低于 12：角点纹理不足或 ROI 过小。
- `corner_mode=PREDICT` 频繁出现：LK 质量差或相机晃动过快。
- `target_center_ms` 高：检查红环分割与拟合开销。
- `imu_ms` 或 `gimbal_ms` 高：分别检查姿态读取、解算或云台日志输出。
- `display_ms` 高：显示提交/刷新可能阻塞主循环。
- 周期重检帧高于普通帧：应检查重检间隔是否过短。

## 9. 性能优化原则

项目中的性能优化遵循以下优先级：

1. 减少全分辨率像素处理。
2. 不在每帧重复静态边框检测。
3. 优先使用轮廓和几何约束，不进行不必要的像素拟合。
4. 暗红 HSV 和像素兜底只在必要时执行。
5. `image2cv(..., copy=False)` 和 `cv2image(..., copy=False)` 减少内存拷贝。
6. 不在可信靶心上进行卡尔曼平滑。
7. 不在每帧刷新角点特征。
8. 保持高频分支只做必要的单目标处理。

这些优化不应牺牲几何约束。若出现角点漂移，优先修正角点质量、面积约束和候选一致性，而不是再次降低跟踪分辨率。

## 10. 关键算法约束

### 靶纸框约束

- 候选面积不能低于画面 12%。
- 候选不能位于画面边界。
- 候选必须近似为凸四边形。
- 重检候选面积比限制在约 0.85 到 1.15。
- 一致异常候选需要连续出现两次后才更新。

### 角点跟踪约束

- 角点特征数量至少 12。
- RANSAC 内点比例至少 75%。
- 四角面积不能变化超过设定范围。
- 每角逐帧位移不能超过设定阈值。
- LK 失败最多外推 12 帧。

### 靶心约束

- 优先要求至少三个可靠红环。
- 每环角度覆盖至少 6 个扇区。
- 半径误差不能超过最大允许误差。
- 靶心必须落在校正图范围内。
- RMS 超过 5 px 标记低置信。

这些约束可能将部分图像标记为低置信，但比错误靶心传给控制逻辑更安全。

## 11. 部署与运行

### 应用打包

`app.yaml` 中列出的文件必须包含在 MaixApp 中：

- `main.py`
- `app.py`
- `common/config.py`
- `vision/lighting.py`
- `vision/target_frame.py`
- `vision/target_center.py`
- `vision/native_backend.py`
- `vision/kalman.py`
- `control/serial_sender.py`
- `control/gimbal_pid.py`
- `control/camera_attitude.py`
- `common/console_output.py`
- `common/__init__.py`
- `vision/__init__.py`
- `control/__init__.py`
- `vision/native/targetdetect_accel.cpp`
- `vision/native/build_with_maixcdk.sh`
- `vision/native/maixcdk_project/app.yaml`
- `vision/native/maixcdk_project/main/CMakeLists.txt`
- `vision/native/maixcdk_project/main/src/main.cpp`
- `vision/native/maixcdk_project/components/targetdetect_accel/CMakeLists.txt`
- `vision/native/maixcdk_project/components/targetdetect_accel/src/targetdetect_accel.cpp`

应用入口由 `main.py` 启动。

### 原生加速库编译

项目新增 `vision/native/targetdetect_accel.cpp`，用于将以下热点链路迁移到 C++：

- LK 光流 + RANSAC 单应矩阵 + 四角投影
- 透视矩阵、`warpPerspective`、逆透视回投影
- 红环轮廓联合圆心拟合（含 NEON 优化分支）

在设备或交叉编译环境中可使用 MaixCDK 构建脚本：

```bash
cd vision/native
./build_with_maixcdk.sh
```

首次运行时，脚本会进入 `maixcdk menuconfig`；在配置界面选择 **MaixCAM** 平台并保存退出。

完整重新配置或源码新增后使用：

```bash
./build_with_maixcdk.sh build
```

只修改已有源文件时可加快编译：

```bash
./build_with_maixcdk.sh build2
```

不要使用 `maixcdk -p maixcam build`；在当前 MaixCDK CLI 中，平台选择通过 `menuconfig` 保存，实际构建使用 `maixcdk build` 或 `maixcdk build2`。

产物默认输出到 `vision/native/libtargetdetect_accel.so`，依赖库同步到 `vision/native/dl_lib/`。运行时可通过环境变量 `TARGETDETECT_NATIVE_LIB` 和 `TARGETDETECT_NATIVE_DL_LIB_DIR` 覆盖库与依赖目录。

### GPIO 与补光灯

程序使用：

```text
引脚：B3
GPIO：GPIOB3
方向：OUT
```

需要确认：

- 引脚未被其他应用占用。
- B3 未配置为冲突功能。
- 补光灯供电和电平与板载 GPIO 相容。
- 程序退出时会执行关灯操作。

### 摄像头

摄像头固定配置为：

- 分辨率：640×480
- 格式：BGR888
- 帧率：30 FPS

修改分辨率时需要同步检查：

- 靶纸面积阈值。
- 跟踪 ROI 尺寸。
- 校正图比例。
- 屏幕状态栏坐标。
- 性能统计含义。

## 12. 调参指南

调参应一次只改变一个参数组，并观察对应日志。

### 靶纸检测失败

检查：

1. 靶纸边框是否为暗黑色。
2. `FRAME_HSV_LOW/HIGH` 是否适合实际光照。
3. 摄像头曝光是否过强或过暗。
4. 是否只有局部纸边进入画面。
5. 是否被补光改变了色相。

优先调整光照和黑框阈值，不优先放宽面积约束。

### 靶心低置信

检查：

1. 红环是否完整显示。
2. 红环是否被遮挡。
3. 红环是否偏离理论半径。
4. 透视校正是否准确。
5. 四角是否贴合靶纸。

优先修正四角和透视几何，再调整 `FIT_RMS_WARNING_THRESHOLD`。

### 边框乱动

检查：

1. `CORNER_TRACK_MIN_FEATURES` 是否过低。
2. 摄像头晃动是否超过 LK 可跟踪范围。
3. 靶纸背景纹理是否与框内纹理混淆。
4. 重检候选面积和位移约束是否过宽。
5. 是否存在多个矩形轮廓。

优先提高候选一致性，不要直接放松凸性和面积约束。

### 帧率下降

检查：

1. `target_frame`、`target_center` 和 `app_and_render` 中哪项最高。
2. 是否频繁进入周期重检。
3. 是否频繁进入像素兜底拟合。
4. 屏幕绘制是否包含不必要的高成本操作。
5. 分辨率和补光是否影响实际图像复杂度。

## 13. 故障排查

### 没有检测到靶框

可能原因：

- 光照和 HSV 阈值不匹配。
- 靶纸面积占画面不足。
- 靶纸部分位于画面边缘。
- 靶纸被遮挡或不完整。
- 四角轮廓没有成功近似为四边形。

处置：

- 棐查静止画面中的黑框。
- 先调整静止画面的光照、阈值和重检比例，测试重检结果。
- 记录黑色掩膜和候选轮廓。

### 靶心不准

可能原因：

- 四角透视几何偏差。
- 红色掩膜混入非靶心红色区域。
- 暗红兜底错误接受。
- 校正比例或理论环半径与实际靶纸不一致。

处置：

- 确认角点框贴合实际靶纸。
- 确认校正图中的五环半径。
- 对比不同 RMS 和低置信状态。

### 黄点不准

当前黄点只用于低置信或丢失预测，不代表精准绿点。

若预测黄点偏差明显：

- 检查绿点是否持续提供正确测量。
- 检查丢失帧数和目标速度方向。
- 不应在绿点精确时继续平滑绿点。
- 若真实坐标能稳定获得，优先停止卡尔曼平滑。

### 补光灯一直变化

已采用锁存逻辑：

- 只在关闭且亮度低时开启。
- 开启后不再自动关闭。
- 如果仍发现异常开关，检查 GPIO 反馈、供电或外部硬件控制逻辑。

## 14. 建议的后续维护方向

以下改进有助于进一步提升稳定性，但不应影响现有精确绿点逻辑：

1. 将角点候选稳定性参数纳入运行时配置。
2. 增加每帧 `corner_mode` 与候选几何偏差日志。
3. 增加暗红兜底是否接受的状态。
4. 增加环拟合 RMS、红环像素数和角度覆盖日志。
5. 使用外部文件或应用设置界面动态调整阈值。
6. 在实际设备上增加长时间摇动相机和遮挡靶纸的回归测试。
7. 分离视觉算法模块与 Maix 硬件接口，便于在 PC 上单元测试。
8. 对灰度和透视操作增加形状、类型、边界异常保护。
9. 对摄像头读取失败和 GPIO 初始化失败增加明确错误提示。

## 15. 新人快速上手顺序

建议按以下顺序阅读和理解代码：

1. 阅读 `common/config.py`，建立分辨率、半径、阈值和跟踪参数的整体印象。
2. 阅读 `main.py` 和 `app.py`，理解主循环执行顺序。
3. 阅读 `vision/target_frame.py`，理解靶纸框和四角跟踪。
4. 阅读 `vision/target_center.py`，理解靶心提取和置信度。
5. 阅读 `vision/kalman.py`，理解丢失帧预测。
6. 阅读 `vision/lighting.py`，理解补光状态。
7. 阅读 `control/gimbal_pid.py`，理解水平、俯仰误差和 PID 输出。
8. 修改前先运行静止画面、轻微晃动、快速晃动和目标遮挡四类场景。
9. 观察每秒一次的 `timing` 日志，定位速度瓶颈。

完成以上步骤后，可以只修改参数或局部模块，而不会破坏整体坐标和状态转换关系。
