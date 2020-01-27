# 使用 CUDA C/C++ 统一内存和 nvprof 管理加速应用程序内存 #

## Lesson 1 ##  

如要确保优化加速代码库的尝试真正取得成功，唯一方法便是分析应用程序以获取有关其性能的定量信息。nvprof 是指 NVIDIA 命令行分析器。该分析器附带于CUDA工具包中，能为加速应用程序分析提供强大功能。

nvprof 使用起来十分简单，最基本用法是向其传递使用 nvcc 编译的**可执行文件**(*也即 nvprof的使用是在cu文件编译通过后对其可执行文件进行分析*)的路径。随后 nvprof 会继续执行应用程序，并在此之后打印应用程序 GPU 活动的摘要输出、CUDA API 调用以及统一内存活动的相关信息。  

nvprof命令行安装与使用请参考InstallCuda.md文档中的详细说明  

### 作业1 ###  

[01-vector-add.cu](./01-vector-add.cu)（<------您可点击打开此文件链接和本实验中的任何源文件链接并进行编辑）是一个简单易用的加速向量加法程序。使用下方两个代码执行单元（按住 CTRL 并点击即可）。第一个代码执行单元将编译（及运行）向量加法程序。第二个代码执行单元将运用 nvprof 分析刚编译好的可执行文件。

应用程序分析完毕后，请使用分析输出中显示的信息回答下列问题：

- 1 此应用程序中唯一调用的 CUDA 核函数的名称是什么？
- 2 此应用程序中唯一调用的 CUDA 核函数的名称是什么？
- 3 此核函数的运行时间为？在某处记录此时间：您将优化此应用程序，还会希望得知所能取得的最大优化速度。  
- 4 首先列出您将用于更新执行配置的 3 至 5 种不同方法，确保涵盖一系列不同的网格和线程块大小组合。
- 5 使用所列的其中一种方法编辑 01-vector-add.cu 程序。
- 6 使用下方的两个代码执行单元编译和分析更新后的代码。
- 7 记录核函数执行的运行时，应与分析输出中给出的相同。
- 8 对以上列出的每个可能实现的优化重复执行编辑/分析/记录循环  



## Lesson 2 ##  

<div align="center"><iframe src="https://view.officeapps.live.com/op/view.aspx?src=https://developer.download.nvidia.com/training/courses/C-AC-01-V1/AC_UM_NVPROF-zh/NVPROF_UM_1-zh.pptx" frameborder="0" width="900" height="550" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe></div>

**关于设备上的SM**  
每个设备的SM数不同 需要进行查询  
由于 GPU 上的 SM 数量会因所用的特定 GPU 而异，因此为支持可移植性，您不得将 SM 数量硬编码到代码库中。相反，应该以编程方式获取此信息。

以下所示为在 CUDA C/C++ 中获取 C 结构的方法，该结构包含当前处于活动状态的 GPU 设备的多个属性，其中包括设备的 SM 数量：

```c
int deviceId;
cudaGetDevice(&deviceId);                  // `deviceId` now points to the id of the currently active GPU.

cudaDeviceProp props;
cudaGetDeviceProperties(&props, deviceId); // `props` now has many useful properties about
                                           // the active GPU device.  
```  

### 作业2 ###  

Exercise: Query the Device
目前，[01-get-device-properties.cu](./01-get-device-properties.cu) 包含众多未分配的变量，并将打印一些无用信息，这些信息用于描述当前处于活动状态的 GPU 设备的详细信息。

扩建该代码以打印源代码中指示的所需设备属性的实际值。为获取操作支持并查看相关介绍，请参阅[CUDA 运行时文档](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html)以帮助识别设备属性结构中的相关属性。

### 作业3 ###  

通过查询设备的 SM 数量重构您一直在 [01-vector-add.cu](./01-vector-add.cu) 内执行的 addVectorsInto 核函数，以便其启动时的网格包含数倍于设备上 SM 数量的线程块数。


**关于SM对并行编程**  

- 1 warp是CUDA最小的执行单元  
- 2 CUDA的设备在实际执行过程中，会以block为单位分配给SM进行运算；而block中的thread又会以warp（线程束）为单位，对thread进行分组计算。  
- 3 请思考  如果一个设备中有2个SM，启动kernel1<<<64,1>>> kernel2<<<2,32>>> kernel3<<<1,64>>> 执行时间的大小顺序为？ 

## Lesson 3 ##  

您一直使用 cudaMallocManaged 分配旨在供主机或设备代码使用的内存，并且现在仍在享受这种方法的便利之处，即在实现自动内存迁移且简化编程的同时，而无需深入了解 cudaMallocManaged 所分配统一内存 (UM) 实际工作原理的详细信息。nvprof 提供有关加速应用程序中 UM 管理的详细信息，并在利用这些信息的同时结合对 UM 工作原理的更深入理解，进而为优化加速应用程序创造更多机会。  

<div align="center"><iframe src="https://view.officeapps.live.com/op/view.aspx?src=https://developer.download.nvidia.com/training/courses/C-AC-01-V1/AC_UM_NVPROF-zh/NVPROF_UM_2-zh.pptx" frameborder="0" width="900" height="550" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe></div>  


**Unified Memory Migration**  

分配 UM 时，内存尚未驻留在主机或设备上。主机或设备尝试访问内存时会发生 页错误，此时主机或设备会批量迁移所需的数据。同理，当 CPU 或加速系统中的任何 GPU 尝试访问尚未驻留在其上的内存时，会发生页错误并触发迁移。

能够执行页错误并按需迁移内存对于在加速应用程序中简化开发流程大有助益。此外，在处理展示稀疏访问模式的数据时（例如，在应用程序实际运行之前无法得知需要处理的数据时），以及在具有多个 GPU 的加速系统中，数据可能由多个 GPU 设备访问时，按需迁移内存将会带来显著优势。

有些情况下（例如，在运行时之前需要得知数据，以及需要大量连续的内存块时），我们还能有效规避页错误和按需数据迁移所产生的开销。

本实验的后续内容将侧重于对按需迁nvprof 会提供描述所分析应用程序 UM 行为的输出。在本练习中，您将对一款简易应用程序作出一些修改，并会在每次更改后利用 nvprof 的统一内存输出部分，探讨 UM 数据迁移的行为方式。


### 作业4 ### 

[01-page-faults.cu](./01-page-faults.cu) 包含 hostFunction 和 gpuKernel 函数，我们可以通过这两个函数并使用数字 1 初始化 2<<24 单元向量的元素。主机函数和 GPU 核函数目前均未使用。

对于以下 4 个问题中的每一问题，请根据您对 UM 行为的理解，首先假设应会发生何种页错误，然后使用代码库中所提供 2 个函数中的其中一个或同时使用这两个函数编辑 01-page-faults.cu 以创建场景，以便您测试假设。

如要测试您的假设，请使用下方的代码执行单元编译及分析您的代码。请务必针对您正进行的 4 个实验，记录您的假设以及从 nvprof 输出中获取的结果，尤其是 CPU 和 GPU 页错误。如您遇到问题，可点击以下链接获取 4 个实验中每个实验的参考解决方案。

- 1 当统一内存仅由 CPU 访问时会出现什么情况？（解决方案）
- 2 当统一内存仅由 GPU 访问时会出现什么情况？（解决方案）
- 3 当统一内存先由 CPU 访问后由 GPU 访问时会出现什么情况？（解决方案）
- 4 当统一内存先由 GPU 访问后由 CPU 访问时会出现什么情况？（解决方案）  

### 作业5 ###  

返回您一直在本实验中执行的[01-vector-add.cu](./01-vector-add.cu)程序，查看程序在当前状态下的代码库，并假设您期望发生何种页错误。查看上一个重构的分析输出（可通过向上滚动查找输出或通过执行下方的代码执行单元进行查看），并观察分析器输出的统一内存部分。您可否根据代码库的内容对页错误描述作一解释？  

### 作业6 ###  

当 nvprof 给出核函数所需的执行时间时，则在此函数执行期间发生的主机到设备页错误和数据迁移都会包含在所显示的执行时间中。

带着这样的想法来将 01-vector-add.cu 程序中的 initWith 主机函数重构为 CUDA 核函数，以便在 GPU 上并行初始化所分配的向量。成功编译及运行重构的应用程序后，但在对其进行分析之前，请假设如下内容：

- 1 您期望重构会对 UM 页错误行为产生何种影响？
- 2 您期望重构会对所报告的 addVectorsInto 运行时产生何种影响？  

### 异步内存存取 ###  

**目的**:为了排除页错误以提升运行效率  
在主机到设备和设备到主机的内存传输过程中，我们使用一种技术来减少页错误和按需内存迁移成本，此强大技术称为异步内存预取。通过此技术，程序员可以在应用程序代码使用统一内存 (UM) 之前，在后台将其异步迁移至系统中的任何 CPU 或 GPU 设备。*此举可以减少页错误和按需数据迁移所带来的成本，并进而提高 GPU 核函数和 CPU 函数的性能*。

此外，预取往往会以*更大的数据块*来迁移数据，因此*其迁移次数要低于按需迁移*。此技术非常适用于以下情况：在运行时之前已知数据访问需求且数据访问并未采用稀疏模式。

CUDA 可通过*cudaMemPrefetchAsync*函数，轻松将托管内存异步预取到 GPU 设备或 CPU。以下所示为如何使用该函数将数据预取到当前处于活动状态的 GPU 设备，然后再预取到 CPU：  

```c
int deviceId;
cudaGetDevice(&deviceId);                                         // The ID of the currently active GPU device.

cudaMemPrefetchAsync(pointerToSomeUMData, size, deviceId);        // Prefetch to GPU device.
cudaMemPrefetchAsync(pointerToSomeUMData, size, cudaCpuDeviceId); // Prefetch to host. `cudaCpuDeviceId` is a
                                                                  // built-in CUDA variable.
```

### 作业7 ###  

此时，实验中的[01-vector-add.cu](./01-vector-add.cu)程序不仅应启动 CUDA 核函数以将 2 个向量添加到第三个解向量（所有向量均通过 cudaMallocManaged 函数进行分配），还应在 CUDA 核函数中并行初始化其中的每个向量。  

在该应用程序中使用 cudaMemPrefetchAsync 函数开展 3 个实验，以探究其会对页错误和内存迁移产生何种影响。

- 1 当您将其中一个初始化向量预取到主机时会出现什么情况？
- 2 当您将其中两个初始化向量预取到主机时会出现什么情况？
- 3 当您将三个初始化向量全部预取到主机时会出现什么情况？
在进行每个实验之前，请先假设 UM 的行为表现（尤其就页错误而言），以及其对所报告的初始化核函数运行时会产生何种影响，然后运行 nvprof 进行验证。  

### 作业8 ###  

请为该函数添加额外的内存预取回 CPU，以验证 addVectorInto 核函数的正确性。然后再次假设 UM 所受影响，并在 nvprof 中进行分析确认。  

### 作业9 ###  

[01-saxpy.cu](./01-saxpy.cu)为您提供一个基本的[SAXPY](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_1)加速应用程序。该程序目前包含一些您需要找到并修复的错误，在此之后您才能使用 nvprof 成功对其进行编译、运行和分析。

在修复完错误并对应用程序进行分析后，您需记录 saxpy 核函数的运行时，然后采用迭代方式优化应用程序，并在每次迭代后使用 nvprof 进行分析验证，以便了解代码更改对核函数性能和 UM 行为产生的影响。

您的最终目标是在不修改 N 的情况下分析准确的 saxpy 核函数，以便在 50us 内运行。如您遇到问题，请参阅 解决方案，您亦可随时对其进行编译和分析。

