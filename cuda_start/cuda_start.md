# 使用Cuda C/C++ 加速应用程序 #  

*Author: JieLi*  
*Contents from NVIDIA Courses*  

## 预备基础 ##  
- 1 在 C 中声明变量、编写循环并使用 if/else 语句。
- 2 在 C 中定义和调用函数。
- 3 在 C 中分配数组。  

## Lesson 1 ##  

<div align="center"><iframe src="https://view.officeapps.live.com/op/view.aspx?src=https://www.nvidia.com/content/dam/en-zz/zh_cn/Solutions/deep-learning/downloads/1-AcceleratingApplicationswithCUDAC-C/AC_CUDA_C_1.pptx" frameborder="0" width="900" height="550" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe></div>    

*sample 1 for your code*  

``` cpp
void CPUFunction()
{
  printf("This function is defined to run on the CPU.\n");
}

__global__ void GPUFunction()
{
  printf("This function is defined to run on the GPU.\n");
}

int main()
{
  CPUFunction();

  GPUFunction<<<1, 1>>>();
  cudaDeviceSynchronize();
}
```  
需要明确的概念  
**cuda编程的异步性** 指的是利用GPU的多个cuda core并行执行指令 该过程与CPU执行是异步。因此会出现一种情况——CPU或GPU先执行完其上的程序指令(通常是CPU)。因此我们需要同步二者，使得程序能返回预期的结果。请特别注意上述代码中的*cudaDeviceSynchronize();*。并尝试删去该语句前后程序运行结果的区别。   

**cuda编程的线程并行** cuda编程通过指定并发的线程数线程块对性能进行部分优化。如*GPUFunction<<<1, 1>>>();* 即指定启动1个线程块1个线程。具体的参数与机器的SM数以及计算需求等都有关。  

### 作业1 ###  
重构  [hello-gpu.cu](./hello-gpu.cu) 要求如下    
- 1 重构源文件中的 helloGPU 函数，以便该函数实际上在 GPU 上运行，并打印指示执行此操作的消息。  
- 2 成功重构 [hello-gpu.cu](./hello-gpu.cu) 后，请进行以下修改  
    - 1 移除对 cudaDeviceSynchronize 的调用。在编译和运行代码之前，猜猜会发生什么情况，可以回顾一下核函数采取的是异步启动，且 cudaDeviceSynchronize 会使主机执行暂作等待，直至核函数执行完成后才会继续。完成后，请替换对 cudaDeviceSynchronize 的调用。
    - 2 重构 hello-gpu，以便 Hello from the GPU 在 Hello from the CPU 之前打印。
    - 3 重构 hello-gpu，以便 Hello from the GPU 打印两次，一次是在 Hello from the CPU 之前，另一次是在 Hello from the CPU 之后。  

*编译指令*  

请采用如下命令行编译指令  
```shell
nvcc -arch=sm_xx -o hello-gpu hello-gpu.cu -run
```  
具体介绍可参考环境部署文档  
或在VS201X中分别点击编译 生成 开始执行(不调试)  

## Lesson 2 ##  

<div align="center"><iframe src="https://view.officeapps.live.com/op/view.aspx?src=https://developer.download.nvidia.com/training/courses/C-AC-01-V1/AC_CUDA_C-zh/AC_CUDA_C_2-zh.pptx" frameborder="0" width="900" height="550" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe></div>  


程序员可通过执行配置指定有关如何启动核函数以在多个 GPU 线程中并行运行的详细信息。更准确地说，程序员可通过执行配置指定线程组（称为线程块或简称为块）数量以及其希望每个线程块所包含的线程数量。执行配置的语法如下：

<<< NUMBER_OF_BLOCKS, NUMBER_OF_THREADS_PER_BLOCK>>>

启动核函数时，核函数代码由每个已配置的线程块中的每个线程执行。

因此，如果假设已定义一个名为 someKernel 的核函数，则下列情况为真：

- 1 someKernel<<<1, 1>>() 配置为在具有单线程的单个线程块中运行后，将只运行一次。
- 2 someKernel<<<1, 10>>() 配置为在具有 10 线程的单个线程块中运行后，将运行 10 次。
- 3 someKernel<<<10, 1>>() 配置为在 10 个线程块（每个均具有单线程）中运行后，将运行 10 次。
someKernel<<<10, 10>>() 配置为在 10 个线程块（每个均具有 10 线程）中运行后，将运行 100 次。  

### 作业2 ###  

源代码文件 [01-basic-parallel.cu](./01-basic-parallel.cu) 
- 1 重构 firstParallel 函数以便在 GPU 上作为 CUDA 核函数启动。在使用下方 nvcc 命令编译和运行 01-basic-parallel.cu 后，您应仍能看到函数的输出。
- 2 重构 firstParallel 核函数以便在 5 个线程中并行执行，且均在同一个线程块中执行。在编译和运行代码后，您应能看到输出消息已打印 5 次。
- 3 再次重构 firstParallel 核函数，并使其在 5 个线程块内并行执行（每个线程块均包含 5 个线程）。编译和运行之后，您应能看到输出消息现已打印 25 次。  

## Lesson 3 ##  

<div align="center"><iframe src="https://view.officeapps.live.com/op/view.aspx?src=https://developer.download.nvidia.com/training/courses/C-AC-01-V1/AC_CUDA_C-zh/AC_CUDA_C_3-zh.pptx" frameborder="0" width="900" height="550" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe></div>  

**线程组织结构**  

线程(thread)-->线程块(block)-->网格(grid)  

CUDA 核函数可以访问能够识别如下两种索引的特殊变量：正在执行核函数的线程（位于线程块内）索引和线程所在的线程块（位于网格内）索引。这两种变量分别为 threadIdx.x 和 blockIdx.x。  

### 作业3 ###

[01-thread-and-block-idx.cu](./01-thread-and-block-idx.cu) 文件包含一个正在打印失败消息的执行中的核函数。打开文件以了解如何更新执行配置，以便打印成功消息。重构后，请使用下方代码执行单元编译并运行代码以确认您的工作。

### 作业4 ###  

考虑CPU循环如何以并行方式实现？事实上，对于非马尔科夫链形式的循环可以由并行实现  

目前，[01-single-block-loop.cu](./01-single-block-loop.cu) 内的 loop 函数运行着一个“for 循环”并将连续打印 0 至 9 之间的所有数字。将 loop 函数重构为 CUDA 核函数，使其在启动后并行执行 N 次迭代。重构成功后，应仍能打印 0 至 9 之间的所有数字。  

## Lesson 4 ##  

<div align="center"><iframe src="https://view.officeapps.live.com/op/view.aspx?src=https://developer.download.nvidia.com/training/courses/C-AC-01-V1/AC_CUDA_C-zh/AC_CUDA_C_4-zh.pptx" frameborder="0" width="900" height="550" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe></div>  

线程块包含的线程具有数量限制：确切地说是 1024 个。  
CUDA 核函数可以访问给出块中线程数的特殊变量：blockDim.x。通过将此变量与 blockIdx.x 和 threadIdx.x 变量结合使用，并借助惯用表达式 threadIdx.x + blockIdx.x * blockDim.x 在包含多个线程的多个线程块之间组织并行执行，并行性将得以提升。以下是详细示例。

执行配置 <<<10, 10>>> 将启动共计拥有 100 个线程的网格，这些线程均包含在由 10 个线程组成的 10 个线程块中。因此，我们希望每个线程（0 至 99 之间）都能计算该线程的某个唯一索引。  

### 作业5 ###  

[02-multi-block-loop.cu](./02-multi-block-loop.cu) 内的 loop 函数运行着一个“for 循环”并将连续打印 0 至 9 之间的所有数字。将 loop 函数重构为 CUDA 核函数，使其在启动后并行执行 N 次迭代。重构成功后，应仍能打印 0 至 9 之间的所有数字。对于本练习，作为附加限制，请使用启动至少 2 个线程块的执行配置。 

**使用cudaMallocManaged进行内存分配**  

```c  
// CPU-only

int N = 2<<20;
size_t size = N * sizeof(int);

int *a;
a = (int *)malloc(size);

// Use `a` in CPU-only program.

free(a);
// Accelerated

int N = 2<<20;
size_t size = N * sizeof(int);

int *a;
// Note the address of `a` is passed as first argument.
cudaMallocManaged(&a, size);

// Use `a` on the CPU and/or on any GPU in the accelerated system.

cudaFree(a);
```  
### 作业6 ###

[01-double-elements.cu](./01-double-elements.cu) 程序会分配一个数组、在主机上使用整数值对其进行初始化并尝试在 GPU 上对其中的每个值并行加倍，然后在主机上确认加倍操作是否成功。目前，程序将无法执行：因其正尝试在主机和设备上与指针 a 指向的数组进行交互，但仅分配可在主机上访问的数组（使用 malloc）。重构应用程序以满足以下条件

## Lesson 5 ##  

<div align="center"><iframe src="https://view.officeapps.live.com/op/view.aspx?src=https://developer.download.nvidia.com/training/courses/C-AC-01-V1/AC_CUDA_C-zh/AC_CUDA_C_5-zh.pptx" frameborder="0" width="900" height="550" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe></div>  

**创建并行循环所需确切线程数量的执行配置**  

常见示例与希望选择的最佳线程块大小有关。例如，鉴于 GPU 的硬件特性，所含线程的数量为 32 的倍数的线程块是为理想的选择，因其具备性能上的优势。假设我们要启动一些线程块且每个线程块中均包含 256 个线程（32 的倍数），并需运行 1000 个并行任务（此处使用极小的数量以便于说明），则任何数量的线程块均无法在网格中精确生成 1000 个总线程，因为没有任何整数值在乘以 32 后可以恰好等于 1000。  

以下是编写执行配置的惯用方法示例，适用于 N 和线程块中的线程数已知，但无法保证网格中的线程数和 N 之间完全匹配的情况。如此一来，便可确保网格中至少始终拥有 N 所需的线程数，且超出的线程数至多仅可相当于 1 个线程块的线程数量：  

``` c
// Assume `N` is known
int N = 100000;

// Assume we have a desire to set `threads_per_block` exactly to `256`
size_t threads_per_block = 256;

// Ensure there are at least `N` threads in the grid, but only 1 block's worth extra
size_t number_of_blocks = (N + threads_per_block - 1) / threads_per_block;

some_kernel<<<number_of_blocks, threads_per_block>>>(N);
```  

由于上述执行配置致使网格中的线程数超过 N，因此需要注意 some_kernel 定义中的内容，以确保 some_kernel 在由其中一个*多余*线程执行时不会尝试访问超出范围的数据元素  

### 作业7 ###
[02-mismatched-config-loop.cu](./02-mismatched-config-loop.cu) 中的程序使用 cudaMallocManaged 为包含 1000 个元素的整数数组分配内存，然后试图使用 CUDA 核函数以并行方式初始化数组中的所有值。此程序假设 N 和 threads_per_block 的数量均为已知。您的任务是完成以下两个目标  

- 1 为 number_of_blocks 分配一个值，以确保线程数至少与指针 a 中可供访问的元素数同样多。
- 2 更新 initializeElementsTo 核函数以确保不会尝试访问超出范围的数据元素。  

## Lesson 6 ##  

<div align="center"><iframe src="https://view.officeapps.live.com/op/view.aspx?src=https://www.nvidia.com/content/dam/en-zz/zh_cn/Solutions/deep-learning/downloads/1-AcceleratingApplicationswithCUDAC-C/AC_CUDA_C_6.pptx" frameborder="0" width="900" height="550" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe></div>

**网格跨度循环**
为了要创建具有超高性能的执行配置，或出于需要，一个网格中的线程数量可能会小于数据集的大小。请思考一下包含 1000 个元素的数组和包含 250 个线程的网格（此处使用极小的规模以便于说明）。此网格中的每个线程将需使用 4 次。如要实现此操作，一种常用方法便是在核函数中使用网格跨度循环。  

在网格跨度循环中，每个线程将在网格内使用 tid+bid*bdim 计算自身唯一的索引，并对数组内该索引的元素执行相应运算，然后将网格中的线程数添加到索引并重复此操作，直至超出数组范围。例如，对于包含 500 个元素的数组和包含 250 个线程的网格，网格中索引为 20 的线程将执行如下操作：

- 1 对包含 500 个元素的数组的元素 20 执行相应运算
将其索引增加 250，使网格的大小达到 270
- 2 对包含 500 个元素的数组的元素 270 执行相应运算
将其索引增加 250，使网格的大小达到 520
- 3 由于 520 现已超出数组范围，因此线程将停止工作  

也即单线程可以对多个数组下标进行多次操作，以达到大型数组操作的并行要求  
具体示例如下  

``` c
__global void kernel(int *a, int N)
{
  int indexWithinTheGrid = threadIdx.x + blockIdx.x * blockDim.x;
  int gridStride = gridDim.x * blockDim.x;

  for (int i = indexWithinTheGrid; i < N; i += gridStride)
  {
    // do work on a[i];
  }
}

```  

### 作业8 ###  

重构 [03-grid-stride-double.cu](./03-grid-stride-double.cu) 以在 doubleElements 核函数中使用网格跨度循环，进而使小于 N 的网格可以重用线程以覆盖数组中的每个元素。程序会打印数组中的每个元素是否均已加倍，而当前该程序会准确打印出 FALSE。

### Error Handling ###

与在任何应用程序中一样，加速 CUDA 代码中的错误处理同样至关重要。即便不是大多数，也有许多 CUDA 函数（例如，内存管理函数）会返回类型为 cudaError_t 的值，该值可用于检查调用函数时是否发生错误。以下是对调用 cudaMallocManaged 函数执行错误处理的示例：

``` c
cudaError_t err;
err = cudaMallocManaged(&a, N)                    // Assume the existence of `a` and `N`.

if (err != cudaSuccess)                           // `cudaSuccess` is provided by CUDA.
{
  printf("Error: %s\n", cudaGetErrorString(err)); // `cudaGetErrorString` is provided by CUDA.
}
```  

启动定义为返回 void 的核函数后，将不会返回类型为 cudaError_t 的值。为检查启动核函数时是否发生错误（例如，如果启动配置错误），CUDA 提供 cudaGetLastError 函数，该函数会返回类型为 cudaError_t 的值。

```c
/*
 * This launch should cause an error, but the kernel itself
 * cannot return it.
 */

someKernel<<<1, -1>>>();  // -1 is not a valid number of threads.

cudaError_t err;
err = cudaGetLastError(); // `cudaGetLastError` will return the error from above.
if (err != cudaSuccess)
{
  printf("Error: %s\n", cudaGetErrorString(err));
}
```  

最后，为捕捉异步错误（例如，在异步核函数执行期间），请务必检查后续同步 CUDA 运行时 API 调用所返回的状态（例如 cudaDeviceSynchronize）；如果之前启动的其中一个核函数失败，则将返回错误。

[01-add-error-handling.cu](./01-add-error-handling.cu) 会编译、运行并打印已加倍失败的数组元素。不过，该程序不会指明其中是否存在任何错误。重构应用程序以处理 CUDA 错误，以便您可以了解程序出现的问题并进行有效调试。您将需要调查在调用 CUDA 函数时可能出现的同步错误，以及在执行 CUDA 核函数时可能出现的异步错误。

构建宏示例  

```c  
#include <stdio.h>
#include <assert.h>

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

int main()
{

/*
 * The macro can be wrapped around any function returning
 * a value of type `cudaError_t`.
 */

  checkCuda( cudaDeviceSynchronize() )
}
```

### 终极作业 向量相加 ###

[01-vector-add.cu](./01-vector-add.cu) 包含一个可正常运作的 CPU 向量加法应用程序。加速其 addVectorsInto 函数，使之在 GPU 上以 CUDA 核函数运行并使其并行执行工作。

- 1 扩充 addVectorsInto 定义，使之成为 CUDA 核函数。
- 2 选择并使用有效的执行配置，以使 addVectorsInto 作为 CUDA 核函数启动。
- 3 更新内存分配，内存释放以反映主机和设备代码需要访问 3 个向量：a、b 和 result。
- 4 重构 addVectorsInto 的主体：将在单个线程内部启动，并且只需对输入向量执行单线程操作。确保线程从不尝试访问输入向量范围之外的元素，并注意线程是否需对输入向量的多个元素执行操作。
在 CUDA 代码可能以其他方式静默失败的位置添加错误处理。  

### 高阶作业 矩阵乘法 ###  

文件 [01-matrix-multiply-2d.cu](./01-matrix-multiply-2d.cu) 包含一个功能齐全的主机函数 matrixMulCPU。您的任务是扩建 CUDA 核函数 matrixMulGPU。源代码将使用这两个函数执行矩阵乘法，并比较它们的答案以验证您编写的 CUDA 核函数是否正确。

- 1 您将需创建执行配置，其参数均为 dim3 值，且 x 和 y 维度均设为大于 1。
- 2 在核函数主体内部，您将需要按照惯例在网格内建立所运行线程的唯一索引，但应为线程建立两个索引：一个用于网格的 x 轴，另一个用于网格的 y 轴。  

### 高阶作业 Accelerate A Thermal Conductivity Application ###

在下面的练习中，您将为模拟金属银二维热传导的应用程序执行加速操作。

将 [01-heat-conduction.cu](./01-heat-conduction.cu) 内的 step_kernel_mod 函数转换为在 GPU 上执行，并修改 main 函数以恰当分配在 CPU 和 GPU 上使用的数据。step_kernel_ref 函数在 CPU 上执行并用于检查错误。由于此代码涉及浮点计算，因此不同的处理器甚或同一处理器上的简单重排操作都可能导致结果略有出入。为此，错误检查代码会使用错误阈值，而非查找完全匹配。  

此任务中的原始热传导 CPU 源代码取自于休斯顿大学的文章[An OpenACC Example Code for a C-based heat conduction code](http://docplayer.net/30411068-An-openacc-example-code-for-a-c-based-heat-conduction-code.html)（基于 C 的热传导代码的 OpenACC 示例代码）。  

[本题解答](./01-heat-conduction-solution.cu)  
 



