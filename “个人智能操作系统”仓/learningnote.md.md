# 张量化计算与 FEALPy 多后端实践

# task1

四个 Python 科学计算库：

四个库之间的关系可以概括为：

NumPy 负责数据表示与基础运算，SciPy 负责数值算法，SymPy 负责符号推导，Matplotlib 负责结果可视化。



## numpy

实现创建创建张量、张量的形状、张量基本运算、内积（点积）、缩并、爱因斯坦求和

广播机制

```Python
A:  (4, 1)
B:     (2,)   ← 缺的前面补 1，视为 (1, 2)
-----------------
比较: 4↔1    1↔2
      ↓       ↓
    广播成 4  广播成 2
-----------------
结果:  (4, 2)
```

下面三种写法结果完全相同：

```Bash
np.einsum('ij,jk->ik', A, B)
np.dot(A, B)
A @ B   # 实现矩阵乘法
```

### **`einsum`**** 的灵活性**

`einsum` 不仅能做矩阵乘法，还能表达很多其他运算：

1. 矩阵转置

```Bash
np.einsum('ij->ji', A)
# 等价于 A.T
```

2. 逐元素乘法

```Bash
np.einsum('ij,ij->ij', A, B)
# 等价于 A * B
```

3. 求迹

```Bash
np.einsum('ii->', A)
# 等价于 np.trace(A)
```

## SciPy

### 线性代数运算（矩阵求逆、特征值、特征向量）

SciPy 的 `linalg` 模块提供矩阵求逆、特征值和特征向量计算等功能。

```Python
from scipy import linalg
import numpy as np

A = np.array([[1, 2], [3, 4]])

A_inv = linalg.inv(A)

eigvals, eigvecs = linalg.eig(A)

print("逆矩阵：", A_inv)
print("特征值：", eigvals)
print("特征向量：", eigvecs)
```

`eigvals` 存储特征值，`eigvecs` 的每一列对应一个特征向量。

### 稀疏矩阵

稀疏矩阵是指绝大多数元素为零、非零元素相对较少的矩阵。

|比较内容|COO|CSR|CSC|
|---|---|---|---|
|核心思想|记录坐标|按行压缩|按列压缩|
|data|非零元素|非零元素|非零元素|
|indices |行号与列号|列号|行号|
|indptr|不需要|行起始位置|列起始位置|
|主要优势|容易构造|按行操作高效|按列操作高效|
|常见用途|组装稀疏矩阵|矩阵向量乘法、迭代求解|列操作、部分直接求解算法|

（1）COO 格式

使用三个数组分别存储非零值、行索引和列索引。

```Plain Text
data = [10, 21, 33, 3, 12, 1, 4]
row  = [0, 1, 1, 2, 3, 3, 3]
col  = [3, 0, 2, 2, 0, 1, 3]
```

（2）CSR 格式

```Plain Text
data    = [10, 21, 33, 3, 12, 1, 4]
indices = [3, 0, 2, 2, 0, 1, 3]
indptr  = [0, 1, 3, 4, 7]
```

其中：

`data[indptr[i]:indptr[i+1]]` 表示第 i 行的全部非零元素

`indices` 的对应切片表示这些元素所在的列。

（3）CSC 格式

```Python
data    = [21, 12, 1, 33, 3, 10, 4]
indices = [1, 3, 3, 1, 2, 0, 3]
indptr  = [0, 2, 3, 5, 7]
```

CSC 与 CSR 的原理相似，但它按列压缩存储。



#### 稀疏矩阵的创建与求解

```Python
from scipy import sparse
from scipy.sparse.linalg import spsolve
import numpy as np

data = [1, 2, 3, 4]
row = [0, 0, 1, 2]
col = [0, 2, 1, 2]

A = sparse.csr_matrix(
    (data, (row, col)), shape=(3, 3)
)

b = np.array([1, 2, 3])

x = spsolve(A, b)
print(x)
```

`scipy.sparse`：创建、转换和操作稀疏矩阵。

`spsolve`：求解稀疏线性方程组。



### 数值求导与积分

`approx_fprime`：用有限差分近似计算导数（数值微分）；

`approx_fprime` 是为多变量函数设计的，它的输入是数组（向量）。

`integrate.quad`：用自适应数值方法计算定积分（数值积分）。





## SymPy：符号运算

SymPy 主要用于数学表达式的符号运算，与 NumPy、SciPy 进行数值计算的方式不同。

数值计算通常得到具体近似数值，符号计算则可以保留变量与表达式的数学形式，适合公式推导与解析结果验证。



## Matplotlib：数值结果可视化

### 基本绘图结构

Matplotlib 的核心概念包括：

- `Figure`：画布，容纳整个图形。

- `Axes`：绘图区，包含坐标轴、曲线等内容。

- `pyplot`：提供常用绘图接口，通常使用 `plt` 调用。

**折线图、热图可视化张量、向量场可视化**



# task2

### 后端的基本概念

在科学计算中，后端（Backend）指负责执行张量运算的底层计算库，例如 NumPy、PyTorch 和 JAX。

不同计算后端的功能有所区别。

NumPy 实现：

```Plain Text
import numpy as np

def derivative_numpy(f, x0, h=1e-6):
    x1 = x0 + h
    x2 = x0 - h
    return (f(x1) - f(x2)) / (2*h)

x = np.array([0.5, 1.0, 1.23, 2.0])

print(derivative_numpy(np.sin, x))
```

PyTorch 实现：

```Plain Text
import torch

def derivative_pytorch(f, x0, h=1e-6):
    x1 = x0 + h
    x2 = x0 - h
    return (f(x1) - f(x2)) / (2*h)

x = torch.tensor([0.5, 1.0, 1.23, 2.0])

print(derivative_pytorch(torch.sin, x))
```

可以发现，两种代码的算法逻辑基本一致，只是底层张量创建方式与计算函数不同。

PyTorch 的自动微分

除数值差分外，PyTorch 还可以通过 Autograd 自动微分机制计算导数。

自动微分会记录计算过程，再按链式法则求导，不需要选择差分步长。

```Python
import torch
def derivative_autograd(x0):
    x = torch.tensor(
        x0, requires_grad=True
    )

    y = torch.sin(x)

    g = torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(x)
    )[0]

    return g

x = [0.5, 1.0, 1.23, 2.0]
print(derivative_autograd(x))
```

### 为什么需要统一后端

在实际科研中，可能需要先使用 NumPy 在 CPU 上调试程序，再切换到 PyTorch，利用 GPU 进行加速。

FEALPy 后端管理器（bm）提供统一接口，屏蔽底层实现差异



### FEALPy 后端管理器

### backend\_manager 的作用

FEALPy 使用 `backend_manager` 统一管理张量后端，通常简写为 `bm`。

```Python
from fealpy.backend import backend_manager as bm
```

之后就可以使用 `bm` 提供的统一接口进行数值运算。



FealPy 分别用 NumPy 和 PyTorch 运行同一段上层代码：

```Python
from fealpy.backend import backend_manager as bm

def fd_derivative_bm(f, x0, h=1e-6):
    x1 = bm.array([x0 + h], dtype=bm.float64)
    x2 = bm.array([x0 - h], dtype=bm.float64)
    return ((f(x1) - f(x2)) / (2*h)).item()

# 使用 NumPy 后端
bm.set_backend('numpy')
print('backend numpy:', fd_derivative_bm(bm.sin, 1.23))

# 使用 PyTorch 后端
bm.set_backend('pytorch')
print('backend pytorch:', fd_derivative_bm(bm.sin, 1.23))
```

只需分别调用 `bm.set_backend('numpy')` 和 `bm.set_backend('pytorch')`，上层的 `fd_derivative_bm` 函数不需要修改就能在不同后端运行。



**多后端稀疏运算。**

`fealpy.sparse` 可以用 COO 或 CSR 表示矩阵。构造 COO 时，行号、列号使用整数张量，值使用浮点张量，三者都在当前后端和设备上创建。

```Python
from fealpy.backend import backend_manager as bm
from fealpy.sparse import coo_matrix, csr_matrix

bm.set_backend('numpy')
I = bm.array([0, 1, 1, 2, 3], dtype=bm.int64)
J = bm.array([1, 0, 2, 1, 3], dtype=bm.int64)
K = bm.array([2, 3, 3, 1, 2], dtype=bm.int64)
V = bm.array([1., -2., 1., -2., 1.], dtype=bm.float64)

A1 = coo_matrix((V, (I, J, K)), shape=(4, 4, 4))
A2 = coo_matrix((None, (I, J, K)), shape=(4, 4, 4))
B = csr_matrix((V, (I, J)), shape=(4, 4))
```

**计时与日志。**

FEALPy 的 `timer` 通过标签记录相邻步骤的耗时，比较乘法速度时要把后端切换、数据构造和首次预热放在计时区间之外。

`logger` 用 `DEBUG`、`INFO`、`WARNING`、`ERROR` 表示不同级别的信息。在本次实验中，`INFO` 记录后端、矩阵规模与非零元数。即使计时表打印失败，也能从最后一条日志看出矩阵已经构造完成。



# 实践练习

## 一、张量加权求和

- `np.arange(24)`：生成从 `0` 到 `23` 的 24 个整数。

- `.reshape(2, 3, 4)`：把这些数排成“2 层，每层 3 行，每行 4 个数”。

第一层是：

```Python
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
```

第二层是：

```Python
[[12 13 14 15]
 [16 17 18 19]
 [20 21 22 23]]
```

4 个权重：

```Python
w = np.array([1, 2, -1, 0.5])
```

**每行的 4 个数分别乘上这 4 个权重，再相加。**

例如第一行：

0x1\+1x2\+2x\(\-1\)\+3x0\.5=1\.5

原来每行有 4 个数，计算后变成 1 个数，所以形状从 `(2, 3, 4)` 变成 `(2, 3)`。

```Python
import numpy as np

# 1. 创建三维张量：2 层，每层 3 行，每行 4 个数
T = np.arange(24).reshape(2, 3, 4)

# 2. 为每行的 4 个数指定权重
w = np.array([1, 2, -1, 0.5])

# 3. 沿最后一个轴进行加权求和
M = np.einsum("ijk,k->ij", T, w)

print("T 的形状：", T.shape)
print("w 的形状：", w.shape)
print("M 的形状：", M.shape)
print("加权求和结果：")
print(M)

# 4. 用最直观的循环重新计算，验证 einsum 的结果
M_check = np.zeros((2, 3))

for i in range(2):
    for j in range(3):
        for k in range(4):
            M_check[i, j] += T[i, j, k] * w[k]

print("两种计算方式是否一致：", np.allclose(M, M_check))
```

输出

```Python
T 的形状： (2, 3, 4)
w 的形状： (4,)
M 的形状： (2, 3)
加权求和结果：
[[ 1.5 11.5 21.5]
 [31.5 41.5 51.5]]
两种计算方式是否一致： True
```

## 二、稀疏矩阵求解

需要求解的是 A u = b。其中 A、b 由题目给定的规则生成，未知量是向量 u。

1. n=50 表示区间 \(0,1\) 内有 50 个点，不含端点。

2. h=1/\(n\+1\)=1/51；内点坐标是 h、2h、……、50h。使用 `np.arange(1,n+1)*h` 可准确生成这些内点。

3. A 的主对角线为 2/h²，上下相邻对角线均为 \-1/h²，其余位置都是 0。

4. b 的第 i 项是 x\_i\(1\-x\_i\)。

5. `spsolve(A,b)` 直接求解这个稀疏线性方程组，无须先算逆矩阵。

6. 将 x\_i 与 u\_i 画成折线图。

[稀疏矩阵求解](https://github.com/ouyangjianghua-suanhai/task5-screenshots/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202026-11-19%20194801.png)

### n=3 时的手工计算

h=1/4，内点坐标为 \(1/4,1/2,3/4\)。所以：

```Plain Text
A = [[ 32, -16,   0],
     [-16,  32, -16],
     [  0, -16,  32]]

b = [3/16, 1/4, 3/16]
```

对应的三个方程是：

```Plain Text
32u1 - 16u2          = 3/16   （1）
-16u1 + 32u2 - 16u3  = 1/4    （2）
        -16u2 + 32u3 = 3/16   （3）
```

最终：

```Plain Text
u = [5/256, 7/256, 5/256]
  = [0.01953125, 0.02734375, 0.01953125]
```

```Python
def solve_sparse_system(n):
    """建立 n 个内点对应的三对角稀疏矩阵，并求解 A @ u = b。"""
    h = 1.0 / (n + 1)

    # 内点不包括 0 和 1：h、2h、……、nh。
    x = np.arange(1, n + 1) * h

    # 主对角线有 n 项；上下相邻对角线分别有 n-1 项。
    main_diagonal = np.full(n, 2.0 / h**2)
    side_diagonal = np.full(n - 1, -1.0 / h**2)

    # offsets=-1、0、1 分别表示下对角线、主对角线、上对角线。
    # CSR 是稀疏存储格式，避免将大矩阵中的所有零元素都存起来。
    A = diags(
        [side_diagonal, main_diagonal, side_diagonal],
        offsets=[-1, 0, 1],
        shape=(n, n),
        format="csr",
    )

    b = x * (1.0 - x)
    u = spsolve(A, b)
    return x, A, b, u

def task2(output_dir):
    """完成 n=50 的计算和绘图，再用 n=3 的手算答案验证实现。"""
    print("\n第二题：三对角稀疏方程组")

    x, A, b, u = solve_sparse_system(n=50)

    # 残差越接近 0，说明算出的 u 越能满足这个离散方程组。
    # 这不是数值解与连续问题精确解之间的误差。
    residual = np.max(np.abs(A @ u - b))
    print("n=50 的矩阵形状：", A.shape)
    print("矩阵非零元素个数：", A.nnz)
    print("解向量前 5 项：", u[:5])
    print(f"最大绝对残差 max|Au-b|：{residual:.3e}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        x, u, color="#2563eb", marker="o", markersize=3,
        linewidth=1.8, label="稀疏方程组数值解"
    )
    ax.set_title("第二题：三对角稀疏方程组的解（n=50）")
    ax.set_xlabel(r"内点坐标 $x_i$")
    ax.set_ylabel(r"解 $u_i$")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    ax.text(
        0.03, 0.96, f"最大绝对残差：{residual:.2e}",
        transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(facecolor="white", edgecolor="#cbd5e1", alpha=0.9),
    )
    fig.tight_layout()
    fig.savefig(output_dir / "task2_sparse_solution.png", dpi=200)

    # 小规模验证：h=1/4，主对角线为 32，上下对角线为 -16。
    # 手算得到 u=(5/256, 7/256, 5/256)。推导见配套说明。
    x_small, A_small, b_small, u_small = solve_sparse_system(n=3)
    u_manual = np.array([5.0 / 256, 7.0 / 256, 5.0 / 256])
    agrees = np.allclose(u_small, u_manual, rtol=0, atol=1e-12)

    print("\nn=3 的内点坐标：", x_small)
    print("n=3 的矩阵 A：\n", A_small.toarray())
    print("n=3 的右端向量 b：", b_small)
    print("n=3 的程序解：", u_small)
    print("n=3 的手算解：", u_manual)
    print("程序解与手算解是否一致：", agrees)
    if not agrees:
        raise AssertionError("n=3 的程序解未通过手算对照验证。")

    return x, A, b, u
```



## 三、改造积分案例

目标函数为 $u(x)=e^{-x}\cos(2x)$，在 $[0,2\pi]$ 上连续可积

`IntegralExperiment` 把函数、上下限与采样点数保存在同一个对象里。SymPy 对符号表达式积分，再由 `lambdify` 把同一表达式转换为可以接收 NumPy 数组的函数。这样修改目标函数时，两种计算会一起更新。

[改造积分](https://github.com/ouyangjianghua-suanhai/task5-screenshots/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202026-11-19%20195415.png)

```Python
class IntegralExperiment:
    """把符号计算、数值计算、误差比较和绘图组织在一个类里。"""

    def __init__(self, n=1001):
        # 1001 个等距采样点，对应 1000 个小区间。
        self.n = n
        self.symbol = sp.symbols("x", real=True)
        self.expression = sp.exp(-self.symbol) * sp.cos(2 * self.symbol)

        # 保留 SymPy 的精确 pi，避免把符号积分上限提前变成近似小数。
        self.lower = sp.Integer(0)
        self.upper = 2 * sp.pi

    def symbolic_integral(self):
        """SymPy：先算出精确表达式，作为对照基准。"""
        return sp.simplify(
            sp.integrate(
                self.expression, (self.symbol, self.lower, self.upper)
            )
        )

    def sample_function(self):
        """NumPy：产生采样坐标及函数值。"""
        x = np.linspace(float(self.lower), float(self.upper), self.n)

        # 将同一个符号表达式变成可作用于 NumPy 数组的数值函数。
        # 这样符号积分与数值积分使用的函数保持一致。
        function = sp.lambdify(self.symbol, self.expression, modules="numpy")
        y = function(x)
        return x, y

    def numerical_integral(self, x, y):
        """SciPy：用复合 Simpson 公式进行数值积分。"""
        return float(simpson(y, x=x))

    def plot(self, x, y, integral_value, error, output_dir):
        """Matplotlib：绘制完整曲线，但只填充正值区域。"""
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(
            x, y, color="#4338ca", linewidth=2,
            label=r"$u(x)=e^{-x}**\c**os(2x)$",
        )
        ax.axhline(0, color="#64748b", linewidth=1, linestyle="--")

        # where 控制填充范围；interpolate 处理曲线与零轴的交点。
        ax.fill_between(
            x, 0, y, where=(y > 0), interpolate=True,
            color="#38bdf8", alpha=0.4, label="仅填充 u(x)>0 的区域",
        )

        ax.text(
            0.97, 0.95,
            f"全区间数值积分：{integral_value:.12f}\n"
            f"绝对误差：{error:.3e}\n采样点数：{self.n}",
            transform=ax.transAxes, ha="right", va="top", fontsize=11,
            bbox=dict(facecolor="white", edgecolor="#cbd5e1", alpha=0.95),
        )
        ax.set_title("第三题：衰减振荡函数的积分与正值区域")
        ax.set_xlabel("x")
        ax.set_ylabel("u(x)")
        ax.set_xlim(float(self.lower), float(self.upper))
        ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        ax.set_xticklabels(["0", r"$**\p**i/2$", r"$**\p**i$", r"$3**\p**i/2$", r"$2**\p**i$"])
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(output_dir / "task3_integral.png", dpi=200)

    def run(self, output_dir):
        """依次完成符号积分、采样、数值积分、误差比较、绘图。"""
        print("\n第三题：积分综合案例")
        exact_expression = self.symbolic_integral()
        exact_value = float(sp.N(exact_expression, 17))
        x, y = self.sample_function()
        numerical_value = self.numerical_integral(x, y)

        # 绝对误差：数值积分与精确积分参考值之差的绝对值。
        error = abs(numerical_value - exact_value)
        agrees = np.isclose(numerical_value, exact_value, rtol=0, atol=1e-7)

        print("目标函数：", self.expression)
        print("积分区间：[0, 2*pi]")
        print("符号积分结果：", exact_expression)
        print(f"精确积分参考值：{exact_value:.15f}")
        print(f"数值积分结果：  {numerical_value:.15f}")
        print(f"绝对误差：{error:.3e}")
        print("误差是否小于 1e-7：", agrees)
        if not agrees:
            raise AssertionError("数值积分未通过精确积分对照验证。")

        # 积分计算使用完整的 y，包含正值与负值；只有绘图填充限制为正值。
        self.plot(x, y, numerical_value, error, output_dir)
        return exact_expression, numerical_value, error

def main():
    # 文件移动到其他文件夹后，图片仍保存到 Python 文件旁边。
    output_dir = Path(__file__).resolve().parent
    np.set_printoptions(precision=10, suppress=True)

    task2(output_dir)
    experiment = IntegralExperiment(n=1001)
    experiment.run(output_dir)

    plt.show()

if __name__ == "__main__":
    main()

```

|核对项|结果|
|---|---|
|符号积分 $(1-e^{-2\pi})/5$ 的小数值|0\.199626511453658|
|1001 个采样点的 Simpson 积分|0\.199626511358590|
|绝对误差|9\.507 × 10⁻¹¹|



## 四、多后端数值一致性与切换实验

对 `sin(x)` 求导，使用步长 `h=1e-6` 的中心差分。每次先切换后端，再创建 `float64` 张量。`bm.sin` 和数组运算由当前后端执行，算完才把结果转成 NumPy 数组做统一比较。

[后端切换](https://github.com/ouyangjianghua-suanhai/task5-screenshots/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202026-11-19%20195625.png)

```Python
import numpy as np
from fealpy.backend import backend_manager as bm

# 同一个数值微分函数，供两个后端使用
def numerical_derivative(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)

# 选取 4 个输入点
points = [0.5, 1.0, 1.23, 2.0]
exact = np.cos(points)
results = {}

for backend in ["numpy", "pytorch"]:
    # 先切换后端，再创建数据
    bm.set_backend(backend)

    if backend == "pytorch":
        bm.set_default_device("cpu")

    # 两个后端统一使用双精度
    x = bm.array(points, dtype=bm.float64)

    derivative = numerical_derivative(bm.sin, x)

    # 转为 NumPy 数组，方便统一打印和比较
    results[backend] = bm.to_numpy(derivative).copy()

    print(f"\n后端：{backend}")
    print("数值导数：", results[backend])
    print("精确导数：", exact)

    error = np.max(np.abs(results[backend] - exact))
    print("相对精确导数的最大绝对误差：", error)

backend_error = np.max(
    np.abs(results["numpy"] - results["pytorch"])
)

print("\n两个后端之间的最大绝对差：", backend_error)
print(
    "两个后端的结果是否接近：",
    np.allclose(
        results["numpy"],
        results["pytorch"],
        atol=1e-8,
        rtol=0,
    ),
)

```

|输入 $x$|NumPy 数值导数|PyTorch 数值导数|解析参考 $\cos(x)$|
|---|---|---|---|
|0\.50|0\.87758256|0\.87758256|0\.87758256|
|1\.00|0\.54030231|0\.54030231|0\.54030231|
|1\.23|0\.33423773|0\.33423773|0\.33423773|
|2\.00|−0\.41614684|−0\.41614684|−0\.41614684|



## 五、稀疏矩阵\-向量乘法性能对比

测试 1000、10000、100000、1000000 阶稀疏矩阵，在两个后端上进行矩阵—向量乘法，记录耗时并分析。

使用 COO 格式保存对角矩阵 `A`，对角值从 1 等距取到 2。每个规模都固定随机种子 `2026`，使两个后端拿到相同的随机向量。

```Python
from time import perf_counter
from fealpy.sparse import coo_matrix
from fealpy.utils import timer
from fealpy import logger

# 让 INFO 级别的日志也显示出来
logger.setLevel("INFO")

sizes = [1000, 10000, 100000, 1000000]
repeats = 5
single_ms = {}

for backend in ["numpy", "pytorch"]:
    bm.set_backend(backend)

    if backend == "pytorch":
        bm.set_default_device("cpu")

    logger.info(f"切换到后端：{backend}，设备：CPU")

    for n in sizes:
        logger.info(f"开始构造 {n} 阶对角稀疏矩阵")

        # 每组重置随机种子，使两个后端使用相同输入
        rng = np.random.default_rng(2026)

        diagonal_numpy = np.linspace(1.0, 2.0, n)
        vector_numpy = rng.random(n)

        # 索引与数值都转换为当前后端的数据
        index = bm.arange(n, dtype=bm.int64)
        values = bm.array(diagonal_numpy, dtype=bm.float64)
        vector = bm.array(vector_numpy, dtype=bm.float64)

        # 第 k 个非零元素位于第 k 行、第 k 列
        A = coo_matrix(
            (values, (index, index)),
            shape=(n, n),
        )

        # 预热一次，不放入正式计时
        y = A @ vector

        logger.info(
            f"矩阵构造完成：n={n}，非零元素={A.nnz}，开始计时"
        )

        # 每轮持续做乘法至少 50 ms，让 FEALPy 能分辨起止时间。
        # timer 表给出整轮耗时；作业汇总表给出折算后的单次耗时。
        samples = []
        tmr = timer()
        next(tmr)
        for k in range(repeats):
            count = 0
            start = perf_counter()
            while True:
                y = A @ vector
                count += 1
                elapsed = perf_counter() - start
                if elapsed >= 0.05:
                    break
            tmr.send(f"{backend}, n={n}, 第{k + 1}轮，共{count}次乘法")
            samples.append(1000 * elapsed / count)
        tmr.send(None)
        tmr.close()
        single_ms[backend, n] = float(np.median(samples))

        # 对角矩阵的乘法结果可以直接逐元素验证
        expected = diagonal_numpy * vector_numpy
        correct = np.allclose(
            bm.to_numpy(y),
            expected,
            atol=1e-12,
            rtol=1e-12,
        )

        print(
            f"结果验证：backend={backend}, "
            f"n={n}, 是否正确={correct}"
        )

        logger.info(f"完成测试：backend={backend}, n={n}")
print("\n单次乘法耗时（毫秒，5 轮平均单次耗时的中位数）")
print("矩阵阶数    NumPy CPU/ms    PyTorch CPU/ms")
for n in sizes:
    print(f"{n:>8}  {single_ms['numpy', n]:>14.6f}  {single_ms['pytorch', n]:>14.6f}")

```

[性能对比](https://github.com/ouyangjianghua-suanhai/task5-screenshots/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202026-11-19%20195905.png)

|矩阵阶数|非零元素数|NumPy CPU（毫秒/次）|PyTorch CPU（毫秒/次）|两组乘法结果验证|
|---|---|---|---|---|
|1,000|1,000|0\.002300|0\.021896|均通过|
|10,000|10,000|0\.008426|0\.103441|均通过|
|100,000|100,000|0\.088636|1\.023141|均通过|
|1,000,000|1,000,000|5\.039210|9\.865433|均通过|



