import torch
import triton
import triton.language as tl

@triton.jit
def weighted_sum_fwd(
    x_ptr, weight_ptr,
    output_ptr,
    x_stride_row, x_stride_dim,
    weight_stride_dim,
    output_stride_row,
    ROWS,D,
    ROWS_TILE_SIZE:tl.constexpr, D_TILE_SIZE:tl.constexpr,#声明编译时常量，Tile分块的形状在编译时必须为已知
    ):
    row_tile_idx = tl.program_id(0)#检查正在运行哪个thread block，即获取当前线程块处理的张量子块

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(ROWS, D,),
        strides = (x_stride_row, x_stride_dim),
        offsets = (row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape = (ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1,0),#列优先的存储顺序
    )
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides = (weight_stride_dim,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),#行优先
    )
    output_block_ptr=tl.make_block_ptr(
        output_ptr,
        shape=(ROWS,),
        strides = (output_stride_row,),
        offsets = (row_tile_idx*ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    #initialize a buffer to write to
    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)

    for i in range(tl.cdiv(D,D_TILE_SIZE)):#向上取整的除法
        #load the current block pointer
        #考虑行、列无法整除块，需要对2个维度进行边界检查
        row = tl.load(x_block_ptr, boundary_check=(0,1), padding_option = 'zero')#从指向的内存位置加载数据，只对第二个维度进行边界检查，如果超出边界，填充为0
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option='zero')

        #compute the weighted sum of the row
        output += tl.sum(row*weight[None,:], axis = 1)

        #移动指针到下一个块
        x_block_ptr = x_block_ptr.advance(0, D_TILE_SIZE)
        weight_block_ptr = weight_block_ptr.advance(D_TILE_SIZE,)
    
    tl.store(output_block_ptr, output, boundary_check=(0,))


#wrap the kernel in a pytorch autograd function   
import torch, einops
from einops import rearrange
class WeightedSumFunc(torch.autograd.Function): #torch.autofrad.Function是 PyTorch 中用于实现自定义前向和反向传播逻辑的基类。
    @staticmethod 
    #静态方法不依赖类实例，可以直接通过类名调用。
    def forward(ctx, x, weight):
        D, output_dims = x.shape[-1, ], x.shape[:-1]

        input_shape = x.shape
        #输入二维化
        x=rearrange(x, '... d -> (...) d') # '... d -> (...) d' 表示将所有前面的维度合并为一个维度，最后一维保持不变。

        ctx.save_for_backward(x, weight)

        assert len(weight.shape) ==1 and weight.shape[0]==D, 'Dimentsion mismatch'
        assert x.is_cuda and weight.is_cuda, 'Expected CUDA tensors'
        assert x.is_contiguous(), 'Our pointer arithmetic will assume contiguous x'

        ctx.D_TILE_SIZE = triton.next_power_of_2(D)//16 #将 D 向上取整到最近的 2 的幂，再将结果除以16，表示每个线程块处理的列数
        ctx.ROWS_TILE_SIZE=16 # 每个线程同时处理16行数据
        ctx.input_shape = input_shape
        
        #初始化一个空的结果张量，但元素不一定为0
        y = torch.empty(output_dims, device = x.device)

        n_rows = y.numel()#输出y的元素总数

        #weight_sum_fwd函数是已经被@triton.jit装饰的函数体，因此在Triton中，可以使用[]，即用于指定内核的网格大小（grid size），即 GPU 上线程块的分布方式。
        #triton的内核函数调用语法如下：kernel[grid](args)
        #这里的grid是一个元组
        weighted_sum_fwd[(triton.cdiv(n_rows, ctx.ROWS_TILE_SIZE),)](
            x, weight,
            y,
            x.stride(0), x.stride(1),#输出维度的步长
            weight.stride(0),
            y.stride(0),
            ROWS=n_rows, D=D,
            ROWS_TILE_SIZE = ctx.ROWS_TILE_SIZE, D_TILE_SIZE = ctx.D_TILE_SIZE, 
        )
        return y.view(input_shape[:-1])

@triton.jit
def weighted_sum_backward(
    x_ptr, weight_ptr, # input
    grad_output_ptr, # grad input
    grad_x_ptr, partial_grad_weight_ptr, # grad outputs
    stride_xr, stride_xd,
    stride_wd,
    stride_gr,
    stride_gxr, stride_gxd,
    stride_gwb, stride_gwd,
    NUM_ROWS, D,
    ROWS_TILE_SIZE:tl.constexpr, D_TILE_SIZE:tl.constexpr,
):
    row_title_idx = tl.program_id(0)
    n_row_tiles = tl.num_programs(0)

    #inputs
    grad_outputs_block_ptr = tl.make_block_ptr(
        grad_output_ptr,
        shape = (NUM_ROWS,), strides = (stride_gr,),
        offsets = (row_title_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    x_block_ptr = tl.make_block_ptr(
    x_ptr,
    shape=(NUM_ROWS, D,), strides=(stride_xr, stride_xd),
    offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
    block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
    order=(1, 0),
    )

    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,), strides=(stride_wd,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    grad_x_block_ptr = tl.make_block_ptr(
        grad_x_ptr,
        shape=(NUM_ROWS, D,), strides=(stride_gxr, stride_gxd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
        )

    partial_grad_weight_block_ptr = tl.make_block_ptr(
        partial_grad_weight_ptr,
        shape=(n_row_tiles, D,), strides=(stride_gwb,stride_gwd),
        offsets=(row_tile_idx, 0),
        block_shape=(1, D_TILE_SIZE),
        order=(1,0),
    )

    for i in range(tl.cdiv(D,D_TILE_SIZE)):
        grad_output = tl.load(grad_outputs_block_ptr, boundary_check=(0,), padding_option='zero')#加载输出梯度

        #outer product for grad_x
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option='zero')
        grad_x_row = grad_output[:, None] * weight[None, :] #输出梯度与w的外积
        tl.store(grad_x_block_ptr, grad_x_row, boundary_check=(0,1))

        #权重梯度
        row = tl.load(x_block_ptr, boundary_check=(0,1), padding_option='zero')
        grad_weight_row = tl.sum(row*grad_output[:,None], axis=0, keep_dims=True)
        tl.store(partial_grad_weight_block_ptr, grad_weight_row, boundary_check=(1,))

        #移走下一个块的指针
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
        partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance((0,D_TILE_SIZE))
        grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE))


class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        D, output_dims = x.shape[-1, ], x.shape[:-1]
        input_shape = x.shape
        #输入二维化
        x=rearrange(x, '... d -> (...) d') # '... d -> (...) d' 表示将所有前面的维度合并为一个维度，最后一维保持不变。

        ctx.save_for_backward(x, weight)

        assert len(weight.shape) ==1 and weight.shape[0]==D, 'Dimentsion mismatch'
        assert x.is_cuda and weight.is_cuda, 'Expected CUDA tensors'
        assert x.is_contiguous(), 'Our pointer arithmetic will assume contiguous x'

        ctx.D_TILE_SIZE = triton.next_power_of_2(D)//16 #将 D 向上取整到最近的 2 的幂，再将结果除以16，表示每个线程块处理的列数
        ctx.ROWS_TILE_SIZE=16 # 每个线程同时处理16行数据
        ctx.input_shape = input_shape
        
        #初始化一个空的结果张量，但元素不一定为0
        y = torch.empty(output_dims, device = x.device)

        n_rows = y.numel()#输出y的元素总数
        #weight_sum_fwd函数是已经被@triton.jit装饰的函数体，因此在Triton中，可以使用[]，即用于指定内核的网格大小（grid size），即 GPU 上线程块的分布方式。
        #triton的内核函数调用语法如下：kernel[grid](args)
        #这里的grid是一个元组

        #在kernel中访问内存的1D tensor，方式为:x[row, col]：内存地址=x_ptr + row*stride(0) + col * stride(1)
                                                            # =x_ptr + row*64 + col*1
        weighted_sum_fwd[(triton.cdiv(n_rows, ctx.ROWS_TILE_SIZE),)](
            x, weight,
            y,
            x.stride(0), x.stride(1),#输出维度的步长，stride() 返回一个元组，表示在内存中沿每个维度移动一个元素需要跳过的元素数量。例如#stride(0)=4:沿第0维（行）移动1行，需要跳过4个元素
            weight.stride(0),
            y.stride(0),
            ROWS=n_rows, D=D,
            ROWS_TILE_SIZE = ctx.ROWS_TILE_SIZE, D_TILE_SIZE = ctx.D_TILE_SIZE, 
        )
        return y.view(input_shape[:-1])

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        ROWS_TILE_SIZE, D_TILE_SIZE,= ctx.ROWS_TILE_SIZE, ctx.D_TILE_SIZE,
        n_rows, D = x.shape #.shape返回元组，可以与解包的特性配合使用

        partial_grad_weight = torch.empty((triton.cdiv(n_rows, ROWS_TILE_SIZE), D), device = x.device, dtype=x.dtype)
        grad_x = torch.empty_like(x)
        #torch.empty_like用于创建张量的核心函数，核心作用是生成一个和指定张量「形状、数据类型、设备、布局完全相同」但未初始化的空张量——“未初始化” 意味着张量内的值是内存中的随机垃圾值，不会自动填充 0 或其他默认值
        #torch.empty()是手动指定，而不是模仿已有的张量创建未初始化张量
        weighted_sum_backward[(triton.cdiv(n_rows, ROWS_TILE_SIZE),)](
            x, weight,
            grad_out,
            grad_x, partial_grad_weight,
            x.stride(0), x.stride(1),
            weight.stride(0),
            grad_out.stride(0),
            grad_x.stride(0),grad_x.stride(1),
            partial_grad_weight.stride(0), partial_grad_weight.stride(1),
            NUM_ROWS = n_rows, D=D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE, D_TILE_SIZE=D_TILE_SIZE,
        )
        grad_weight = partial_grad_weight.sum(axis=0)
        return grad_x, grad_weight
    

def no_triton_forward(x, weight):
    """
    不使用 Triton 的 PyTorch 原生实现
    
    功能：计算加权和 output[i] = sum(x[i, :] * weight[:])
    
    对比 Triton 版本：
    - PyTorch: 一行代码 (x * weight).sum(dim=-1)
    - Triton:  需要手动管理内存指针、tile 大小、循环等（~50 行代码）
    """
    # PyTorch 原生实现：简单直观
    # 自动处理广播：(N, D) * (D,) -> (N, D)
    # 自动求和：sum(dim=-1) -> (N,)
    return (x * weight).sum(dim=-1)


class WeightedSumPyTorch(torch.autograd.Function):
    """
    PyTorch 原生实现的加权求和（不使用 Triton）
    
    对比 Triton 版本的主要区别：
    1. 无需手动管理内存指针和 stride
    2. 无需指定 tile 大小和 grid 配置
    3. 代码简洁，易于理解和维护
    4. 性能可能略低于 Triton（但通常足够好）
    """
    @staticmethod
    def forward(ctx, x, weight):
        """
        x: (..., D) 任意形状，最后一维是 D
        weight: (D,)
        输出: (...,) 除了最后一维的所有维度
        """
        # 保存用于反向传播
        ctx.save_for_backward(x, weight)
        ctx.input_shape = x.shape
        
        # PyTorch 原生实现：一行代码！
        # 自动处理：
        # - 广播：(..., D) * (D,) -> (..., D)
        # - 求和：sum(dim=-1) -> (...,)
        output = (x * weight).sum(dim=-1)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        自动计算梯度
        
        grad_output: (...,) 输出的梯度
        返回: grad_x: (..., D), grad_weight: (D,)
        """
        x, weight = ctx.saved_tensors
        
        # 计算 x 的梯度
        # grad_output: (...,) -> (..., 1) 通过广播
        # weight: (D,) -> (1, ..., 1, D) 通过广播
        # 结果: (..., D)
        grad_x = grad_output.unsqueeze(-1) * weight
        
        # 计算 weight 的梯度
        # grad_output: (...,) -> (..., 1) 通过广播
        # x: (..., D)
        # 结果: (..., D) -> sum over all dims except last -> (D,)
        grad_weight = (grad_output.unsqueeze(-1) * x).sum(dim=tuple(range(x.ndim - 1)))
        
        return grad_x, grad_weight



if '__name__' == 'main':
    x = torch.randn(3,2,device ='cuda')
    weight = torch.randn(2,3,device ='cuda')
    f_weightedsum = WeightedSumFunc.apply # torch.autograd.Function 的一个方法，用于调用自定义的前向和反向传播逻辑。
    result = f_weightedsum(x, weight) #ctx因为是autograd.Function的入口封装方法打来的，因此会自动创建一个ctx（上下文对象），因此只需要关注ctx后面的参数
    #forward 和 backward 就是通过固定的函数名称（关键字） 被 PyTorch 识别的 ——PyTorch 内部会严格检查继承自 autograd.Function 的类是否实现了名为 forward 和 backward 的静态方法，
    # 这是 PyTorch 自动微分机制的 “约定式编程” 规则







def test_timing_flash_forward_backward():
    # TODO: 需要先实现 FlashAttention2 类
    # from your_flash_attention_module import FlashAttention2
    
    n_heads = 16
    d_head = 64
    sequence_length = 16384
    q, k, v = torch.randn(3, n_heads, sequence_length, d_head, device='cuda', dtype=torch.bfloat16, requires_grad=True)

    # flash = torch.compile(FlashAttention2.apply)
    
    def flash_forward_backward():
        # o = flash(q, k, v, True)
        # loss = o.sum()
        # loss.backward()
        pass  # TODO: 实现 FlashAttention 后再启用
    
    # results = triton.testing.do_bench(flash_forward_backward, rep=10000, warmup=1000)
    # print(results)


def compare_triton_vs_pytorch():
    """
    对比 Triton 和 PyTorch 原生实现的代码差异
    
    运行此函数可以看到两种实现的使用方式
    """
    print("=" * 80)
    print("Triton vs PyTorch 实现对比")
    print("=" * 80)
    
    # 创建测试数据
    x = torch.randn(1000, 512, device='cuda', requires_grad=True)
    weight = torch.randn(512, device='cuda', requires_grad=True)
    
    print("\n1. PyTorch 原生实现（简单版本）:")
    print("   " + "=" * 76)
    print("   def forward(x, weight):")
    print("       return (x * weight).sum(dim=-1)  # 一行代码！")
    print("\n   特点：")
    print("   ✅ 代码简洁（1行核心代码）")
    print("   ✅ 自动内存管理")
    print("   ✅ 自动并行化")
    print("   ✅ 易于理解和维护")
    
    pytorch_output = WeightedSumPyTorch.apply(x, weight)
    print(f"\n   输出形状: {pytorch_output.shape}")
    
    print("\n2. Triton 实现（复杂版本）:")
    print("   " + "=" * 76)
    print("   @triton.jit")
    print("   def weighted_sum_fwd(x_ptr, weight_ptr, output_ptr, ...):")
    print("       # 需要手动：")
    print("       # - 创建块指针（~20行）")
    print("       # - 管理内存访问（~10行）")
    print("       # - 循环处理数据块（~15行）")
    print("       # - 手动加载/存储（~5行）")
    print("       # 总计：~50+ 行代码")
    print("\n   特点：")
    print("   ✅ 性能可能更优（深度优化）")
    print("   ✅ 精确控制内存访问")
    print("   ⚠️  代码复杂（50+行）")
    print("   ⚠️  需要手动管理内存")
    print("   ⚠️  调试困难")
    
    triton_output = WeightedSumFunc.apply(x, weight)
    print(f"\n   输出形状: {triton_output.shape}")
    
    # 验证结果一致性
    if torch.allclose(pytorch_output, triton_output, atol=1e-5):
        print("\n✅ 两种实现结果一致！")
    else:
        print("\n⚠️  两种实现结果有差异（可能是数值精度问题）")
        print(f"   差异: {(pytorch_output - triton_output).abs().max().item():.2e}")
    
    print("\n" + "=" * 80)
    print("总结：")
    print("  - PyTorch: 适合大多数情况，简单高效")
    print("  - Triton:  适合性能关键路径，需要极致优化时使用")
    print("=" * 80)