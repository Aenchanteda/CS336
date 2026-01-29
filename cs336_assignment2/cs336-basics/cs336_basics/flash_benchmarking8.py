from turtle import forward
import torch,math

class flash_attention_2_no_triton(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=True):

        Q_size = 32
        K_size = 64
        #if is_causal:
        #            mask = torch.tril(torch.ones(scores.shape[-2:]))
        #            scores.masked_fill_(mask==0, float('-inf'))  
                
        #Tiled_Q = torch.split(Q, Q_size, dim=Q[-2])
        #Tiled_K = torch.split(K, K_size, dim=K[-2])
        #Tiled_V = torch.split(V, K_size, dim=V[-2])
        O = torch.zeros_like(Q)
        L = torch.zeros(*Q.shape[:-1], device=Q.device, dtype=Q.dtype)
        for i in range(0, Q.shape[-2], Q_size):
            q_end = min(i+Q_size, Q.shape[-2])
            q_tile = Q[..., i:q_end, :]
            o = torch.zeros_like(q_tile)
            l = torch.zeros(*q_tile.shape[:-1], device=Q.device, dtype=Q.dtype)
            m = torch.full((*q_tile.shape[:-1],), float('-inf'), device=Q.device, dtype=Q.dtype)
            for j in range (0, K.shape[-2], K_size):
                k_end = min(j+K_size, K.shape[-2])
                k_tile = K[...,j:k_end, :]
                v_tile = V[...,j:k_end, :]
                pre_softmax = q_tile @ k_tile.transpose(-2,-1) / math.sqrt(Q.shape[-1])
                if is_causal:
                    # Create causal mask: query position >= key position
                    q_idx = torch.arange(i, q_end, device=Q.device)[:, None]
                    k_idx = torch.arange(j, k_end, device=K.device)[None, :]
                    mask = q_idx >= k_idx
                    pre_softmax = pre_softmax.masked_fill(~mask, float('-inf'))
                m_new = torch.max(m, pre_softmax.max(dim = -1)[0])
                P = torch.exp(pre_softmax - m_new.unsqueeze(-1))
                if is_causal:
                    P = P.masked_fill(~mask, 0)
                l = torch.exp(m - m_new) * l + P.sum(dim=-1)
                o = torch.exp(m.unsqueeze(-1) - m_new.unsqueeze(-1)) *  o + P @ v_tile
                m = m_new
            o = o / l.unsqueeze(-1)
            l = m + torch.log(l)
            O[...,i:q_end, :] = o
            L[...,i:q_end] = l
        ctx.save_for_backward(Q,K,V,L,O)
        ctx.is_causal = is_causal
        return O
    
    @staticmethod
    @torch.compile(backend='inductor', dynamic=True)
    def backward(ctx, dO):
        Q, K, V, L, O = ctx.saved_tensors
        is_causal = ctx.is_causal
        scale  = math.sqrt(Q.shape[-1])
        D = torch.exp(L)
        #公式13
        S = Q @ K.transpose(-2,-1) / scale
        if is_causal:
            mask = torch.tril(torch.ones_like(S, dtype=torch.bool))
            S = S.masked_fill(~mask, float('-inf'))
        #公式14
        P = torch.exp(S - L.unsqueeze(-1))
        if is_causal:
            P = P.masked_fill(~mask, 0)
        #公式15
        dV = P.transpose(-2,-1) @ dO
        #公式16
        dP = dO @ V.transpose(-2,-1)
        #公式17
        dO_dot_O = (dO * O).sum(dim=-1, keepdim=True)
        dS = P * (dP - dO_dot_O)
        #公式18
        dQ = dS @ K / scale
        #公式19
        dK = dS.transpose(-2,-1) @ Q / scale

        return dQ, dK, dV, None




import triton, math
import triton.language as tl
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,# stride_qb是Q在batch维度上的stride，标识Q从一个batch跳到下一个batch需要经过的元素数量
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    IS_CAUSAL:tl.constexpr,
    D:tl.constexpr,
    Q_TILE_SIZE:tl.constexpr,
    K_TILE_SIZE:tl.constexpr,
):
    #grid的形状是[Q_TILE_SIZE, batch_size]
    query_tile_index = tl.program_id(0) #当前program在第0维的索引，标识当前program处理的数据范围
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(#内存地址计算：address = Q_ptr + batch_idx * stride(0) + query_idx * stride(1) + dim_idx * stride(2)
        Q_ptr + batch_index * stride_qb,#指向当前batch的起始地址。##处理不同batch下的Q数据，Q.shape=[4,128,64],batch0在地址Q_ptr+0, batch1在地址Q_ptr+8192,以此类推。
        shape = (N_QUERIES, D), #tensor的完整形状（全局）
        strides = (stride_qq, stride_qd), #每个维度上移动一个元素需要的内存步长，对应 PyTorch 的 stride(1) 和 stride(2)
        offsets = (query_tile_index * Q_TILE_SIZE, 0), #当前 tile 在完整张量中的起始位置，定位到要加载的 tile 的起始位置
        block_shape = (Q_TILE_SIZE, D), #每次加载的tile大小（局部）
        order = (1,0), #储存顺序为列优先（先遍历第 1 维，再遍历第 0 维）
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape = (N_QUERIES,D),
        strides = (stride_oq, stride_od),
        offsets = (query_tile_index * Q_TILE_SIZE, 0),
        block_shape = (Q_TILE_SIZE, D),
        order = (1,0), #储存顺序为列优先
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape = (N_QUERIES, ),
        strides = (stride_lq,),
        offsets = (query_tile_index * Q_TILE_SIZE,),
        block_shape = (Q_TILE_SIZE, ),
        order = (0,),
    )

    o = tl.zeros([Q_TILE_SIZE,D], dtype = tl.float32)
    l = tl.zeros([Q_TILE_SIZE],dtype = tl.float32)
    m = tl.full([Q_TILE_SIZE], float('-inf'), dtype = tl.float32)
    Q = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option = 'zero').to(tl.float32)
    
    # 计算当前 query tile 的全局起始位置
    q_start = query_tile_index * Q_TILE_SIZE

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        k_start = j * K_TILE_SIZE
        
        # 创建 K_block_ptr 指向当前 tile
        K_block_ptr = tl.make_block_ptr(
            K_ptr + batch_index * stride_kb,
            shape = (N_KEYS, D),
            strides = (stride_kk, stride_kd),
            offsets = (k_start, 0),
            block_shape = (K_TILE_SIZE, D),
            order = (1,0),
        )
        
        # 重新创建 V_block_ptr 指向当前 tile
        V_block_ptr = tl.make_block_ptr(
            V_ptr + batch_index * stride_vb,
            shape = (N_KEYS, D),
            strides = (stride_vk, stride_vd),
            offsets = (k_start, 0),
            block_shape = (K_TILE_SIZE, D),
            order = (1,0),
        )
        
        K = tl.load(K_block_ptr, boundary_check = (0,1), padding_option = 'zero').to(tl.float32)
        V = tl.load(V_block_ptr, boundary_check = (0,1), padding_option = 'zero').to(tl.float32)
        S = tl.dot(Q, tl.trans(K)) / scale
        
        # 应用 causal mask（如果需要）
        if IS_CAUSAL:
            # 创建 mask：query 位置 >= key 位置
            q_idx = tl.arange(0, Q_TILE_SIZE)[:, None]  # [Q_TILE_SIZE, 1]
            k_idx = tl.arange(0, K_TILE_SIZE)[None, :]  # [1, K_TILE_SIZE]
            q_global = q_start + q_idx  # [Q_TILE_SIZE, 1], q_start 是当前 tile 在序列中的起始位置
            k_global = k_start + k_idx  # [1, K_TILE_SIZE]
            mask = q_global >= k_global  # [Q_TILE_SIZE, K_TILE_SIZE]
            S = tl.where(mask, S, float('-1e6')) #mask==True保留，False设为-inf
            #tl.where(condition, x, y)，其中x是condition == True 时选择的值，y是condition == False 时选择的值
        
        s_max = tl.max(S, axis=1)  # [Q_TILE_SIZE]
        m_new = tl.maximum(m, s_max)  # [Q_TILE_SIZE]
        P = tl.exp(S - m_new[:, None])  # [Q_TILE_SIZE, K_TILE_SIZE]
        l = tl.exp(m - m_new) * l + tl.sum(P, axis=1)  # [Q_TILE_SIZE]
        o = tl.exp(m[:, None] - m_new[:, None]) * o + tl.dot(P, V)  # [Q_TILE_SIZE, D]
        m = m_new
    o = o / l[:, None]
    l = m + tl.log(l)
    tl.store(O_block_ptr, o, boundary_check = (0,1)) #将o写入O_block_ptr内存地址，(0,1)表示对第 0 维不校验、第 1 维校验
    tl.store(L_block_ptr, l, boundary_check = (0,))


class triton_kernel_flash_attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal = False):
        batch_size, N_QUERIES, D = Q.shape
        scale  = math.sqrt(D)
        N_KEYS = K.shape[-2] 
        
        # 保存原始数据类型
        original_dtype = Q.dtype
        # 如果输入是 bf16，转换为 float32 进行计算
        if original_dtype == torch.bfloat16:
            Q = Q.to(torch.float32)
            K = K.to(torch.float32)
            V = V.to(torch.float32)

        # 使用固定的 tile 大小
        Q_TILE_SIZE = 32
        K_TILE_SIZE = 64

        # 计算需要多少个 query tiles（向上取整）
        num_q_tiles = (N_QUERIES + Q_TILE_SIZE - 1) // Q_TILE_SIZE

        O = torch.zeros_like(Q)
        L = torch.zeros(*Q.shape[:-1], device=Q.device, dtype=Q.dtype)
        ctx.save_for_backward(O, L)
        flash_fwd_kernel[(num_q_tiles, batch_size)](
            Q, K, V,
            O, L,

            Q.stride(0), 
            Q.stride(1), 
            Q.stride(2), 

            K.stride(0), 
            K.stride(1), 
            K.stride(2), 

            V.stride(0), 
            V.stride(1), 
            V.stride(2), 

            O.stride(0), 
            O.stride(1), 
            O.stride(2), 

            L.stride(0), 
            L.stride(1), 
        
            N_QUERIES, 
            N_KEYS,
            scale,
            is_causal,
            D,
            Q_TILE_SIZE,
            K_TILE_SIZE,
        )

        # 如果原始类型是 bf16，转换回 bf16
        if original_dtype == torch.bfloat16:
            O = O.to(original_dtype)
            L = L.to(original_dtype)
            Q = Q.to(original_dtype)
            K = K.to(original_dtype)
            V = V.to(original_dtype)
        
        ctx.save_for_backward (Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

     
    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        
        # 保存原始数据类型
        original_dtype = Q.dtype
        # 如果输入是 bf16，转换为 float32 进行计算
        if original_dtype == torch.bfloat16:
            Q = Q.to(torch.float32)
            K = K.to(torch.float32)
            V = V.to(torch.float32)
            L = L.to(torch.float32)
            O = O.to(torch.float32)
            dO = dO.to(torch.float32)
        
        scale  = math.sqrt(Q.shape[-1])
        #公式13
        S = Q @ K.transpose(-2,-1) / scale
        if is_causal:
            mask = torch.tril(torch.ones_like(S, dtype=torch.bool))
            S = S.masked_fill(~mask, float('-inf'))
        #公式14
        P = torch.exp(S - L.unsqueeze(-1))
        if is_causal:
            P = P.masked_fill(~mask, 0)
        #公式15
        dV = P.transpose(-2,-1) @ dO
        #公式16
        dP = dO @ V.transpose(-2,-1)
        #公式17
        dO_dot_O = (dO * O).sum(dim=-1, keepdim=True)
        dS = P * (dP - dO_dot_O)
        #公式18
        dQ = dS @ K / scale
        #公式19
        dK = dS.transpose(-2,-1) @ Q / scale

        return dQ, dK, dV, None

import pandas as pd
from triton.testing import do_bench
def main():
    results = []
    batch_size = 1
    dtypes = [torch.bfloat16,torch.float32]

    # (b) Cartesian product of d_model and seq_len
    d_model = [16, 32]
    seq_len = [128, 256]
    is_causal = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_warmup = 10
    n_iterations = 50
 
    print("="*100)
    print("FlashAttention-2 Benchmarking")
    print("="*100)
    print(f"Device: {device}")
    print(f"Batch size: {batch_size} (fixed)")
    print(f"Causal masking: {is_causal} (fixed)")
    print(f"Warm-up iterations: {n_warmup}")
    print(f"Benchmark iterations: {n_iterations}")
    print("="*100)



    for i in seq_len:
        for j in d_model:
            for dtype in dtypes:
                torch.manual_seed(42)
                Q = torch.randn(batch_size, i, j, device=device, dtype=dtype, requires_grad=True)
                K = torch.randn(batch_size, i, j,  device=device, dtype=dtype, requires_grad=True)
                V = torch.randn(batch_size, i, j,  device=device, dtype=dtype, requires_grad=True)
                #warm-up stage:
                for _ in range(n_warmup):
                    for _ in range(n_iterations):
                        O = flash_attention_2_no_triton.apply(Q,K,V,True)
                        dO = torch.randn_like(O)
                        O.backward(dO, retain_graph=True)
                        Q.grad = None   
                        K.grad = None
                        V.grad = None
                print('warm-up完成，开始pytorch版本的benchmarking')
                #pytorch benchmarking stage:

                #forward pass:
                forward_time_latency = []
                for _ in range(n_iterations):
                    forward_latency = do_bench(lambda:flash_attention_2_no_triton.apply(Q,K,V,is_causal))
                    forward_time_latency.append(forward_latency)
                forward_meantime_latency = torch.tensor(forward_time_latency).mean().item()

                #backward pass:
                backward_time_latency = []
                for _ in range(n_iterations):
                    O = flash_attention_2_no_triton.apply(Q,K,V,True)
                    dO = torch.randn_like(O)
                    backward_latency = do_bench(lambda:O.backward(dO, retain_graph=True))
                    backward_time_latency.append(backward_latency)
                    Q.grad = None
                    K.grad = None
                    V.grad = None
                backward_meantime_latency = torch.tensor(backward_time_latency).mean().item()

                #end-2-end (forward + backward)
                e2e_time_latency = []
                for _ in range(n_iterations):
                    def forward_backward():
                        O = flash_attention_2_no_triton.apply(Q,K,V,is_causal)
                        dO = torch.randn_like(O)
                        O.backward(dO,retain_graph=True)
                        return O
                    latency = do_bench(forward_backward,warmup=0)
                    e2e_time_latency.append(latency)
                    Q.grad = None
                    K.grad = None
                    V.grad = None
                
                e2e_meantime_latency = torch.tensor(e2e_time_latency).mean().item()

                #benchmarking triton implementation 
                print('开始triton版本的benchmarking')

                #triton forward pass
                forward_time_triton = []
                for _ in range(n_iterations):
                    latency = do_bench(lambda:triton_kernel_flash_attention.apply(Q,K,V,is_causal),warmup=0)
                    forward_time_triton.append(latency)
                forward_meantime_triton = torch.tensor(forward_time_triton).mean().item()

                #测量backward pass
                backward_time_triton = [] 
                for _ in range(n_iterations):
                    O = triton_kernel_flash_attention.apply(Q,K,V,is_causal)
                    dO = torch.randn_like(O)
                    latency = do_bench(lambda:O.backward(dO,retain_graph=True), warmup=0)
                    backward_time_triton.append(latency)
                    Q.grad = None
                    K.grad = None
                    V.grad = None
                backward_meantime_triton = torch.tensor(backward_time_triton).mean().item()
                
                #end-2-end(forward+backward)
                e2e_time_triton = []
                for _ in range(n_iterations):
                    def forward_backward():
                        O =triton_kernel_flash_attention.apply(Q,K,V,is_causal)
                        dO = torch.randn_like(O)
                        O.backward(dO,retain_graph=True)
                        return O
                    latency = do_bench(forward_backward,warmup=0)
                    e2e_time_triton.append(latency)
                    Q.grad = None
                    K.grad = None
                    V.grad = None
                e2e_meantime_triton = torch.tensor(e2e_time_triton).mean().item()
        

                results.append({'seq_len':i,
                'd_model': j,
                'dtype': str(dtype).split('.')[-1],  # 'bfloat16' or 'float32'
                'forward_meantime_latency': forward_meantime_latency,
                'backward_meantime_latency': backward_meantime_latency,
                'e2e_meantime_latency': e2e_meantime_latency,
                'forward_meantime_triton': forward_meantime_triton,
                'backward_meantime_triton': backward_meantime_triton,
                'e2e_meantime_triton': e2e_meantime_triton})
                
    df = pd.DataFrame(results)
    #print
    print('\n' + '='*150)
    print('summary table')
    print('='*150)

    #Group by dtype for better readability
    for dtype in ['bfloat16','float32']:
        df_dtype = df[df['dtype']==dtype]
        print(f"\n{'='*150}")
        print(f"Precision: {dtype.upper()}")
        print(f"{'='*150}")
        print(f"{'Seq Len':<10} {'d_model':<10} {'PyTorch Forward':<20} {'PyTorch Backward':<20} {'PyTorch E2E':<20} "
              f"{'Triton Forward':<20} {'Triton Backward':<20} {'Triton E2E':<20}")
        print("-"*150)

        for _, row in df_dtype.iterrows():#df.iterrows()逐行遍历，返回迭代器，每次迭代生成一个（行索引，列索引）的元组
            #第一个元素是行的索引（index）；第二个元素是该行的所有数据（Pandas Series 对象，可通过列名取值）
            print(f"{row['seq_len']:<10} {row['d_model']:<10} "
                  f"{row['forward_meantime_latency']:<20.4f} {row['backward_meantime_latency']:<20.4f} {row['e2e_meantime_latency']:<20.4f} "
                  f"{row['forward_meantime_triton']:<20.4f} "  # 修正：直接格式化
                  f"{row['backward_meantime_triton']:<20.4f} "
                  f"{row['e2e_meantime_triton']:<20.4f}")
    return 

if __name__=='__main__':
    results_df = main()



                
