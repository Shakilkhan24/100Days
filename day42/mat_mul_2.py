import torch

import triton
import triton.language as tl

@triton.jit
def get_1d_offest(size,n_prev_chunks):
    return n_prev_chunks * size + tl.arange(0,size)

@triton.jit
def get_2d_offest(offs_0,offs_1,stride_0,stride_1):
    return tl.expand_dims(offs_0,1)*stride_0 + tl.expand_dims(offs_1,0)*stride_1

@triton.jit
def get_1d_mask(offs,max):
    return offs<max

@triton.jit
def get_2d_mask(offs_0,offs_1,max_0,max_1):
    return (tl.expand_dims(offs_0,1) < max_0) & (tl.expand_dims(offs_1,0) < max_1)


@triton.jit
def naive_matmul_k(
    a_ptr,b_ptr,c_ptr,
    m,n,k,
    stride_am,stride_ak,
    stride_bk,stride_bn,
    stride_cm,stride_cn,
    bm:tl.constexpr,
    bn:tl.constexpr,
    bk:tl.constexpr,
):
    pid_m,pid_n = tl.program_id(0), tl.program_id(1)
    
    rm = get_1d_offest(size=bm,n_prev_chunks=pid_m)
    rn = get_1d_offest(size=bn,n_prev_chunks=pid_n)
    rk = get_1d_offest(size=bk,n_prev_chunks=0)
    
    offs_a = a_ptr + get_2d_offest(rm,rk,stride_am,stride_ak)
    offs_b = b_ptr + get_2d_offest(rk,rn,stride_bk,stride_bn)
    
    acc = tl.zeros((bm,bn),dtype=tl.float32)
    for _ in range(0,k,bk):
        
        a = tl.load(offs_a)
        b = tl.load(offs_b)
        
        acc += tl.dot(a,b)
        
        offs_a += bk * stride_ak
        offs_b += bk * stride_bk
    
    c = c_ptr + get_2d_offest(rn,rn,stride_cm,stride_cn)
    mask = get_2d_mask(rm,rn,m,n)
    tl.store(c,acc,mask)
    
def matmul(a,b,matmul_k_fn,bs=16,group_sz=None):
    (m,k) , (_,n) = a.shape, b.shape
    c = torch.empty((m,n),device=a.device,dtype=torch.float16)
    
    grid = lambda meta : (triton.cdiv(n,meta['bm']),triton.cdiv(n,meta['bn']))
    
    matmul_k_fn[grid](
        a , b, c,
        m, n ,k,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0),c.stride(1),
        bn=bs,bm=bs,bk=bs
    )
    return c

     
a = torch.ones((3,4),dtype=torch.float32,device="cuda")
b = torch.ones((4,5),dtype=torch.float32,device="cuda")

print(matmul(a,b,naive_matmul_k))