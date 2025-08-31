import triton
import triton.language as tl
import torch
import math
import numpy as np

@triton.jit 
def flash_fwd_kernel(Q_ptr, K_ptr, V_ptr,O_ptr, L_ptr,
                     stride_qb, stride_qq, stride_qd,
                     stride_kb, stride_kk, stride_kd,
                     stride_vb, stride_vk, stride_vd,
                     stride_ob, stride_oq, stride_od,
                     stride_lb, stride_lq,N_QUERIES, N_KEYS, scale,
                     D: tl.constexpr, Q_TILE_SIZE: tl.constexpr,K_TILE_SIZE: tl.constexpr,is_causal: tl.constexpr):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    
    
    Q_block_ptr = tl.make_block_ptr(Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),block_shape=(Q_TILE_SIZE, D),order=(1, 0),)
    Qi=tl.load(Q_block_ptr)
    
    #声明需要输出的指针
    O_block_ptr=tl.make_block_ptr(O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),block_shape=(Q_TILE_SIZE, D),order=(1, 0),)
    
    L_block_ptr=tl.make_block_ptr(L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),strides=(stride_lq, ),
        offsets=(query_tile_index * Q_TILE_SIZE,),block_shape=(Q_TILE_SIZE,),order=(0,),)
    
   
    #打印输出的规范: tl.device_print("Qi",Qi)
    O_1=tl.zeros((Q_TILE_SIZE,D),dtype=tl.float32)
    l_1=tl.zeros((Q_TILE_SIZE,),dtype=tl.float32)
    m_1=tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)
    
    
    K_block_ptr=tl.make_block_ptr(K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),strides=(stride_kk, stride_kd),
        offsets=(0, 0),block_shape=(K_TILE_SIZE, D),order=(1, 0),)
    
    V_block_ptr=tl.make_block_ptr(V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),strides=(stride_vk, stride_vd),
        offsets=(0, 0),block_shape=(K_TILE_SIZE, D),order=(1, 0),)
    for i in range(0,tl.cdiv(N_KEYS,K_TILE_SIZE)):
        Kj=tl.load(K_block_ptr)
        Vj=tl.load(V_block_ptr)
        
        
        if is_causal==False:
            Sij=tl.dot(Qi,tl.trans(Kj))*scale
        else:
            Sij=tl.dot(Qi,tl.trans(Kj))*scale
            mask=tl.zeros((Q_TILE_SIZE,K_TILE_SIZE),dtype=tl.float32)
            
            #考虑构建行和列的索引位置 
            #row=tl.arange(query_tile_index*Q_TILE_SIZE ,query_tile_index*Q_TILE_SIZE+Q_TILE_SIZE )
            a = query_tile_index * Q_TILE_SIZE+tl.arange(0, Q_TILE_SIZE)
            b=i*K_TILE_SIZE+tl.arange(0, K_TILE_SIZE)
            
            a=a.reshape((Q_TILE_SIZE,1)).broadcast_to((Q_TILE_SIZE,K_TILE_SIZE))
            b=b.reshape((1,K_TILE_SIZE)).broadcast_to((Q_TILE_SIZE,K_TILE_SIZE))
            
            mask = tl.where(a>=b, 0.0, -1e6)
            
            Sij+=mask
            
            
            
        mj=tl.maximum(m_1,tl.max(Sij,axis=1))
        Pij=tl.exp(Sij-mj.reshape((Q_TILE_SIZE,1)).broadcast_to((Q_TILE_SIZE,K_TILE_SIZE)))
        lj=tl.exp(m_1-mj)*l_1+tl.sum(Pij,axis=-1)
        #triton中没有diag_embed 所以此处用广播+Hadamard乘积代替
        Oj=tl.exp(m_1-mj).reshape((Q_TILE_SIZE,1)).broadcast_to((Q_TILE_SIZE,D))*O_1+tl.dot(Pij.to(Vj.dtype),Vj)
        
        O_1=Oj
        l_1=lj
        m_1=mj
    
        K_block_ptr=K_block_ptr.advance((K_TILE_SIZE,0))
        V_block_ptr=V_block_ptr.advance((K_TILE_SIZE,0))
        
    #triton中没有diag_embed 所以此处用广播+Hadamard乘积代替
    Oi=(1.0/l_1).reshape((Q_TILE_SIZE,1)).broadcast_to((Q_TILE_SIZE,D))*O_1

    Li=m_1+tl.log(l_1)
    tl.store(O_block_ptr, value=Oi)
    tl.store(L_block_ptr, value=Li)
    
    
@triton.jit 
def flash_bwd_kv_kernel(Q_ptr, K_ptr, V_ptr,O_ptr, L_ptr, dO_ptr,D_ptr,
                     stride_dob,stride_doq,stride_dod,
                     stride_db,stride_dq,
                     stride_qb, stride_qq, stride_qd,
                     stride_kb, stride_kk, stride_kd,
                     stride_vb, stride_vk, stride_vd,
                     stride_ob, stride_oq, stride_od,
                     stride_lb, stride_lq,N_QUERIES, N_KEYS, scale,
                     D: tl.constexpr, Q_TILE_SIZE: tl.constexpr,K_TILE_SIZE: tl.constexpr,is_causal: tl.constexpr,
                     dK_ptr,dV_ptr,
                     stride_dkb, stride_dkk, stride_dkd,
                     stride_dvb, stride_dvk, stride_dvd):
    #这里的启动网格应该是(Tk,batch_size)
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    
    #外层循环需要的量, offset跟着启动网格的位置key_tile_index, 要有相应的offset
    K_block_ptr=tl.make_block_ptr(K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),strides=(stride_kk, stride_kd),
        offsets=(key_tile_index*K_TILE_SIZE, 0),block_shape=(K_TILE_SIZE, D),order=(1, 0),)
    
    V_block_ptr=tl.make_block_ptr(V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),strides=(stride_vk, stride_vd),
        offsets=(key_tile_index*K_TILE_SIZE, 0),block_shape=(K_TILE_SIZE, D),order=(1, 0),)
    
    #内层循环需要的量, offset都是0
    Q_block_ptr = tl.make_block_ptr(Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),strides=(stride_qq, stride_qd),
        offsets=(0, 0),block_shape=(Q_TILE_SIZE, D),order=(1, 0),)
    
    O_block_ptr=tl.make_block_ptr(O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),strides=(stride_oq, stride_od),
        offsets=(0, 0),block_shape=(Q_TILE_SIZE, D),order=(1, 0),)
    
    dO_block_ptr=tl.make_block_ptr(dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),strides=(stride_doq, stride_dod),
        offsets=(0, 0),block_shape=(Q_TILE_SIZE, D),order=(1, 0),)
    
    D_block_ptr=tl.make_block_ptr(D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),strides=(stride_dq,),
        offsets=(0,),block_shape=(Q_TILE_SIZE,),order=(0,),)
    
    L_block_ptr=tl.make_block_ptr(L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),strides=(stride_lq,),
        offsets=(0,),block_shape=(Q_TILE_SIZE,),order=(0,),)
    
    Kj=tl.load(K_block_ptr)
    Vj=tl.load(V_block_ptr)
    
    dK_block_ptr=tl.make_block_ptr(dK_ptr + batch_index * stride_dkb,
        shape=(N_KEYS, D),strides=(stride_dkk, stride_dkd),
        offsets=(key_tile_index*K_TILE_SIZE, 0),block_shape=(K_TILE_SIZE, D),order=(1, 0),)
    dV_block_ptr=tl.make_block_ptr(dV_ptr + batch_index * stride_dvb,
        shape=(N_KEYS, D),strides=(stride_dvk, stride_dvd),
        offsets=(key_tile_index*K_TILE_SIZE, 0),block_shape=(K_TILE_SIZE, D),order=(1, 0),)
    dKj=tl.zeros_like(Kj)
    dVj=tl.zeros_like(Vj)
    for i in range(0,tl.cdiv(N_QUERIES,Q_TILE_SIZE)):
        Qi=tl.load(Q_block_ptr)
        Oi=tl.load(O_block_ptr)
        dOi=tl.load(dO_block_ptr)
        Li=tl.load(L_block_ptr)
        Di=tl.load(D_block_ptr)
        
        if is_causal==False:
            Sij=tl.dot(Qi,tl.trans(Kj))*scale
        else:
            Sij=tl.dot(Qi,tl.trans(Kj))*scale
            a = i * Q_TILE_SIZE+tl.arange(0, Q_TILE_SIZE)
            b=key_tile_index*K_TILE_SIZE+tl.arange(0, K_TILE_SIZE)
            
            a=a.reshape((Q_TILE_SIZE,1)).broadcast_to((Q_TILE_SIZE,K_TILE_SIZE))
            b=b.reshape((1,K_TILE_SIZE)).broadcast_to((Q_TILE_SIZE,K_TILE_SIZE))
            
            mask = tl.where(a>=b, 0.0, -1e6)
            Sij+=mask
            
        Pij=tl.exp(Sij-Li.reshape((Q_TILE_SIZE,1)).broadcast_to((Q_TILE_SIZE,K_TILE_SIZE)))
        
        dVj+=tl.dot(tl.trans(Pij),dOi)
        
        dPij=tl.dot(dOi,tl.trans(Vj))
        dSij=Pij*(dPij-Di.reshape((Q_TILE_SIZE,1)).broadcast_to((Q_TILE_SIZE,K_TILE_SIZE)))*scale
        
        
        dKj+=tl.dot(tl.trans(dSij),Qi)
        
        Q_block_ptr=Q_block_ptr.advance((Q_TILE_SIZE,0))
        O_block_ptr=O_block_ptr.advance((Q_TILE_SIZE,0))
        dO_block_ptr=dO_block_ptr.advance((Q_TILE_SIZE,0))
        L_block_ptr=L_block_ptr.advance((Q_TILE_SIZE,))
        D_block_ptr=D_block_ptr.advance((Q_TILE_SIZE,))
    tl.store(dK_block_ptr,dKj)
    tl.store(dV_block_ptr,dVj)
        
@triton.jit 
def flash_bwd_q_kernel(Q_ptr, K_ptr, V_ptr,O_ptr, L_ptr, dO_ptr,D_ptr,
                     stride_dob,stride_doq,stride_dod,
                     stride_db,stride_dq,
                     stride_qb, stride_qq, stride_qd,
                     stride_kb, stride_kk, stride_kd,
                     stride_vb, stride_vk, stride_vd,
                     stride_ob, stride_oq, stride_od,
                     stride_lb, stride_lq,N_QUERIES, N_KEYS, scale,
                     D: tl.constexpr, Q_TILE_SIZE: tl.constexpr,K_TILE_SIZE: tl.constexpr,is_causal: tl.constexpr,
                     dQ_ptr,
                     stride_dqb,stride_dqq,stride_dqd):
    
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)
        
        dQ_block_ptr=tl.make_block_ptr(dQ_ptr + batch_index * stride_dqb,
            shape=(N_QUERIES, D),strides=(stride_dqq, stride_dqd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),block_shape=(Q_TILE_SIZE, D),order=(1, 0),)      #Bq x D
        Q_block_ptr = tl.make_block_ptr(Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D),strides=(stride_qq, stride_qd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),block_shape=(Q_TILE_SIZE, D),order=(1, 0),)   #Bq x D
        
        Qi=tl.load(Q_block_ptr)
        
        dO_block_ptr=tl.make_block_ptr(dO_ptr+ batch_index *stride_dob , 
                    shape=(N_QUERIES,D), strides=(stride_doq,stride_dod), 
                    offsets=(query_tile_index * Q_TILE_SIZE, 0), block_shape=(Q_TILE_SIZE, D),order=(1, 0),)
        
        dOi=tl.load(dO_block_ptr)
        
        L_block_ptr=tl.make_block_ptr(L_ptr+batch_index *stride_lb, 
                                      shape=(N_QUERIES,), strides=(stride_lq,), 
                                      offsets=(query_tile_index * Q_TILE_SIZE,), block_shape=(Q_TILE_SIZE,), order=(0,))
        
        Li=tl.load(L_block_ptr)
        
        D_block_ptr=tl.make_block_ptr(D_ptr+batch_index *stride_db, 
                                      shape=(N_QUERIES,), strides=(stride_dq,), 
                                      offsets=(query_tile_index * Q_TILE_SIZE,), block_shape=(Q_TILE_SIZE,), order=(0,))
        
        Di=tl.load(D_block_ptr)
        
        #内圈循环的变量 不需要offset
        K_block_ptr=tl.make_block_ptr(K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),strides=(stride_kk, stride_kd),
            offsets=(0, 0),block_shape=(K_TILE_SIZE, D),order=(1, 0),)
        
        V_block_ptr=tl.make_block_ptr(V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),strides=(stride_vk, stride_vd),
            offsets=(0, 0),block_shape=(K_TILE_SIZE, D),order=(1, 0),)

        dQ=tl.zeros_like(Qi)
        for j in range(0,tl.cdiv(N_KEYS,K_TILE_SIZE)):
            Kj=tl.load(K_block_ptr)
            Vj=tl.load(V_block_ptr)
            if is_causal==False:
                Sij=tl.dot(Qi,tl.trans(Kj))*scale
            else:
                Sij=tl.dot(Qi,tl.trans(Kj))*scale
                a = query_tile_index * Q_TILE_SIZE+tl.arange(0, Q_TILE_SIZE)
                b=j*K_TILE_SIZE+tl.arange(0, K_TILE_SIZE)
                
                a=a.reshape((Q_TILE_SIZE,1)).broadcast_to((Q_TILE_SIZE,K_TILE_SIZE))
                b=b.reshape((1,K_TILE_SIZE)).broadcast_to((Q_TILE_SIZE,K_TILE_SIZE))
                
                mask = tl.where(a>=b, 0.0, -1e6)
                
                Sij+=mask
                
            Pij=tl.exp(Sij-Li.reshape((Q_TILE_SIZE,1)).broadcast_to((Q_TILE_SIZE,K_TILE_SIZE)))
            dPij=tl.dot(dOi,tl.trans(Vj))
            
            dSij=Pij*(dPij-Di.reshape((Q_TILE_SIZE,1)).broadcast_to((Q_TILE_SIZE,K_TILE_SIZE)))
            
            dQ+=tl.dot(dSij,Kj)*scale
            
            K_block_ptr=K_block_ptr.advance((K_TILE_SIZE,0))
            V_block_ptr=V_block_ptr.advance((K_TILE_SIZE,0))
        
        tl.store(dQ_block_ptr, dQ)
    
class Attention_Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx,Q,K,V,is_causal=False):
        batch_szie=K.shape[0]
        d=K.shape[-1]
        Nk=K.shape[-2]
        Nq=Q.shape[-2]
        
        
        
        #初始化result Tensor:O和L
        O=torch.zeros(size=(batch_szie,Nq,d),device=Q.device).to(dtype=torch.float32)
        L=torch.zeros(size=(batch_szie,Nq),device=Q.device)
        
        ctx.Q_TILE_SIZE =16
        ctx.K_TILE_SIZE=16
        ctx.is_causal=is_causal
        
        Tq=Nq//ctx.Q_TILE_SIZE
        
        flash_fwd_kernel[(Tq,batch_szie)](Q,K,V,O,L,Q.stride(0),Q.stride(1),Q.stride(2),
                                          K.stride(0),K.stride(1),K.stride(2),V.stride(0),V.stride(1),V.stride(2),
                                          O.stride(0),O.stride(1),O.stride(2),L.stride(0),L.stride(1),Nq,Nk,1.0/math.sqrt(d),
                                          d,ctx.Q_TILE_SIZE,ctx.K_TILE_SIZE,ctx.is_causal)
        ctx.save_for_backward(Q,O,K,V,L)
        return O
    
    def backward(ctx, dO):
        
        Q,O,K,V,L=ctx.saved_tensors
        is_causal=ctx.is_causal
        
        batch_szie=K.shape[0]
        d=K.shape[-1]
        Nk=K.shape[-2]
        Nq=Q.shape[-2]
        D=torch.sum(dO*O,dim=-1)
        
        Tk=Nk//ctx.K_TILE_SIZE
        Tq=Nq//ctx.Q_TILE_SIZE
        
        dQ=torch.zeros_like(Q)
        dK=torch.zeros_like(K)
        dV=torch.zeros_like(V)
        
        flash_bwd_kv_kernel[Tk,batch_szie](Q,K,V,O,L,dO, D,dO.stride(0),dO.stride(1),dO.stride(2),
                                        D.stride(0),D.stride(1),
                                        Q.stride(0),Q.stride(1),Q.stride(2),
                                          K.stride(0),K.stride(1),K.stride(2),V.stride(0),V.stride(1),V.stride(2),
                                          O.stride(0),O.stride(1),O.stride(2),L.stride(0),L.stride(1),Nq,Nk,1.0/math.sqrt(d),
                                          d,ctx.Q_TILE_SIZE,ctx.K_TILE_SIZE,ctx.is_causal,
                                          dK,dV,
                                          dK.stride(0),dK.stride(1),dK.stride(2),
                                          dV.stride(0),dV.stride(1),dV.stride(2))
        
        flash_bwd_q_kernel[Tq,batch_szie](Q,K,V,O,L,dO, D,dO.stride(0),dO.stride(1),dO.stride(2),
                                        D.stride(0),D.stride(1),
                                        Q.stride(0),Q.stride(1),Q.stride(2),
                                          K.stride(0),K.stride(1),K.stride(2),V.stride(0),V.stride(1),V.stride(2),
                                          O.stride(0),O.stride(1),O.stride(2),L.stride(0),L.stride(1),Nq,Nk,1.0/math.sqrt(d),
                                          d,ctx.Q_TILE_SIZE,ctx.K_TILE_SIZE,ctx.is_causal,
                                          dQ,dQ.stride(0),dQ.stride(1),dQ.stride(2))
        return dQ,dK,dV,None
        
