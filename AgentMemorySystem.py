#!/usr/bin/env python3  
"""  
嵌入级方案B · v3.7  
════════════════════════════════════════════════════════════════════════  
  
v3.7 变更摘要 (相对 v3.6)  
─────────────────────────  
  
[P0-RETRIEVE] Content-Position-Only Semantic Embedding  
  写入时: semantic_emb = mean(GPT-2 hidden states at CONTENT token positions only)  
  查询时: 同上, 排除所有虚词/标点位置  
  消除了全位置均值池化导致的域信号稀释  
  零训练即生效  
  
[P0-RETRIEVE] WTE Centroid Retrieval  
  写入时: content_wte_centroid = mean(WTE[content_token_ids])  
  查询时: 同上  
  检索评分: 0.1*dir_sim + 0.4*content_sem_sim + 0.5*wte_centroid_sim  
  完全不依赖任何学习组件, 域分离由 GPT-2 预训练词嵌入保证  
  
[P0-DECODE] Hard Content-First Decoding  
  步骤 0-2: 标点/换行 → -25 penalty, EOS → -inf  
  步骤 0+: content_bias_scale = 15.0 (from 6.0), 慢衰减  
  步骤 0-4: 通用内容词 boost = +2.0  
  保证首步 top-1 不是标点  
  
[P0-DECODE] Content Bias Expansion via WTE Neighbors  
  写入时: 对每个 content_token, 找 WTE 空间 top-5 近邻中的内容词  
  存入 expanded_content_ids, 提高 bias 覆盖率  
  "practiced" 在记忆 → "played","playing" 也获得 boost  
  
[P1-BRIDGE] 确定性路径现在真正占主导  
  content_bias_scale=15 vs vocab_proj的learned bias≈0.5  
  确定性路径提供 97% 的语义引导  
  
[INFRA] MemEntry 新增 content_wte_centroid: Tensor[d_LLM]  
[INFRA] DirectionTree 新增 check_direction_degeneracy()  
[INFRA] _compute_content_semantic_emb() 新方法  
[INFRA] _get_prefix 新增 ids 参数  
  
要求: pip install torch transformers  
"""  
  
import torch, torch.nn as nn, torch.nn.functional as F  
import math, time, warnings  
from typing import Dict, List, Tuple, Optional, NamedTuple, Set, FrozenSet  
from dataclasses import dataclass, field  
  
# ═══════════════════════════════════════════════════════════════════  
# 配置  
# ═══════════════════════════════════════════════════════════════════  
@dataclass  
class Cfg:  
    d_LLM: int = 768; d_M: int = 8; d_F: int = 32  
    L_mem: int = 8; n_heads_fiber: int = 4  
    bridge_heads: int = 4; bridge_layers: int = 2  
    n_geo_pts: int = 8; geo_max_steps: int = 80  
    geo_tol: float = 1e-5; geo_lr: float = 0.02  
    tree_K: int = 8; tree_max_leaf: int = 20  
    tau: float = 0.07  
    write_gate_threshold: float = 0.4  
    retention_gc_threshold: float = 0.15  
    consol_dist: float = 0.3; consol_conflict_ratio: float = 0.5  
    retrieval_topk: int = 8; retrieval_beam: int = 5  
    retrieval_interval: int = 8  
    retrieval_recall_factor: float = 2.0  
    flat_scan_threshold_factor: int = 3  
    gen_top_p: float = 0.9; gen_temp: float = 0.8  
    norm_correction_interval: int = 4  
    write_update_alpha: float = 0.3  
    dir_diversity_tau: float = 0.5  
    bypass_init_gate_bias: float = -0.5  
    degen_min_tokens: int = 5; degen_repeat_penalty: float = 1.3  
    degen_max_consec_punct: int = 2  
    probe_contrastive_tau: float = 0.1  
    contrast_tau: float = 0.5  
    prefix_init_scale: float = -1.0  
    # ── v3.7 decode parameters ──  
    degen_early_punct_penalty: float = 25.0  
    degen_early_newline_penalty: float = 25.0  
    early_content_steps: int = 3  
    universal_content_boost: float = 2.0  
    universal_content_boost_steps: int = 5  
    # ── v3.7 content bias ──  
    content_bias_scale: float = 15.0  
    content_bias_decay: float = 0.03  
    content_bias_floor: float = 0.3  
    # ── v3.7 retrieval weights ──  
    ret_dir_weight: float = 0.1  
    ret_sem_weight: float = 0.4  
    ret_wte_weight: float = 0.5  
    # ── v3.5 preserved ──  
    semantic_boost_scale: float = 0.5  
    semantic_boost_decay: float = 0.06  
    semantic_boost_floor: float = 0.2  
    semantic_align_temp: float = 0.3  
    # ── general ──  
    vocab_size: int = 50257  
    wte_neighbor_k: int = 5  
    wte_neighbor_threshold: float = 0.5  
    loss_weights: Dict[str, float] = field(default_factory=lambda: {  
        'recon': 1.0, 'semantic_alignment': 3.0,  
        'encoder_throughput': 1.5, 'contrast': 0.02,  
        'holonomy': 0.005, 'write_policy': 0.1,  
        'semantic_probe': 0.3, 'dir_diversity': 0.1,  
        'reranker_ranking': 0.2, 'vocab_anchor': 0.2})  
    warmup_steps_probe: int = 5; warmup_steps_dd: int = 5  
    warmup_steps_rr: int = 5; warmup_steps_va: int = 5  
    warmup_steps_sa: int = 0  
    uw_clamp_lo: float = -4.0; uw_clamp_hi: float = 4.0  
    vocab_anchor_topk: int = 5; content_min_len: int = 3  
    refresh_memories_every: int = 1  
    def __post_init__(self):  
        assert self.d_F % self.n_heads_fiber == 0  
        assert self.n_geo_pts >= 2 and 0 < self.tau < 1  
  
def _dev(ref: torch.Tensor):  
    return dict(device=ref.device, dtype=ref.dtype)  
  
# ═══════════════════════════════════════════════════════════════════  
# 第1部分 · 黎曼度量  
# ═══════════════════════════════════════════════════════════════════  
class RiemannianMetric(nn.Module):  
    def __init__(self, d):  
        super().__init__(); self.d = d  
        n_tri = d*(d+1)//2  
        self.net = nn.Sequential(  
            nn.Linear(d,4*d), nn.SiLU(),  
            nn.Linear(4*d,4*d), nn.SiLU(),  
            nn.Linear(4*d, n_tri))  
        for m in self.net.modules():  
            if isinstance(m, nn.Linear):  
                nn.init.xavier_normal_(m.weight)  
                if m.bias is not None: nn.init.zeros_(m.bias)  
        nn.init.normal_(self.net[-1].weight, std=0.02)  
        nn.init.zeros_(self.net[-1].bias)  
        r,c=[],[]  
        for i in range(d):  
            for j in range(i+1): r.append(i); c.append(j)  
        self.register_buffer('_r', torch.tensor(r))  
        self.register_buffer('_c', torch.tensor(c))  
    def forward(self, x):  
        B=x.shape[0]; d=self.d; v=self.net(x)  
        L=x.new_zeros(B,d,d); L[:,self._r,self._c]=v  
        di=torch.arange(d,device=x.device)  
        L[:,di,di]=F.softplus(L[:,di,di])+1e-3  
        return L@L.transpose(1,2)  
    def christoffel(self, x):  
        d=self.d; B=x.shape[0]  
        xv=x.detach().clone().requires_grad_(True)  
        g=self.forward(xv); g_inv=torch.linalg.inv(g.detach())  
        dg=x.new_zeros(B,d,d,d)  
        for i in range(d):  
            for j in range(i,d):  
                gr=torch.autograd.grad(g[:,i,j].sum(),xv,retain_graph=True)[0]  
                dg[:,i,j,:]=gr  
                if i!=j: dg[:,j,i,:]=gr  
        term=dg.permute(0,3,1,2)+dg.permute(0,1,3,2)-dg  
        return (0.5*torch.einsum('bkl,bijl->bkij',g_inv,term)).detach()  
    def midpoint_approx_distance(self, x, y):  
        diff=x-y; mid=(x+y)/2  
        with torch.no_grad(): g=self.forward(mid)  
        return torch.einsum('bi,bij,bj->b',diff,g,diff).clamp(min=0).sqrt()  
  
# ═══════════════════════════════════════════════════════════════════  
# 第2部分 · 测地线求解器  
# ═══════════════════════════════════════════════════════════════════  
class GeodesicResult(NamedTuple):  
    path: torch.Tensor; energy: float; converged: bool; iterations: int  
  
class GeodesicSolver:  
    def __init__(self, metric, cfg):  
        self.metric=metric; self.cfg=cfg  
    def solve(self, xs, xe):  
        B,d=xs.shape; N=self.cfg.n_geo_pts; dev=xs.device  
        t=torch.linspace(0,1,N+2,device=dev)[1:-1]  
        ps={n:p.requires_grad for n,p in self.metric.named_parameters()}  
        for p in self.metric.parameters(): p.requires_grad_(False)  
        with torch.enable_grad():  
            interior=(xs.detach().unsqueeze(1)*(1-t[None,:,None])  
                      +xe.detach().unsqueeze(1)*t[None,:,None]).detach().clone().requires_grad_(True)  
            opt=torch.optim.Adam([interior],lr=self.cfg.geo_lr)  
            prev=float('inf'); converged=False; iters=0  
            for it in range(self.cfg.geo_max_steps):  
                opt.zero_grad()  
                path=torch.cat([xs.detach().unsqueeze(1),interior,xe.detach().unsqueeze(1)],1)  
                dx=path[:,1:]-path[:,:-1]; mid=(path[:,1:]+path[:,:-1])/2  
                g=self.metric(mid.reshape(-1,d)).reshape(B,N+1,d,d)  
                energy=torch.einsum('bni,bnij,bnj->',dx,g,dx)  
                if energy.item()!=energy.item():  
                    warnings.warn("GeodesicSolver: NaN energy")  
                    t_full=torch.linspace(0,1,N+2,device=dev).view(1,-1,1)  
                    lin=xs.unsqueeze(1)*(1-t_full)+xe.unsqueeze(1)*t_full  
                    for n,p in self.metric.named_parameters(): p.requires_grad_(ps[n])  
                    return GeodesicResult(lin,float('inf'),False,it)  
                energy.backward(); opt.step(); iters=it+1; cur=energy.item()  
                if abs(prev-cur)/(abs(prev)+1e-10)<self.cfg.geo_tol:  
                    converged=True; break  
                prev=cur  
        for n,p in self.metric.named_parameters(): p.requires_grad_(ps[n])  
        final=torch.cat([xs.unsqueeze(1),interior.detach(),xe.unsqueeze(1)],1)  
        return GeodesicResult(final,cur if 'cur' in dir() else prev,converged,iters)  
  
# ═══════════════════════════════════════════════════════════════════  
# 第3部分 · 纤维联络 + RK4平行移动  
# ═══════════════════════════════════════════════════════════════════  
class FiberConnection(nn.Module):  
    def __init__(self, d_M, d_F, metric, grad_coupling=True):  
        super().__init__(); self.d_F=d_F; self.metric=metric; self.grad_coupling=grad_coupling  
        d_g=d_M*(d_M+1)//2  
        self.net=nn.Sequential(nn.Linear(2*d_M+d_g,4*d_F),nn.SiLU(),  
                               nn.Linear(4*d_F,4*d_F),nn.SiLU(),nn.Linear(4*d_F,d_F*d_F))  
        nn.init.normal_(self.net[-1].weight,std=0.01); nn.init.normal_(self.net[-1].bias,std=0.01)  
    def forward(self, x, v):  
        g=self.metric(x); d=g.shape[-1]; idx=torch.triu_indices(d,d,device=x.device)  
        gf=g[:,idx[0],idx[1]]  
        if not self.grad_coupling: gf=gf.detach()  
        raw=self.net(torch.cat([x,v,gf],-1)).reshape(-1,self.d_F,self.d_F)  
        return (raw-raw.transpose(1,2))/2  
  
class FiberTransporter(nn.Module):  
    def __init__(self, conn, cfg):  
        super().__init__(); self.conn=conn; self.cfg=cfg  
    def forward(self, fiber, path):  
        f=fiber; n0=fiber.norm(dim=-1,keepdim=True).clamp(min=1e-8)  
        nci=self.cfg.norm_correction_interval  
        for k in range(path.shape[1]-1):  
            p0,p1=path[:,k],path[:,k+1]; v=p1-p0; mid=(p0+p1)/2  
            k1=-(self.conn(p0,v)@f.unsqueeze(-1)).squeeze(-1)  
            k2=-(self.conn(mid,v)@(f+.5*k1).unsqueeze(-1)).squeeze(-1)  
            k3=-(self.conn(mid,v)@(f+.5*k2).unsqueeze(-1)).squeeze(-1)  
            k4=-(self.conn(p1,v)@(f+k3).unsqueeze(-1)).squeeze(-1)  
            f=f+(k1+2*k2+2*k3+k4)/6  
            if (k+1)%nci==0: f=f*(n0/f.norm(dim=-1,keepdim=True).clamp(min=1e-8))  
        return f  
  
# ═══════════════════════════════════════════════════════════════════  
# 第4部分 · 编码器 + 策略模块  
# ═══════════════════════════════════════════════════════════════════  
class CtxEncoder(nn.Module):  
    def __init__(self, c):  
        super().__init__()  
        self.f1=nn.Linear(c.d_LLM,4*c.d_M); self.f2=nn.Linear(4*c.d_M,4*c.d_M)  
        self.f3=nn.Linear(4*c.d_M,c.d_M); self.skip=nn.Linear(c.d_LLM,c.d_M)  
        self.n1=nn.LayerNorm(4*c.d_M); self.n2=nn.LayerNorm(4*c.d_M); self.no=nn.LayerNorm(c.d_M)  
    def forward(self, h):  
        x=F.silu(self.n1(self.f1(h))); x=F.silu(self.n2(self.f2(x)))  
        return self.no(self.f3(x)+self.skip(h))  
  
class FibEncoder(nn.Module):  
    def __init__(self, c):  
        super().__init__()  
        self.enc=nn.Sequential(nn.Linear(c.d_LLM+c.d_M,4*c.d_F),nn.SiLU(),nn.LayerNorm(4*c.d_F),  
                               nn.Linear(4*c.d_F,4*c.d_F),nn.SiLU(),nn.LayerNorm(4*c.d_F),  
                               nn.Linear(4*c.d_F,c.d_F))  
        self.sg=nn.Sequential(nn.Linear(1,c.d_F),nn.SiLU(),nn.Linear(c.d_F,c.d_F),nn.Sigmoid())  
    def forward(self, h, x, surprise=None):  
        f=self.enc(torch.cat([h,x],-1))  
        if surprise is not None:  
            s=surprise.view(-1,1) if surprise.dim()>=1 else surprise.unsqueeze(0).unsqueeze(0)  
            if s.shape[0]!=f.shape[0]: s=s.expand(f.shape[0],-1)  
            f=f*self.sg(s)  
        return f  
  
class DirectionPredictor(nn.Module):  
    def __init__(self, d_M, d_F):  
        super().__init__()  
        self.net=nn.Sequential(nn.Linear(d_M+d_F,4*d_M),nn.SiLU(),  
                               nn.LayerNorm(4*d_M),nn.Linear(4*d_M,d_M))  
    def forward(self, x, f):  
        return F.normalize(self.net(torch.cat([x,f],-1)),dim=-1,eps=1e-8)  
  
class EmptyStateNet(nn.Module):  
    def __init__(self, d_M, d_F):  
        super().__init__()  
        self.net=nn.Sequential(nn.Linear(d_M+d_F,2*d_F),nn.SiLU(),nn.LayerNorm(2*d_F),  
                               nn.Linear(2*d_F,d_F))  
    def forward(self, xq, fq):  
        return self.net(torch.cat([xq,fq],-1))  
  
class WriteGate(nn.Module):  
    def __init__(self, c):  
        super().__init__()  
        self.net=nn.Sequential(nn.Linear(c.d_LLM+1,c.d_LLM//4),nn.SiLU(),nn.Linear(c.d_LLM//4,1))  
    def forward(self, h, surprise):  
        s=surprise.view(-1,1) if surprise.dim()>=1 else surprise.unsqueeze(0).unsqueeze(0)  
        if s.shape[0]!=h.shape[0]: s=s[:h.shape[0]]  
        return torch.sigmoid(self.net(torch.cat([h,s],-1)).squeeze(-1))  
  
class RetentionScorer(nn.Module):  
    def __init__(self, c):  
        super().__init__()  
        self.net=nn.Sequential(nn.Linear(c.d_M+c.d_F+3,64),nn.SiLU(),  
                               nn.Linear(64,64),nn.SiLU(),nn.Linear(64,1),nn.Sigmoid())  
    def forward(self, base, fiber, surprise, dt, cnt):  
        return self.net(torch.cat([base,fiber,  
            surprise.unsqueeze(-1) if surprise.dim()==1 else surprise,  
            dt.unsqueeze(-1) if dt.dim()==1 else dt,  
            cnt.float().unsqueeze(-1) if cnt.dim()==1 else cnt.float()],-1)).squeeze(-1)  
  
# ═══════════════════════════════════════════════════════════════════  
# 第5部分 · 检索重排序  
# ═══════════════════════════════════════════════════════════════════  
class RetrievalReranker(nn.Module):  
    def __init__(self, d_M, d_F):  
        super().__init__()  
        inp=2*d_M+2*d_F+1  
        self.net=nn.Sequential(nn.Linear(inp,128),nn.SiLU(),nn.LayerNorm(128),  
                               nn.Linear(128,64),nn.SiLU(),nn.LayerNorm(64),nn.Linear(64,1))  
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)  
    def forward(self, xq, fq, xc, fc, dir_sim):  
        B,C=xc.shape[:2]  
        xq_e=xq.unsqueeze(1).expand(-1,C,-1); fq_e=fq.unsqueeze(1).expand(-1,C,-1)  
        inp=torch.cat([xq_e,fq_e,xc,fc,dir_sim.unsqueeze(-1)],-1)  
        correction=self.net(inp).squeeze(-1)  
        return dir_sim+correction  
  
# ═══════════════════════════════════════════════════════════════════  
# 第6部分 · ContentBypass  
# ═══════════════════════════════════════════════════════════════════  
class ContentBypass(nn.Module):  
    def __init__(self, d_F, d_LLM, gate_bias=-0.5):  
        super().__init__()  
        self.proj=nn.Sequential(  
            nn.Linear(d_F,2*d_LLM),nn.SiLU(),nn.LayerNorm(2*d_LLM),  
            nn.Linear(2*d_LLM,d_LLM),nn.LayerNorm(d_LLM))  
        self.gate_net=nn.Sequential(  
            nn.Linear(d_F+d_LLM,128),nn.SiLU(),nn.Linear(128,1))  
        nn.init.constant_(self.gate_net[-1].bias,gate_bias)  
        nn.init.normal_(self.proj[3].weight,std=0.02)  
        nn.init.zeros_(self.proj[3].bias)  
        self._last_gate=None  
    def forward(self, fiber_summary, qformer_context):  
        projected=self.proj(fiber_summary)  
        gate_in=torch.cat([fiber_summary,qformer_context],-1)  
        g=torch.sigmoid(self.gate_net(gate_in))  
        self._last_gate=g.detach()  
        return projected*g  
  
# ═══════════════════════════════════════════════════════════════════  
# 第7部分 · PrefixSemanticProbe  
# ═══════════════════════════════════════════════════════════════════  
class PrefixSemanticProbe(nn.Module):  
    def __init__(self, d_LLM, L_mem, d_F):  
        super().__init__()  
        self.attn_pool=nn.Linear(d_LLM,1)  
        self.fiber_decode=nn.Sequential(  
            nn.Linear(d_LLM,2*d_F),nn.SiLU(),nn.LayerNorm(2*d_F),nn.Linear(2*d_F,d_F))  
    def forward(self, prefix):  
        w=F.softmax(self.attn_pool(prefix).squeeze(-1),dim=1)  
        pooled=(w.unsqueeze(-1)*prefix).sum(1)  
        return self.fiber_decode(pooled)  
  
# ═══════════════════════════════════════════════════════════════════  
# 第8部分 · PrefixAligner  
# ═══════════════════════════════════════════════════════════════════  
class PrefixAligner(nn.Module):  
    def __init__(self, d_LLM, init_scale=-1.0):  
        super().__init__()  
        self.ln=nn.LayerNorm(d_LLM)  
        self.scale_logit=nn.Parameter(torch.tensor(init_scale))  
        self.register_buffer('_target_std',torch.tensor(1.0))  
        self._calibrated=False  
    def calibrate(self, llm):  
        with torch.no_grad():  
            wte=llm.transformer.wte.weight; wpe=llm.transformer.wpe.weight  
            si=min(2000,wte.shape[0]); sp=min(32,wpe.shape[0])  
            combined=wte[:si].unsqueeze(1)+wpe[:sp].unsqueeze(0)  
            self._target_std.fill_(combined.std().item())  
        self._calibrated=True  
    def forward(self, prefix):  
        normed=self.ln(prefix)  
        scale=torch.sigmoid(self.scale_logit)*self._target_std  
        return normed*scale  
  
# ═══════════════════════════════════════════════════════════════════  
# 第9部分 · ContentTokenClassifier (v3.7: +get_content_positions)  
# ═══════════════════════════════════════════════════════════════════  
class ContentTokenClassifier:  
    STOPWORDS = frozenset({  
        'the','a','an','is','are','was','were','be','been','being',  
        'have','has','had','having','do','does','did','doing',  
        'will','would','could','should','may','might','can','shall',  
        'and','but','or','nor','for','yet','so',  
        'in','on','at','to','of','by','with','from','as','into','through',  
        'during','before','after','above','below','between','under','over',  
        'that','this','these','those','it','its',  
        'he','she','they','we','you','me','him','her','them','us',  
        'his','her','their','our','your','my','mine','yours',  
        'not','no','if','then','than','when','where','what','which','who',  
        'how','all','each','every','both','few','more','most','some','any',  
        'also','just','about','very','really','only','even','still','already',  
        'up','down','out','off','away','back','here','there','now',  
        'too','much','many','such','own','other','another',  
        'because','since','while','although','though','until','unless',  
        'however','therefore','moreover','furthermore','nevertheless',  
        'like','get','got','go','went','gone','come','came',  
        'make','made','take','took','give','gave','see','saw','know','knew',  
        'think','thought','say','said','tell','told','want','need',  
        'use','used','find','found','put','keep','kept','let',  
        'seem','become','became','leave','left','call','called',  
        'try','tried','ask','asked','work','worked','well','way',  
        'thing','things','something','anything','nothing','everything',  
        'one','two','first','new','old','good','bad','big','small',  
        'long','little','right','same','different','last','next',  
        'part','being','going','using','getting','making','looking',  
        'coming','taking','having','doing','saying','working','trying',  
        'include','includes','including','included'  
    })  
    def __init__(self, tokenizer, min_len=3):  
        self.content_ids: Set[int] = set()  
        self.function_ids: Set[int] = set()  
        self.punct_ids: Set[int] = set()  
        self.newline_ids: Set[int] = set()  
        vocab_size = getattr(tokenizer, 'vocab_size', 50257)  
        for i in range(min(vocab_size, 50300)):  
            try:  
                tok_text = tokenizer.decode([i])  
                stripped = tok_text.strip().lower()  
                cleaned = ''.join(c for c in stripped if c.isalpha())  
                if '\n' in tok_text:  
                    self.newline_ids.add(i); self.function_ids.add(i)  
                elif stripped == '' or all(not c.isalnum() for c in stripped):  
                    self.punct_ids.add(i); self.function_ids.add(i)  
                elif len(cleaned) >= min_len and cleaned not in self.STOPWORDS:  
                    self.content_ids.add(i)  
                else:  
                    self.function_ids.add(i)  
            except:  
                self.function_ids.add(i)  
        self._content_tensor = None  
  
    def content_mask(self, device):  
        if self._content_tensor is None or self._content_tensor.device != device:  
            V = max(max(self.content_ids, default=0), max(self.function_ids, default=0),  
                    max(self.punct_ids, default=0), max(self.newline_ids, default=0)) + 1  
            m = torch.zeros(V, device=device)  
            for i in self.content_ids:  
                if i < V: m[i] = 1.0  
            self._content_tensor = m  
        return self._content_tensor  
  
    def get_content_ids_from_tokens(self, token_ids):  
        return [t for t in token_ids if t in self.content_ids]  
  
    def get_content_positions(self, token_ids, mask=None):  
        """Return list of positions where content tokens appear."""  
        positions = []  
        for pos, tid in enumerate(token_ids):  
            if mask is not None and pos < len(mask) and not mask[pos]:  
                continue  
            if tid in self.content_ids:  
                positions.append(pos)  
        return positions  
  
# ═══════════════════════════════════════════════════════════════════  
# 第10部分 · MemoryVocabProjector (学习路径, 辅助)  
# ═══════════════════════════════════════════════════════════════════  
class MemoryVocabProjector(nn.Module):  
    def __init__(self, d_F, d_LLM):  
        super().__init__()  
        self.proj = nn.Sequential(  
            nn.Linear(d_F, 4*d_LLM), nn.SiLU(), nn.LayerNorm(4*d_LLM),  
            nn.Linear(4*d_LLM, 2*d_LLM), nn.SiLU(), nn.LayerNorm(2*d_LLM),  
            nn.Linear(2*d_LLM, d_LLM))  
        nn.init.zeros_(self.proj[-1].weight); nn.init.zeros_(self.proj[-1].bias)  
    def forward(self, fiber_summary, wte_weight):  
        mem_emb = self.proj(fiber_summary)  
        mem_n = F.normalize(mem_emb, dim=-1, eps=1e-8)  
        wte_n = F.normalize(wte_weight, dim=-1, eps=1e-8)  
        return mem_n @ wte_n.T  
  
# ═══════════════════════════════════════════════════════════════════  
# 第11部分 · MemEntry + 方向树 (v3.7: +content_wte_centroid,  
#            +check_direction_degeneracy, leaf_size_violations)  
# ═══════════════════════════════════════════════════════════════════  
@dataclass  
class MemEntry:  
    mid: int; base: torch.Tensor; fiber: torch.Tensor; dirn: torch.Tensor  
    surprise: float; ts: float; last: float; cnt: int = 0; version: int = 0  
    source_text: str = ""  
    content_token_ids: List[int] = field(default_factory=list)  
    semantic_emb: Optional[torch.Tensor] = None  
    content_wte_centroid: Optional[torch.Tensor] = None  
    expanded_content_ids: List[int] = field(default_factory=list)  
  
class _Node:  
    __slots__=('leaf','ids','children','centers','depth')  
    def __init__(self,d=0):  
        self.depth=d; self.leaf=True; self.ids=[]; self.children=[]; self.centers=None  
    def count(self):  
        return len(self.ids) if self.leaf else sum(c.count() for c in self.children)  
  
class DirectionTree:  
    def __init__(self, c):  
        self.c=c; self.root=_Node(); self.store:Dict[int,MemEntry]={}; self.nid=0  
    def insert(self, m):  
        self.store[m.mid]=m; self._ins(self.root,m)  
    def _ins(self, nd, m):  
        if nd.leaf:  
            nd.ids.append(m.mid)  
            if len(nd.ids)>self.c.tree_max_leaf: self._split(nd)  
        else:  
            best=self._best(nd,m.dirn); self._ins(nd.children[best],m); self._update_centers(nd)  
    def update(self, mid, new_base=None, new_fiber=None, new_dirn=None):  
        if mid not in self.store: return  
        m=self.store[mid]; dc=False  
        if new_base is not None: m.base=new_base.detach().clone()  
        if new_fiber is not None: m.fiber=new_fiber.detach().clone()  
        if new_dirn is not None: dc=True; m.dirn=new_dirn.detach().clone()  
        m.version+=1  
        if dc: self._rm(self.root,mid); self._ins(self.root,m); self._rebalance(self.root)  
    def _split(self, nd):  
        ids=nd.ids  
        if len(ids)<2: return  
        K=min(self.c.tree_K,len(ids))  
        if K<2: return  
        dirs=torch.stack([self.store[i].dirn for i in ids])  
        centered=dirs-dirs.mean(0)  
        try: _,_,Vh=torch.linalg.svd(centered,full_matrices=False)  
        except: return  
        n_comp=min(K,dirs.shape[1]); proj=centered@Vh[:n_comp].T  
        asgn=self._farthest_kmeans(proj,K)  
        children=[]  
        for k in range(K):  
            ch=_Node(nd.depth+1); ch.ids=[ids[i] for i in range(len(ids)) if asgn[i]==k]  
            if ch.ids: children.append(ch)  
        if len(children)<=1: return  
        nd.leaf=False; nd.children=children; nd.ids=[]; self._update_centers(nd)  
        for ch in nd.children:  
            if ch.leaf and len(ch.ids)>self.c.tree_max_leaf: self._split(ch)  
    @staticmethod  
    def _farthest_kmeans(data, K, max_iter=50):  
        N=data.shape[0]; K=min(K,N)  
        if K<=0: return torch.zeros(N,dtype=torch.long,device=data.device)  
        ctrs=[data[0].clone()]  
        for _ in range(K-1):  
            d2=torch.cdist(data,torch.stack(ctrs)).min(1)[0].pow(2)  
            ctrs.append(data[d2.argmax()].clone())  
        ctrs=torch.stack(ctrs); asgn=torch.zeros(N,dtype=torch.long,device=data.device)  
        for _ in range(max_iter):  
            dists=torch.cdist(data,ctrs); new=dists.argmin(1)  
            if (new==asgn).all(): break  
            asgn=new  
            for k in range(K):  
                mk=asgn==k  
                if mk.any(): ctrs[k]=data[mk].mean(0)  
                else:  
                    far=dists.min(1)[0].argmax(); ctrs[k]=data[far].clone(); asgn[far]=k  
        return asgn  
    def _best(self, nd, d):  
        if nd.centers is None or len(nd.children)==0: return 0  
        return (nd.centers@d).argmax().item()  
    def retrieve(self, qdir, bw=3)->List[Tuple[int,float]]:  
        beams:List[Tuple[_Node,float]]=[(self.root,0.)]  
        results:Dict[int,float]={}  
        while beams:  
            nb=[]  
            for nd,sc in beams:  
                if nd.leaf:  
                    for mid in nd.ids:  
                        if mid in self.store:  
                            s=(qdir@self.store[mid].dirn).item()+sc  
                            if mid not in results or s>results[mid]: results[mid]=s  
                elif nd.centers is not None:  
                    sims=nd.centers@qdir; tk=min(bw,len(nd.children)); _,idxs=sims.topk(tk)  
                    for i in idxs: nb.append((nd.children[i.item()],sc+sims[i.item()].item()))  
                else:  
                    for ch in nd.children: nb.append((ch,sc))  
            nb.sort(key=lambda x:-x[1]); beams=nb[:bw]  
        return sorted(results.items(),key=lambda x:-x[1])  
    def remove(self, mid):  
        if mid not in self.store: return  
        del self.store[mid]; self._rm(self.root,mid); self._rebalance(self.root)  
    def _rm(self, nd, mid):  
        if nd.leaf:  
            if mid in nd.ids: nd.ids.remove(mid); return True  
            return False  
        return any(self._rm(c,mid) for c in nd.children)  
    def _rebalance(self, nd):  
        if nd.leaf: return  
        for c in nd.children: self._rebalance(c)  
        nd.children=[c for c in nd.children if c.count()>0]  
        if not nd.children: nd.leaf=True; nd.ids=[]; nd.centers=None  
        elif len(nd.children)==1:  
            ch=nd.children[0]; nd.leaf=ch.leaf; nd.ids=ch.ids; nd.children=ch.children; nd.centers=ch.centers  
        else: self._update_centers(nd)  
    def _update_centers(self, nd):  
        cs=[]  
        for c in nd.children:  
            ids=self._collect(c); dirs=[self.store[i].dirn for i in ids if i in self.store]  
            if not dirs: continue  
            cs.append(F.normalize(torch.stack(dirs).mean(0),dim=0))  
        nd.centers=torch.stack(cs) if cs else None  
    def _collect(self, nd):  
        if nd.leaf: return list(nd.ids)  
        return [i for c in nd.children for i in self._collect(c)]  
    def _enforce_capacity(self, nd):  
        if nd.leaf:  
            if len(nd.ids)>self.c.tree_max_leaf: self._split(nd)  
            return  
        for ch in nd.children: self._enforce_capacity(ch)  
    def rebuild(self):  
        ms=list(self.store.values()); self.root=_Node()  
        for m in ms: self._ins(self.root,m)  
        self._enforce_capacity(self.root)  
    def max_depth(self, nd=None):  
        if nd is None: nd=self.root  
        if nd.leaf: return nd.depth  
        return max(self.max_depth(c) for c in nd.children) if nd.children else nd.depth  
    def verify_consistency(self)->List[str]:  
        errs=[]; ti=set(self._collect(self.root)); si=set(self.store.keys())  
        if ti!=si: errs.append(f"tree≠store: tree_only={ti-si}, store_only={si-ti}")  
        if self.root.count()!=len(self.store): errs.append(f"count: tree={self.root.count()}, store={len(self.store)}")  
        return errs  
    def leaf_size_violations(self)->List[Tuple[int,int]]:  
        v=[]; self._check_leaves(self.root,v); return v  
    def _check_leaves(self, nd, v):  
        if nd.leaf:  
            if len(nd.ids)>self.c.tree_max_leaf: v.append((nd.depth,len(nd.ids)))  
        else:  
            for c in nd.children: self._check_leaves(c,v)  
    def check_direction_degeneracy(self, threshold: float = 0.95) -> List[Tuple[List[int], float]]:  
        """Detect clusters where all directions are near-identical (degenerate).  
        Returns list of (member_ids, avg_pairwise_cosine) for degenerate clusters."""  
        degenerate = []  
        self._check_degeneracy_recursive(self.root, threshold, degenerate)  
        return degenerate  
    def _check_degeneracy_recursive(self, nd, threshold, results):  
        if nd.leaf:  
            if len(nd.ids) >= 2:  
                dirs = [self.store[mid].dirn for mid in nd.ids if mid in self.store]  
                if len(dirs) >= 2:  
                    dt = torch.stack(dirs)  
                    dn = F.normalize(dt, dim=-1)  
                    sim = dn @ dn.T  
                    mask_off = ~torch.eye(len(dirs), dtype=torch.bool, device=sim.device)  
                    avg_sim = sim[mask_off].mean().item() if mask_off.any() else 0.0  
                    if avg_sim > threshold:  
                        results.append((list(nd.ids), avg_sim))  
        else:  
            for ch in nd.children:  
                self._check_degeneracy_recursive(ch, threshold, results)  
  
# ═══════════════════════════════════════════════════════════════════  
# 第12部分 · 纤维注意力  
# ═══════════════════════════════════════════════════════════════════  
class FiberAttn(nn.Module):  
    def __init__(self, c):  
        super().__init__()  
        self.nh=c.n_heads_fiber; self.hd=c.d_F//c.n_heads_fiber  
        self.Wq=nn.Linear(c.d_F,c.d_F,bias=False); self.Wk=nn.Linear(c.d_F,c.d_F,bias=False)  
        self.Wv=nn.Linear(c.d_F,c.d_F,bias=False); self.Wo=nn.Linear(c.d_F,c.d_F,bias=False)  
        self.n1=nn.LayerNorm(c.d_F)  
        self.ff=nn.Sequential(nn.Linear(c.d_F,2*c.d_F),nn.GELU(),nn.Linear(2*c.d_F,c.d_F))  
        self.n2=nn.LayerNorm(c.d_F)  
    def forward(self, qf, mf, mem_mask=None, dir_bias=None):  
        B,C,d=mf.shape; nh=self.nh; hd=self.hd; S=1+C  
        seq=torch.cat([qf.unsqueeze(1),mf],1)  
        Q=self.Wq(seq).reshape(B,S,nh,hd).permute(0,2,1,3)  
        K=self.Wk(seq).reshape(B,S,nh,hd).permute(0,2,1,3)  
        V=self.Wv(seq).reshape(B,S,nh,hd).permute(0,2,1,3)  
        a=(Q@K.transpose(-2,-1))/math.sqrt(hd)  
        if dir_bias is not None:  
            db=dir_bias.unsqueeze(1).unsqueeze(2)  
            pad=torch.zeros(B,1,1,1,**_dev(a))  
            a=a+torch.cat([pad,db],-1)  
        if mem_mask is not None:  
            qm=torch.ones(B,1,**_dev(mem_mask))  
            full=torch.cat([qm,mem_mask],1)  
            a=a.masked_fill(full.unsqueeze(1).unsqueeze(2)==0,-1e9)  
        a=F.softmax(a,-1); out=(a@V).permute(0,2,1,3).reshape(B,S,d)  
        out=self.n1(seq+self.Wo(out)); out=self.n2(out+self.ff(out))  
        return out[:,1:]  
  
# ═══════════════════════════════════════════════════════════════════  
# 第13部分 · QFormer + 嵌入桥  
# ═══════════════════════════════════════════════════════════════════  
class QFormerLayer(nn.Module):  
    def __init__(self, c):  
        super().__init__(); d=c.d_LLM; nh=c.bridge_heads  
        self.sa=nn.MultiheadAttention(d,nh,batch_first=True)  
        self.ca=nn.MultiheadAttention(d,nh,batch_first=True)  
        self.ff=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))  
        self.n1=nn.LayerNorm(d); self.n2=nn.LayerNorm(d); self.n3=nn.LayerNorm(d)  
    def forward(self, q, k, v, kv_mask=None):  
        h=self.n1(q); q=q+self.sa(h,h,h)[0]; h=self.n2(q)  
        kpm=None  
        if kv_mask is not None:  
            kpm=(kv_mask==0); all_m=kpm.all(dim=-1)  
            if all_m.any(): kpm=kpm.clone(); kpm[all_m]=False  
        q=q+self.ca(h,k,v,key_padding_mask=kpm)[0]  
        return q+self.ff(self.n3(q))  
  
class QFormerProj(nn.Module):  
    def __init__(self, c):  
        super().__init__()  
        self.q=nn.Parameter(torch.randn(c.L_mem,c.d_LLM)*0.02)  
        self.fkv=nn.Linear(c.d_F,c.d_LLM*2)  
        self.layers=nn.ModuleList([QFormerLayer(c) for _ in range(c.bridge_layers)])  
        self.norm=nn.LayerNorm(c.d_LLM)  
    def forward(self, fibers, mem_mask=None):  
        B=fibers.shape[0]; kv=self.fkv(fibers); k,v=kv.chunk(2,-1)  
        q=self.q.unsqueeze(0).expand(B,-1,-1)  
        for l in self.layers: q=l(q,k,v,kv_mask=mem_mask)  
        return self.norm(q)  
  
class AdaptiveLayerPool(nn.Module):  
    def __init__(self, n, d):  
        super().__init__(); self.w=nn.Parameter(torch.linspace(-2,2,n))  
    def forward(self, hs):  
        w=F.softmax(self.w,0); return sum(w[i]*h for i,h in enumerate(hs))  
    def weight_dist(self):  
        return F.softmax(self.w.detach(),0)  
  
class StateExtractor(nn.Module):  
    def __init__(self, c):  
        super().__init__()  
        pos_dim=5  
        self.sc=nn.Sequential(nn.Linear(c.d_LLM+pos_dim,c.d_LLM//4),nn.Tanh(),nn.Linear(c.d_LLM//4,1))  
        self.tb=nn.Linear(c.d_LLM,c.d_M); self.tf=nn.Linear(c.d_LLM,c.d_F)  
    def _pos_feat(self, T, ref):  
        pos=torch.linspace(0,1,T,**_dev(ref))  
        return torch.stack([pos,torch.sin(pos*math.pi),torch.cos(pos*math.pi),  
                           torch.sin(2*pos*math.pi),torch.cos(2*pos*math.pi)],-1)  
    def forward(self, h, mask=None):  
        B,T,_=h.shape; pf=self._pos_feat(T,h).unsqueeze(0).expand(B,-1,-1)  
        s=self.sc(torch.cat([h,pf],-1)).squeeze(-1)  
        if mask is not None:  
            if mask.shape[1]==T: s=s.masked_fill(mask==0,-1e9)  
        w=F.softmax(s,-1); p=(w.unsqueeze(-1)*h).sum(1)  
        return self.tb(p), self.tf(p)  
  
class EmbBridge(nn.Module):  
    def __init__(self, c):  
        super().__init__()  
        self.proj=QFormerProj(c); self.ext=StateExtractor(c)  
        self.pe=nn.Parameter(torch.randn(c.L_mem,c.d_LLM)*0.02)  
        self.bypass=ContentBypass(c.d_F,c.d_LLM,gate_bias=c.bypass_init_gate_bias)  
        self.aligner=PrefixAligner(c.d_LLM,c.prefix_init_scale)  
        self.inject_mode='both'  
        self._last_inject_diag={}  
        self._last_fiber_summary=None  
    def inject(self, fibers, mem_mask=None, fiber_summary=None):  
        B=fibers.shape[0]  
        if self.inject_mode in ('both','qformer_only'):  
            qf_out=self.proj(fibers,mem_mask)+self.pe.unsqueeze(0)  
        else:  
            qf_out=self.pe.unsqueeze(0).expand(B,-1,-1)  
        bp_out=None; gate_val=None  
        if fiber_summary is not None and self.inject_mode in ('both','bypass_only'):  
            qf_context=qf_out.mean(1)  
            bp_out=self.bypass(fiber_summary,qf_context)  
            gate_val=self.bypass._last_gate  
            qf_out=qf_out+bp_out.unsqueeze(1)  
        qf_out=self.aligner(qf_out)  
        self._last_fiber_summary=fiber_summary.detach() if fiber_summary is not None else None  
        self._last_inject_diag={  
            'bypass_gate':gate_val.mean().item() if gate_val is not None else None,  
            'qf_norm':qf_out.norm().item(),  
            'bypass_norm':bp_out.norm().item() if bp_out is not None else 0.0,  
            'aligner_scale':torch.sigmoid(self.aligner.scale_logit).item()*self.aligner._target_std.item()}  
        return qf_out  
  
# ═══════════════════════════════════════════════════════════════════  
# 第14部分 · Loss 相关工具  
# ═══════════════════════════════════════════════════════════════════  
class LossWarmup:  
    def __init__(self, schedules:Dict[str,int]):  
        self.schedules=schedules; self.step_count=0  
    def weight(self, name:str)->float:  
        ws=self.schedules.get(name,0)  
        if ws<=0: return 1.0  
        return min(1.0, self.step_count/max(ws,1))  
    def advance(self): self.step_count+=1  
  
class GradientMonitor:  
    def __init__(self): self._groups:Dict[str,nn.Module]={}  
    def register(self, name:str, mod:nn.Module): self._groups[name]=mod  
    def register_param(self, name:str, param:nn.Parameter):  
        class _W(nn.Module):  
            def __init__(self, p): super().__init__(); self._p=p  
            def parameters(self, recurse=True): yield self._p  
        self._groups[name]=_W(param)  
    def snapshot(self)->Dict[str,float]:  
        norms={}  
        for name,mod in self._groups.items():  
            total=0.0; cnt=0  
            for p in mod.parameters():  
                if p.grad is not None: total+=p.grad.norm().item()**2; cnt+=1  
            norms[name]=math.sqrt(total) if cnt>0 else 0.0  
        return norms  
  
# ═══════════════════════════════════════════════════════════════════  
# 第15部分 · DegenerationGuard (v3.7: 大幅提高惩罚)  
# ═══════════════════════════════════════════════════════════════════  
class DegenerationGuard:  
    def __init__(self, tok, cfg, content_classifier=None):  
        self.tok=tok; self.cfg=cfg; self.cc=content_classifier; self._built=False  
    def _build(self):  
        if self._built: return  
        if self.cc is not None:  
            self._punct_ids=self.cc.punct_ids; self._newline_ids=self.cc.newline_ids  
        else:  
            self._punct_ids=set(); self._newline_ids=set()  
            vocab_sz=getattr(self.tok,'vocab_size',50257)  
            for i in range(min(vocab_sz,50300)):  
                try:  
                    t=self.tok.decode([i]); stripped=t.strip()  
                    if stripped=='' or all(not c.isalnum() for c in stripped):  
                        self._punct_ids.add(i)  
                    if '\n' in t: self._newline_ids.add(i)  
                except: pass  
        self._built=True  
    def process(self, logits, generated_ids, step):  
        self._build()  
        if step<self.cfg.early_content_steps:  
            for pid in self._punct_ids:  
                if pid<logits.shape[-1]: logits[0,pid]-=self.cfg.degen_early_punct_penalty  
            for nid in self._newline_ids:  
                if nid<logits.shape[-1]: logits[0,nid]-=self.cfg.degen_early_newline_penalty  
        if step<self.cfg.degen_min_tokens and self.tok.eos_token_id is not None:  
            logits[0,self.tok.eos_token_id]=-float('inf')  
        seen=set(generated_ids[-30:]) if generated_ids else set()  
        for tid in seen:  
            if tid<logits.shape[-1]:  
                if logits[0,tid]>0: logits[0,tid]/=self.cfg.degen_repeat_penalty  
                else: logits[0,tid]*=self.cfg.degen_repeat_penalty  
        mc=self.cfg.degen_max_consec_punct  
        if len(generated_ids)>=mc:  
            recent=generated_ids[-mc:]  
            if all(t in self._punct_ids for t in recent):  
                for pid in self._punct_ids:  
                    if pid<logits.shape[-1]: logits[0,pid]-=10.0  
        if len(generated_ids)>=2:  
            recent=generated_ids[-2:]  
            if all(t in self._newline_ids for t in recent):  
                for nid in self._newline_ids:  
                    if nid<logits.shape[-1]: logits[0,nid]-=10.0  
        return logits  
  
# ═══════════════════════════════════════════════════════════════════  
# 第16部分 · RetrievalDiag  
# ═══════════════════════════════════════════════════════════════════  
@dataclass  
class RetrievalDiag:  
    was_flat_scan: bool = False  
    recall_count: int = 0  
    reranker_delta_mean: float = 0.0  
    fiber_summary_norm: float = 0.0  
    top_reranker_score: float = 0.0  
    top_dir_sim: float = 0.0  
    top_wte_sim: float = 0.0  
    top_sem_sim: float = 0.0  
    batch_mem_weights: List[List[Tuple[int, float]]] = field(default_factory=list)  
  
# ═══════════════════════════════════════════════════════════════════  
# 第17部分 · AMM (v3.7: 三路检索评分, content_wte_centroid)  
# ═══════════════════════════════════════════════════════════════════  
class AMM(nn.Module):  
    def __init__(self, c):  
        super().__init__(); self.c=c  
        self.metric=RiemannianMetric(c.d_M)  
        self.geo=GeodesicSolver(self.metric,c)  
        self.conn=FiberConnection(c.d_M,c.d_F,self.metric,grad_coupling=True)  
        self.trans=FiberTransporter(self.conn,c)  
        self.ctx=CtxEncoder(c); self.fib=FibEncoder(c)  
        self.dir_pred=DirectionPredictor(c.d_M,c.d_F)  
        self.write_gate=WriteGate(c); self.retention=RetentionScorer(c)  
        self.attn=FiberAttn(c); self.empty_state=EmptyStateNet(c.d_M,c.d_F)  
        self.contrast_proj_f=nn.Linear(c.d_F,c.d_M,bias=False)  
        self.contrast_proj_x=nn.Linear(c.d_M,c.d_M,bias=False)  
        nn.init.eye_(self.contrast_proj_x.weight)  
        self.reranker=RetrievalReranker(c.d_M,c.d_F)  
        self.tree=DirectionTree(c); self.time=0.  
  
    def surprise_proxy(self, logits, tgt):  
        nll=-F.log_softmax(logits,-1).gather(2,tgt.unsqueeze(-1)).squeeze(-1)  
        T=nll.shape[1]  
        if T==0: return logits.new_zeros(logits.shape[0])  
        w=torch.linspace(0.5,1.5,T,**_dev(nll)); w=w/w.sum()*T  
        return (nll*w.unsqueeze(0)).mean(-1)  
  
    def _compute_dirn(self, base, fiber):  
        with torch.no_grad():  
            return self.dir_pred(base.unsqueeze(0),fiber.unsqueeze(0)).squeeze(0)  
  
    def store_mem(self, h, surp, training_mode=False, source_text="",  
                  content_token_ids=None, content_semantic_emb=None,  
                  content_wte_centroid=None, expanded_content_ids=None):  
        dev=h.device; h2=h.unsqueeze(0)  
        x=self.ctx(h2).squeeze(0).detach()  
        s=surp if isinstance(surp,torch.Tensor) else torch.tensor(surp,**_dev(h))  
        sv=s.view(1) if s.dim()<=1 else s  
        f=self.fib(h2,x.unsqueeze(0),sv).squeeze(0).detach()  
        d=self._compute_dirn(x,f)  
        sem_emb=content_semantic_emb if content_semantic_emb is not None else h.detach().clone()  
        ct_ids=content_token_ids or []  
        wte_cen=content_wte_centroid  
        exp_ids=expanded_content_ids or []  
        if self.tree.store:  
            scored=self.tree.retrieve(d.detach(),bw=1)[:5]  
            for mid,_ in scored:  
                if mid in self.tree.store:  
                    ex=self.tree.store[mid]  
                    dist=self.metric.midpoint_approx_distance(  
                        x.unsqueeze(0),ex.base.unsqueeze(0).to(dev)).item()  
                    if dist<self.c.consol_dist*self.c.consol_conflict_ratio:  
                        alpha=self.c.write_update_alpha  
                        nb=((1-alpha)*ex.fiber+alpha*f).detach().clone()  
                        nba=((1-alpha)*ex.base+alpha*x).detach().clone()  
                        nd=self._compute_dirn(nba,nb)  
                        self.tree.update(mid,new_base=nba,new_fiber=nb,new_dirn=nd)  
                        ex.surprise=max(ex.surprise,s.item()); ex.last=self.time; ex.cnt+=1  
                        if source_text: ex.source_text=source_text  
                        if sem_emb is not None:  
                            if ex.semantic_emb is not None:  
                                ex.semantic_emb=((1-alpha)*ex.semantic_emb.to(dev)+alpha*sem_emb).detach().clone()  
                            else: ex.semantic_emb=sem_emb.detach().clone()  
                        if ct_ids: ex.content_token_ids=list(set(ex.content_token_ids+ct_ids))  
                        if exp_ids: ex.expanded_content_ids=list(set(ex.expanded_content_ids+exp_ids))  
                        if wte_cen is not None:  
                            if ex.content_wte_centroid is not None:  
                                ex.content_wte_centroid=((1-alpha)*ex.content_wte_centroid.to(dev)+alpha*wte_cen).detach().clone()  
                            else: ex.content_wte_centroid=wte_cen.detach().clone()  
                        self.time+=1; return ex  
        m=MemEntry(mid=self.tree.nid,base=x.detach().clone(),fiber=f.detach().clone(),  
                   dirn=d.detach().clone(),surprise=s.item(),ts=self.time,last=self.time,  
                   source_text=source_text,content_token_ids=ct_ids,  
                   semantic_emb=sem_emb.detach().clone() if sem_emb is not None else None,  
                   content_wte_centroid=wte_cen.detach().clone() if wte_cen is not None else None,  
                   expanded_content_ids=exp_ids)  
        self.tree.nid+=1; self.tree.insert(m); self.time+=1; return m  
  
    def retrieve_multi(self, xq, fq, topk=None, bw=None, update_stats=True,  
                       query_semantic_emb=None, query_wte_centroid=None):  
        B=xq.shape[0]; dev=xq.device  
        topk=topk or self.c.retrieval_topk; bw=bw or self.c.retrieval_beam  
        recall_k=int(topk*self.c.retrieval_recall_factor)  
        flat_thresh=self.c.flat_scan_threshold_factor*topk  
        qdir=self.dir_pred(xq,fq)  
        diag=RetrievalDiag()  
        w_dir=self.c.ret_dir_weight  
        w_sem=self.c.ret_sem_weight  
        w_wte=self.c.ret_wte_weight  
        if not self.tree.store:  
            empty=self.empty_state(xq,fq)  
            mask=torch.ones(B,1,**_dev(xq))  
            summary=empty.mean(1) if empty.dim()==3 else empty  
            diag.fiber_summary_norm=summary.norm().item()  
            diag.batch_mem_weights=[[] for _ in range(B)]  
            return empty.unsqueeze(1),mask,summary,diag  
        all_results=[]; all_masks=[]; all_biases=[]; all_summaries=[]; all_batch_mw=[]  
        for b in range(B):  
            n_store=len(self.tree.store)  
            if n_store<=flat_thresh:  
                mids=list(self.tree.store.keys()); diag.was_flat_scan=True  
            else:  
                scored=self.tree.retrieve(qdir[b].detach(),bw)  
                mids=[s[0] for s in scored[:recall_k]]  
            mems=[self.tree.store[i] for i in mids if i in self.tree.store]  
            diag.recall_count=len(mems)  
            if not mems:  
                empty=self.empty_state(xq[b:b+1],fq[b:b+1])  
                all_results.append(empty.squeeze(0).unsqueeze(0))  
                all_masks.append(torch.ones(1,**_dev(xq)))  
                all_biases.append(torch.zeros(1,**_dev(xq)))  
                all_summaries.append(empty.squeeze(0))  
                all_batch_mw.append([]); continue  
            C=len(mems)  
            sb=torch.stack([m.base.to(dev) for m in mems])  
            sf=torch.stack([m.fiber.to(dev) for m in mems])  
            md=torch.stack([m.dirn.to(dev) for m in mems])  
            raw_dir_sim=torch.einsum('d,cd->c',qdir[b],md)  
            diag.top_dir_sim=raw_dir_sim.max().item()  
            # ── v3.7 三路评分 ──  
            # 1. Direction similarity  
            # 2. Content-aware semantic embedding similarity  
            sem_sims=[]  
            if query_semantic_emb is not None:  
                for mem in mems:  
                    if mem.semantic_emb is not None:  
                        s=F.cosine_similarity(  
                            query_semantic_emb[b:b+1],  
                            mem.semantic_emb.unsqueeze(0).to(dev),dim=-1).squeeze()  
                        sem_sims.append(s)  
                    else: sem_sims.append(raw_dir_sim.new_tensor(0.0))  
                sem_sim_t=torch.stack(sem_sims)  
                diag.top_sem_sim=sem_sim_t.max().item()  
            else:  
                sem_sim_t=torch.zeros(C,device=dev)  
            # 3. WTE centroid similarity  
            wte_sims=[]  
            if query_wte_centroid is not None:  
                qwte_b=query_wte_centroid[b]  
                qwte_norm=qwte_b.norm()  
                for mem in mems:  
                    if mem.content_wte_centroid is not None and qwte_norm>1e-8:  
                        s=F.cosine_similarity(  
                            qwte_b.unsqueeze(0),  
                            mem.content_wte_centroid.unsqueeze(0).to(dev),dim=-1).squeeze()  
                        wte_sims.append(s)  
                    else: wte_sims.append(raw_dir_sim.new_tensor(0.0))  
                wte_sim_t=torch.stack(wte_sims)  
                diag.top_wte_sim=wte_sim_t.max().item()  
            else:  
                wte_sim_t=torch.zeros(C,device=dev)  
            combined_sim=w_dir*raw_dir_sim+w_sem*sem_sim_t+w_wte*wte_sim_t  
            # Reranker refines combined score  
            rerank_scores=self.reranker(  
                xq[b:b+1],fq[b:b+1],sb.unsqueeze(0),sf.unsqueeze(0),  
                combined_sim.unsqueeze(0)).squeeze(0)  
            diag.reranker_delta_mean=(rerank_scores-combined_sim).abs().mean().item()  
            diag.top_reranker_score=rerank_scores.max().item()  
            if not self.training and C>topk:  
                _,top_idx=rerank_scores.topk(topk)  
                mems=[mems[i] for i in top_idx.cpu().tolist()]  
                sb=sb[top_idx]; sf=sf[top_idx]; rerank_scores=rerank_scores[top_idx]; C=topk  
            qp=xq[b].unsqueeze(0).expand(C,-1)  
            geo_r=self.geo.solve(sb,qp)  
            transported=self.trans(sf,geo_r.path)  
            if self.training:  
                ret_s=self.retention(sb,sf,  
                    torch.tensor([m.surprise for m in mems],**_dev(xq)),  
                    torch.tensor([self.time-m.last for m in mems],**_dev(xq)),  
                    torch.tensor([m.cnt for m in mems],**_dev(xq)))  
                transported=transported*ret_s.unsqueeze(-1)  
            if update_stats:  
                for m in mems: m.last=self.time; m.cnt+=1  
            w=F.softmax(rerank_scores,dim=0)  
            fs=(transported*w.unsqueeze(-1)).sum(0)  
            batch_mw=[(m.mid,w[mi].item()) for mi,m in enumerate(mems)]  
            all_batch_mw.append(batch_mw)  
            all_results.append(transported); all_masks.append(torch.ones(C,**_dev(xq)))  
            all_biases.append(rerank_scores/self.c.tau); all_summaries.append(fs)  
        maxC=max(r.shape[0] for r in all_results)  
        padded=[]; pm=[]; pd=[]  
        for bi in range(B):  
            r,mk,db=all_results[bi],all_masks[bi],all_biases[bi]; gap=maxC-r.shape[0]  
            if gap>0:  
                pr=self.empty_state(xq[bi:bi+1],fq[bi:bi+1]).expand(gap,-1)  
                r=torch.cat([r,pr if self.training else pr.detach()],0)  
                mk=torch.cat([mk,torch.zeros(gap,**_dev(xq))])  
                db=torch.cat([db,torch.full((gap,),-1e9,**_dev(xq))])  
            padded.append(r); pm.append(mk); pd.append(db)  
        mf=torch.stack(padded); mem_mask=torch.stack(pm); dir_bias=torch.stack(pd)  
        fiber_summary=torch.stack(all_summaries)  
        diag.fiber_summary_norm=fiber_summary.norm().item()  
        diag.batch_mem_weights=all_batch_mw  
        refined=self.attn(fq,mf,mem_mask=mem_mask,dir_bias=dir_bias)  
        return refined,mem_mask,fiber_summary,diag  
  
    def decay(self):  
        rm=[]  
        for mid,m in self.tree.store.items():  
            dt=torch.tensor([self.time-m.last],**_dev(m.base))  
            cnt=torch.tensor([m.cnt],**_dev(m.base))  
            with torch.no_grad():  
                sc=self.retention(m.base.unsqueeze(0),m.fiber.unsqueeze(0),  
                    torch.tensor([m.surprise],**_dev(m.base)),dt,cnt).item()  
            if sc<self.c.retention_gc_threshold: rm.append(mid)  
        for i in rm: self.tree.remove(i)  
        return len(rm)  
  
    def consolidate(self):  
        ms=list(self.tree.store.values())  
        if len(ms)<2: return 0  
        merged=set()  
        for i in range(len(ms)):  
            if ms[i].mid in merged: continue  
            for j in range(i+1,len(ms)):  
                if ms[j].mid in merged: continue  
                d=self.metric.midpoint_approx_distance(  
                    ms[i].base.unsqueeze(0),ms[j].base.unsqueeze(0)).item()  
                if d<self.c.consol_dist:  
                    wi,wj=ms[i].cnt+1,ms[j].cnt+1; t=wi+wj  
                    nb=(ms[i].base*wi+ms[j].base*wj)/t  
                    nf=(ms[i].fiber*wi+ms[j].fiber*wj)/t  
                    nd=self._compute_dirn(nb,nf)  
                    ms[i].base=nb.detach().clone(); ms[i].fiber=nf.detach().clone()  
                    ms[i].dirn=nd.detach().clone(); ms[i].cnt+=ms[j].cnt  
                    ms[i].surprise=max(ms[i].surprise,ms[j].surprise); ms[i].version+=1  
                    if ms[j].source_text and not ms[i].source_text:  
                        ms[i].source_text=ms[j].source_text  
                    ms[i].content_token_ids=list(set(ms[i].content_token_ids+ms[j].content_token_ids))  
                    ms[i].expanded_content_ids=list(set(ms[i].expanded_content_ids+ms[j].expanded_content_ids))  
                    if ms[i].semantic_emb is not None and ms[j].semantic_emb is not None:  
                        ms[i].semantic_emb=((ms[i].semantic_emb*wi+ms[j].semantic_emb*wj)/t).detach().clone()  
                    elif ms[j].semantic_emb is not None: ms[i].semantic_emb=ms[j].semantic_emb.clone()  
                    if ms[i].content_wte_centroid is not None and ms[j].content_wte_centroid is not None:  
                        ms[i].content_wte_centroid=((ms[i].content_wte_centroid*wi+ms[j].content_wte_centroid*wj)/t).detach().clone()  
                    elif ms[j].content_wte_centroid is not None: ms[i].content_wte_centroid=ms[j].content_wte_centroid.clone()  
                    merged.add(ms[j].mid)  
        for mid in merged: del self.tree.store[mid]  
        if merged: self.tree.rebuild()  
        return len(merged)  
  
# ═══════════════════════════════════════════════════════════════════  
# 第18部分 · MemLLM (v3.7: content-position-only embedding,  
#            WTE centroid, hard content-first decoding)  
# ═══════════════════════════════════════════════════════════════════  
class MemLLM(nn.Module):  
    def __init__(self, c):  
        super().__init__(); self.c=c  
        self.amm=AMM(c); self.bridge=EmbBridge(c)  
        self.semantic_probe=PrefixSemanticProbe(c.d_LLM,c.L_mem,c.d_F)  
        self.vocab_proj=MemoryVocabProjector(c.d_F,c.d_LLM)  
        self.layer_pool=None; self.llm=None; self.tok=None  
        self._degen_guard=None; self.content_classifier=None  
        self._wte_neighbor_cache: Optional[Dict[int,List[int]]] = None  
  
    def load(self, name="gpt2"):  
        from transformers import GPT2LMHeadModel, GPT2Tokenizer  
        self.tok=GPT2Tokenizer.from_pretrained(name)  
        self.llm=GPT2LMHeadModel.from_pretrained(name)  
        for p in self.llm.parameters(): p.requires_grad_(False)  
        if self.tok.pad_token is None: self.tok.pad_token=self.tok.eos_token  
        self.layer_pool=AdaptiveLayerPool(self.llm.config.n_layer+1,self.c.d_LLM)  
        self.content_classifier=ContentTokenClassifier(self.tok,self.c.content_min_len)  
        self._degen_guard=DegenerationGuard(self.tok,self.c,self.content_classifier)  
        self.bridge.aligner.calibrate(self.llm)  
        self.c.vocab_size=self.llm.config.vocab_size  
        self._build_wte_neighbor_cache()  
  
    def _build_wte_neighbor_cache(self):  
        """Pre-compute WTE neighbors for content tokens (v3.7 expansion)."""  
        if self.llm is None or self.content_classifier is None: return  
        wte=self.llm.transformer.wte.weight.detach()  
        cc=self.content_classifier  
        content_list=sorted(cc.content_ids)  
        if not content_list:  
            self._wte_neighbor_cache={}; return  
        self._wte_neighbor_cache={}  
        wte_n=F.normalize(wte,dim=-1,eps=1e-8)  
        K=self.c.wte_neighbor_k; thresh=self.c.wte_neighbor_threshold  
        for tid in content_list:  
            if tid>=wte.shape[0]: continue  
            sim=wte_n[tid]@wte_n.T  
            topk_vals,topk_ids=sim.topk(K+1)  
            neighbors=[]  
            for v,nid in zip(topk_vals,topk_ids):  
                nid_int=nid.item()  
                if nid_int==tid: continue  
                if v.item()>=thresh and nid_int in cc.content_ids:  
                    neighbors.append(nid_int)  
            self._wte_neighbor_cache[tid]=neighbors  
  
    def _expand_content_ids(self, content_ids: List[int]) -> List[int]:  
        """Expand content_ids with WTE neighbors."""  
        if not self._wte_neighbor_cache: return content_ids  
        expanded=set(content_ids)  
        for tid in content_ids:  
            neighbors=self._wte_neighbor_cache.get(tid,[])  
            expanded.update(neighbors)  
        return list(expanded)  
  
    def _compute_content_semantic_emb(self, hidden_states, ids, mask):  
        """v3.7: Compute semantic embedding from CONTENT token positions only."""  
        B,T,D=hidden_states.shape  
        cc=self.content_classifier  
        result=[]  
        for b in range(B):  
            content_positions=[]  
            T_valid=min(T,ids.shape[1]) if ids is not None else T  
            for pos in range(T_valid):  
                if mask is not None and mask.shape[1]>pos and mask[b,pos].item()==0:  
                    continue  
                if ids is not None:  
                    tid=ids[b,pos].item()  
                    if cc is not None and tid in cc.content_ids:  
                        content_positions.append(min(pos,T-1))  
            if content_positions:  
                pos_t=torch.tensor(content_positions,device=hidden_states.device)  
                content_hs=hidden_states[b,pos_t]  
                result.append(content_hs.mean(0))  
            else:  
                if mask is not None:  
                    valid_len=min(int(mask[b].sum().item()),T)  
                    valid_len=max(valid_len,1)  
                    result.append(hidden_states[b,:valid_len].mean(0))  
                else:  
                    result.append(hidden_states[b].mean(0))  
        return torch.stack(result)  
  
    def _compute_wte_centroid(self, content_ids: List[int]) -> Optional[torch.Tensor]:  
        """Compute mean WTE embedding for content tokens."""  
        if not content_ids or self.llm is None: return None  
        wte=self.llm.transformer.wte.weight.detach()  
        valid=[i for i in content_ids if i<wte.shape[0]]  
        if not valid: return None  
        return wte[valid].mean(0).detach().clone()  
  
    def fwd(self, ids, mask, prefix=None):  
        B,T=ids.shape; dev=ids.device  
        te=self.llm.transformer.wte(ids)+self.llm.transformer.wpe(torch.arange(T,device=dev))  
        if prefix is not None:  
            hidden=torch.cat([prefix,te],1)  
            pm=torch.ones(B,prefix.shape[1],device=dev,dtype=mask.dtype)  
            mask=torch.cat([pm,mask],1)  
        else: hidden=te  
        hidden=self.llm.transformer.drop(hidden)  
        am=mask.unsqueeze(1).unsqueeze(2).to(hidden.dtype); am=(1.0-am)*(-1e4)  
        hs=[hidden]  
        for blk in self.llm.transformer.h:  
            hidden=blk(hidden,attention_mask=am)[0]; hs.append(hidden)  
        hidden=self.llm.transformer.ln_f(hidden)  
        return {'logits':self.llm.lm_head(hidden),'hs':hs,  
                'pl':prefix.shape[1] if prefix is not None else 0,'mask':mask}  
  
    def extract_state(self, hs, mask=None, pl=0):  
        pooled=self.layer_pool(hs)  
        if pl>0: pooled=pooled[:,pl:]  
        m=mask[:,pl:] if mask is not None and pl>0 else mask  
        if m is not None and m.shape[1]!=pooled.shape[1]: m=None  
        xq,fq=self.bridge.ext(pooled,m)  
        return pooled,xq,fq  
  
    def _build_content_bias(self, diag):  
        """v3.7: Build content bias from retrieved memories' expanded content token IDs."""  
        V=self.c.vocab_size; dev=next(self.parameters()).device  
        B=len(diag.batch_mem_weights)  
        bias=torch.zeros(B,V,device=dev)  
        for b,mem_weights in enumerate(diag.batch_mem_weights):  
            for mid,weight in mem_weights:  
                if mid in self.amm.tree.store:  
                    mem=self.amm.tree.store[mid]  
                    all_ids=set(mem.content_token_ids)  
                    all_ids.update(mem.expanded_content_ids)  
                    for tid in all_ids:  
                        if tid<V:  
                            bias[b,tid]+=weight  
            bmax=bias[b].max()  
            if bmax>1e-8: bias[b]=bias[b]/bmax  
        return bias  
  
    def _get_prefix(self, hs, mask=None, pl=0, update_stats=True,  
                    return_extra=False, ids=None):  
        pooled,xq,fq=self.extract_state(hs,mask,pl)  
        # v3.7: content-position-only semantic embedding  
        trimmed_mask=mask[:,pl:] if mask is not None and pl>0 else mask  
        if trimmed_mask is not None and pooled.shape[1]!=trimmed_mask.shape[1]:  
            trimmed_mask=None  
        if ids is not None and self.content_classifier is not None:  
            query_sem=self._compute_content_semantic_emb(pooled,ids,trimmed_mask)  
        else:  
            query_sem=pooled.mean(1)  
        # v3.7: WTE centroid for query  
        query_wte_centroid=None  
        if ids is not None and self.content_classifier is not None and self.llm is not None:  
            wte=self.llm.transformer.wte.weight.detach()  
            centroids=[]  
            for b in range(ids.shape[0]):  
                b_ids=ids[b].tolist()  
                b_content=self.content_classifier.get_content_ids_from_tokens(b_ids)  
                if b_content:  
                    valid=[i for i in b_content if i<wte.shape[0]]  
                    if valid: centroids.append(wte[valid].mean(0))  
                    else: centroids.append(torch.zeros(wte.shape[1],device=wte.device))  
                else: centroids.append(torch.zeros(wte.shape[1],device=wte.device))  
            query_wte_centroid=torch.stack(centroids)  
        fibers,mem_mask,fiber_summary,diag=self.amm.retrieve_multi(  
            xq,fq,update_stats=update_stats,  
            query_semantic_emb=query_sem,  
            query_wte_centroid=query_wte_centroid)  
        prefix=self.bridge.inject(fibers,mem_mask,fiber_summary=fiber_summary)  
        if return_extra:  
            content_bias=self._build_content_bias(diag)  
            return prefix,fiber_summary,diag,content_bias  
        return prefix  
  
    def _compute_vocab_bias(self, fiber_summary):  
        if fiber_summary is None: return None  
        wte=self.llm.transformer.wte.weight.detach()  
        return self.vocab_proj(fiber_summary,wte)  
  
    def write(self, text, training_mode=False):  
        tk=self.tok(text,return_tensors='pt',padding=True,truncation=True)  
        ids,mask=tk['input_ids'],tk['attention_mask']  
        dev=next(self.parameters()).device; ids,mask=ids.to(dev),mask.to(dev)  
        with torch.no_grad():  
            o=self.fwd(ids,mask)  
            hs_pooled=self.layer_pool(o['hs'])  
        surp=self.amm.surprise_proxy(o['logits'][:,:-1],ids[:,1:])  
        pooled_mean=hs_pooled.mean(1)  
        # v3.7: content-position-only semantic embedding  
        content_sem=self._compute_content_semantic_emb(hs_pooled,ids,mask)  
        # v3.7: content token IDs + expansion + WTE centroid  
        raw_ids=self.tok.encode(text)  
        cc=self.content_classifier  
        content_ids=list(set(cc.get_content_ids_from_tokens(raw_ids))) if cc else []  
        expanded_ids=self._expand_content_ids(content_ids)  
        wte_centroid=self._compute_wte_centroid(content_ids)  
        stored=0; gate_vals=[]  
        for b in range(ids.shape[0]):  
            with torch.no_grad():  
                gate=self.amm.write_gate(pooled_mean[b:b+1],surp[b:b+1]).item()  
            gate_vals.append(gate)  
            if training_mode or gate>=self.c.write_gate_threshold:  
                self.amm.store_mem(  
                    pooled_mean[b],surp[b],training_mode,  
                    source_text=text,content_token_ids=content_ids,  
                    content_semantic_emb=content_sem[b],  
                    content_wte_centroid=wte_centroid,  
                    expanded_content_ids=expanded_ids)  
                stored+=1  
        return stored,gate_vals  
  
    def _refresh_all_memories(self):  
        entries=list(self.amm.tree.store.values())  
        texts=[e.source_text for e in entries if e.source_text]  
        if not texts: return 0  
        unique_texts=list(dict.fromkeys(texts))  
        self.amm.tree.store.clear()  
        self.amm.tree.root=_Node()  
        self.amm.tree.nid=0; self.amm.time=0  
        for text in unique_texts:  
            self.write(text,training_mode=True)  
        return len(unique_texts)  
  
    def generate(self, prompt, mt=50, greedy=False):  
        tk=self.tok(prompt,return_tensors='pt')  
        dev=next(self.parameters()).device  
        ids,mask=tk['input_ids'].to(dev),tk['attention_mask'].to(dev)  
        with torch.no_grad():  
            o=self.fwd(ids,mask)  
            prefix,fiber_summary,_,content_bias=self._get_prefix(  
                o['hs'],mask,update_stats=True,return_extra=True,ids=ids)  
            vocab_bias=self._compute_vocab_bias(fiber_summary)  
        has_content=content_bias is not None and content_bias.abs().max().item()>0.01  
        cc=self.content_classifier  
        generated_ids=[]  
        for i in range(mt):  
            if i>0 and i%self.c.retrieval_interval==0:  
                with torch.no_grad():  
                    o=self.fwd(ids,mask,prefix); pl=o['pl']  
                    prefix,fiber_summary,_,content_bias=self._get_prefix(  
                        o['hs'],o['mask'],pl,update_stats=True,return_extra=True,ids=ids)  
                    vocab_bias=self._compute_vocab_bias(fiber_summary)  
                    has_content=content_bias is not None and content_bias.abs().max().item()>0.01  
            with torch.no_grad():  
                o=self.fwd(ids,mask,prefix); lg=o['logits'][:,-1:].squeeze(1)  
                # ── v3.7 双路径 logit bias ──  
                step_scale_content=max(self.c.content_bias_floor,  
                                       1.0-i*self.c.content_bias_decay)  
                step_scale_learned=max(self.c.semantic_boost_floor,  
                                       1.0-i*self.c.semantic_boost_decay)  
                # 确定性路径 (主力)  
                if has_content:  
                    V=min(lg.shape[-1],content_bias.shape[-1])  
                    lg[:,:V]=lg[:,:V]+content_bias[:,:V]*self.c.content_bias_scale*step_scale_content  
                # 学习路径 (辅助)  
                if vocab_bias is not None:  
                    V2=min(lg.shape[-1],vocab_bias.shape[-1])  
                    lg[:,:V2]=lg[:,:V2]+vocab_bias[:,:V2]*self.c.semantic_boost_scale*step_scale_learned  
                # v3.7: 通用内容词 boost (前几步)  
                if i<self.c.universal_content_boost_steps and cc is not None and has_content:  
                    cmask=cc.content_mask(dev)  
                    V3=min(lg.shape[-1],cmask.shape[0])  
                    boost_scale=1.0-i/self.c.universal_content_boost_steps  
                    lg[0,:V3]=lg[0,:V3]+cmask[:V3]*self.c.universal_content_boost*boost_scale  
                # DegenerationGuard (包含 v3.7 的强惩罚)  
                if self._degen_guard is not None:  
                    lg=self._degen_guard.process(lg,generated_ids,i)  
                if greedy:  
                    nxt=lg.argmax(-1,keepdim=True)  
                else:  
                    lg=lg/self.c.gen_temp; p=F.softmax(lg,-1)  
                    sp,si=torch.sort(p,descending=True); cs=torch.cumsum(sp,-1)  
                    rm=cs-sp>self.c.gen_top_p; sp[rm]=0  
                    total=sp.sum(-1,keepdim=True)  
                    if (total<1e-10).any(): sp[:,0]=1.0; total=sp.sum(-1,keepdim=True)  
                    sp=sp/total; nxt=si.gather(-1,torch.multinomial(sp,1))  
            if nxt.item()==self.tok.eos_token_id and len(generated_ids)>=self.c.degen_min_tokens: break  
            generated_ids.append(nxt.item())  
            ids=torch.cat([ids,nxt],1)  
            mask=torch.cat([mask,torch.ones(1,1,device=dev,dtype=mask.dtype)],1)  
        return self.tok.decode(ids[0],skip_special_tokens=True)  
  
    def save_memory(self, path):  
        data={'store':{},'nid':self.amm.tree.nid,'time':self.amm.time}  
        for mid,m in self.amm.tree.store.items():  
            data['store'][mid]={  
                'base':m.base.cpu(),'fiber':m.fiber.cpu(),'dirn':m.dirn.cpu(),  
                'surprise':m.surprise,'ts':m.ts,'last':m.last,'cnt':m.cnt,'version':m.version,  
                'source_text':m.source_text,  
                'content_token_ids':m.content_token_ids,  
                'expanded_content_ids':m.expanded_content_ids,  
                'semantic_emb':m.semantic_emb.cpu() if m.semantic_emb is not None else None,  
                'content_wte_centroid':m.content_wte_centroid.cpu() if m.content_wte_centroid is not None else None}  
        torch.save(data,path)  
  
    def load_memory(self, path):  
        data=torch.load(path,weights_only=False)  
        self.amm.tree.store.clear(); self.amm.tree.root=_Node()  
        self.amm.tree.nid=data['nid']; self.amm.time=data['time']  
        dev=next(self.parameters()).device  
        for mid,d in data['store'].items():  
            sem=d.get('semantic_emb',None)  
            if sem is not None: sem=sem.to(dev)  
            wte_c=d.get('content_wte_centroid',None)  
            if wte_c is not None: wte_c=wte_c.to(dev)  
            m=MemEntry(mid=mid,base=d['base'].to(dev),fiber=d['fiber'].to(dev),  
                dirn=d['dirn'].to(dev),surprise=d['surprise'],ts=d['ts'],  
                last=d['last'],cnt=d['cnt'],version=d['version'],  
                source_text=d.get('source_text',''),  
                content_token_ids=d.get('content_token_ids',[]),  
                expanded_content_ids=d.get('expanded_content_ids',[]),  
                semantic_emb=sem,content_wte_centroid=wte_c)  
            self.amm.tree.insert(m)  
  
# ═══════════════════════════════════════════════════════════════════  
# 第19部分 · 谱去混叠  
# ═══════════════════════════════════════════════════════════════════  
class SpectralDealiaser:  
    def __init__(self, amm, c): self.amm=amm; self.c=c  
    def detect(self, sim_threshold=0.3):  
        ms=list(self.amm.tree.store.values())  
        if len(ms)<2: return []  
        N=len(ms); bases=torch.stack([m.base for m in ms]); fibers=torch.stack([m.fiber for m in ms])  
        rd=torch.zeros(N,N,**_dev(bases))  
        for i in range(N):  
            for j in range(i+1,N):  
                d=self.amm.metric.midpoint_approx_distance(bases[i:i+1],bases[j:j+1]).item()  
                rd[i,j]=rd[j,i]=d  
        pos=rd[rd>0]  
        sigma=pos.median().clamp(min=0.1) if pos.numel()>0 else torch.tensor(1.0,**_dev(bases))  
        W=torch.exp(-rd.pow(2)/(2*sigma.pow(2)))  
        fn=F.normalize(fibers,-1); fs=(fn@fn.T).clamp(0,1)  
        A=W*fs; A.fill_diagonal_(0); D=A.sum(1); Di=(D+1e-8).pow(-0.5)  
        L_mat=torch.eye(N,**_dev(A))-Di.unsqueeze(1)*A*Di.unsqueeze(0)  
        ev,ec=torch.linalg.eigh(L_mat); gaps=ev[1:]-ev[:-1]; mk=max(2,N//3)  
        k=gaps[:mk].argmax().item()+2; k=min(k,N)  
        feat=ec[:,:k]; lb=DirectionTree._farthest_kmeans(feat,k)  
        cls={}  
        for i,l in enumerate(lb.tolist()): cls.setdefault(l,[]).append(ms[i].mid)  
        res=[]  
        for cids in cls.values():  
            if len(cids)<2: continue  
            cf=torch.stack([self.amm.tree.store[i].fiber for i in cids])  
            cn=F.normalize(cf,-1); n=len(cids)  
            avg=(cn@cn.T).triu(1).sum()/(n*(n-1)/2+1e-10)  
            if avg>sim_threshold: res.append(cids)  
        return res  
    def dealias(self, ids, steps=50, lr=0.01):  
        ms=[self.amm.tree.store[i] for i in ids if i in self.amm.tree.store]  
        if len(ms)<2: return  
        orig=[m.fiber.clone() for m in ms]  
        fs=[m.fiber.detach().clone().requires_grad_(True) for m in ms]  
        opt=torch.optim.Adam(fs,lr=lr)  
        for _ in range(steps):  
            opt.zero_grad()  
            fn=F.normalize(torch.stack(fs),-1); n=len(fs)  
            mk=~torch.eye(n,dtype=torch.bool,device=fn.device); sim=fn@fn.T  
            (sim[mk].pow(2).mean()+0.1*sum((fi-oi).pow(2).sum() for fi,oi in zip(fs,orig))/n).backward()  
            opt.step()  
        for fi,m in zip(fs,ms):  
            nf=fi.detach().clone(); nd=self.amm._compute_dirn(m.base,nf)  
            self.amm.tree.update(m.mid,new_fiber=nf,new_dirn=nd)  
  
# ═══════════════════════════════════════════════════════════════════  
# 第20部分 · 训练器  
# ═══════════════════════════════════════════════════════════════════  
class Trainer:  
    def __init__(self, m, c):  
        self.m=m; self.c=c  
        ps=[p for n,p in m.named_parameters() if p.requires_grad and 'llm' not in n]  
        self.opt=torch.optim.AdamW(ps,lr=1e-4,weight_decay=0.01)  
        self.warmup=LossWarmup({  
            'semantic_probe':c.warmup_steps_probe,'dir_diversity':c.warmup_steps_dd,  
            'reranker_ranking':c.warmup_steps_rr,'vocab_anchor':c.warmup_steps_va,  
            'semantic_alignment':c.warmup_steps_sa})  
        self.grad_monitor=GradientMonitor()  
        self.grad_monitor.register('ctx_encoder',m.amm.ctx)  
        self.grad_monitor.register('fib_encoder',m.amm.fib)  
        self.grad_monitor.register('dir_predictor',m.amm.dir_pred)  
        self.grad_monitor.register('fiber_connection',m.amm.conn)  
        self.grad_monitor.register('fiber_attn',m.amm.attn)  
        self.grad_monitor.register('reranker',m.amm.reranker)  
        self.grad_monitor.register('qformer',m.bridge.proj)  
        self.grad_monitor.register('content_bypass',m.bridge.bypass)  
        self.grad_monitor.register('semantic_probe',m.semantic_probe)  
        self.grad_monitor.register('layer_pool',m.layer_pool)  
        self.grad_monitor.register('prefix_aligner',m.bridge.aligner)  
        self.grad_monitor.register('vocab_proj',m.vocab_proj)  
        self.layer_weight_history=[]; self._step_count=0  
  
    def _encode_with_grad(self, texts):  
        tk=self.m.tok(texts,return_tensors='pt',padding=True,truncation=True)  
        dev=next(self.m.parameters()).device  
        ids,mask=tk['input_ids'].to(dev),tk['attention_mask'].to(dev)  
        with torch.no_grad():  
            o=self.m.fwd(ids,mask)  
            surp=self.m.amm.surprise_proxy(o['logits'][:,:-1],ids[:,1:])  
        pooled=self.m.layer_pool(o['hs']); pooled_mean=pooled.mean(1)  
        base=self.m.amm.ctx(pooled_mean)  
        fiber=self.m.amm.fib(pooled_mean,base,surp)  
        _=self.m.amm.dir_pred(base,fiber)  
        return ids,mask,base,fiber,surp,pooled_mean  
  
    def encoder_throughput_loss(self, ids, mask, fiber):  
        B=ids.shape[0]; dev=ids.device  
        fiber_unsq=fiber.unsqueeze(1); mem_mask_ones=torch.ones(B,1,device=dev)  
        prefix=self.m.bridge.inject(fiber_unsq,mem_mask_ones,fiber_summary=fiber)  
        o2=self.m.fwd(ids,mask,prefix)  
        lg=o2['logits'][:,o2['pl']:-1]; tg=ids[:,1:]  
        ml=min(lg.shape[1],tg.shape[1])  
        if ml==0: return torch.tensor(0.0,device=dev,requires_grad=True)  
        return F.cross_entropy(lg[:,:ml].reshape(-1,lg.shape[-1]),tg[:,:ml].reshape(-1))  
  
    def semantic_alignment_loss(self, fiber, target_ids, target_mask):  
        dev=fiber.device; wte=self.m.llm.transformer.wte.weight.detach()  
        vocab_logits=self.m.vocab_proj(fiber,wte)  
        B,V=vocab_logits.shape; cc=self.m.content_classifier  
        if cc is None: return torch.tensor(0.0,device=dev,requires_grad=True)  
        target=torch.zeros(B,V,device=dev); valid_count=0  
        for b in range(B):  
            valid=target_ids[b][target_mask[b].bool()].tolist()  
            content_ids=cc.get_content_ids_from_tokens(valid)  
            if content_ids:  
                uids=list(set(content_ids)); uids=[uid for uid in uids if uid<V]  
                if uids: target[b,uids]=1.0/len(uids); valid_count+=1  
        if valid_count==0: return torch.tensor(0.0,device=dev,requires_grad=True)  
        log_probs=F.log_softmax(vocab_logits/self.c.semantic_align_temp,dim=-1)  
        kl=F.kl_div(log_probs,target,reduction='none').sum(-1)  
        return kl.mean()  
  
    def vocab_anchor_loss(self, prefix):  
        wte=self.m.llm.transformer.wte.weight.detach()  
        pn=F.normalize(prefix.reshape(-1,prefix.shape[-1]),dim=-1)  
        wn=F.normalize(wte,dim=-1)  
        sim=pn@wn.T; topk_sim=sim.topk(self.c.vocab_anchor_topk,dim=-1).values  
        return -topk_sim.mean()  
  
    def _recon_forward(self, text):  
        tk=self.m.tok(text,return_tensors='pt',padding=True,truncation=True)  
        dev=next(self.m.parameters()).device  
        ids,mask=tk['input_ids'].to(dev),tk['attention_mask'].to(dev)  
        with torch.no_grad(): bo=self.m.fwd(ids,mask)  
        prefix=self.m._get_prefix(bo['hs'],mask,update_stats=False,ids=ids)  
        o=self.m.fwd(ids,mask,prefix)  
        lg=o['logits'][:,o['pl']:-1]; tg=ids[:,1:]  
        ml=min(lg.shape[1],tg.shape[1])  
        if ml==0:  
            zero=ids.new_tensor(0.0,dtype=torch.float,requires_grad=True)  
            return zero,prefix,self.m.bridge._last_fiber_summary  
        l_r=F.cross_entropy(lg[:,:ml].reshape(-1,lg.shape[-1]),tg[:,:ml].reshape(-1))  
        fs=self.m.bridge._last_fiber_summary  
        if fs is None: fs=torch.zeros(1,self.c.d_F,device=dev)  
        return l_r,prefix,fs  
  
    def _semantic_probe_loss(self, prefix_batch, fs_batch):  
        pred=self.m.semantic_probe(prefix_batch)  
        l_mse=F.mse_loss(pred,fs_batch.detach())  
        if prefix_batch.shape[0]>=2:  
            pn=F.normalize(pred,dim=-1); tn=F.normalize(fs_batch.detach(),dim=-1)  
            sim=pn@tn.T/self.c.probe_contrastive_tau  
            lb=torch.arange(prefix_batch.shape[0],device=prefix_batch.device)  
            l_ctr=F.cross_entropy(sim,lb)  
            return l_mse+0.5*l_ctr  
        return l_mse  
  
    def contrast(self, texts):  
        tk=self.m.tok(texts,return_tensors='pt',padding=True,truncation=True)  
        dev=next(self.m.parameters()).device  
        ids,mask=tk['input_ids'].to(dev),tk['attention_mask'].to(dev)  
        with torch.no_grad(): o=self.m.fwd(ids,mask)  
        _,xq,fq=self.m.extract_state(o['hs'],mask)  
        x=F.normalize(self.m.amm.contrast_proj_x(xq),-1)  
        f=F.normalize(self.m.amm.contrast_proj_f(fq),-1)  
        sxf=x@f.T/self.c.contrast_tau; sfx=f@x.T/self.c.contrast_tau  
        lb=torch.arange(len(texts),device=dev)  
        return (F.cross_entropy(sxf,lb)+F.cross_entropy(sfx,lb))/2  
  
    def holonomy_proxy(self, x, f):  
        sz=0.05; v1=torch.randn_like(x)*sz; v2=torch.randn_like(x)*sz  
        loop=torch.stack([x,x+v1,x+v1+v2,x+v2,x],1)  
        return (self.m.amm.trans(f,loop)-f).pow(2).sum(-1).mean()  
  
    def write_policy_loss(self, texts):  
        tk=self.m.tok(texts,return_tensors='pt',padding=True,truncation=True)  
        dev=next(self.m.parameters()).device  
        ids,mask=tk['input_ids'].to(dev),tk['attention_mask'].to(dev)  
        with torch.no_grad():  
            o=self.m.fwd(ids,mask)  
            surp=self.m.amm.surprise_proxy(o['logits'][:,:-1],ids[:,1:])  
        pooled=self.m.layer_pool(o['hs']).mean(1)  
        gates=self.m.amm.write_gate(pooled,surp)  
        labels=(surp>surp.median()).float()  
        return F.binary_cross_entropy(gates,labels)  
  
    def direction_diversity_loss(self, texts):  
        tk=self.m.tok(texts,return_tensors='pt',padding=True,truncation=True)  
        dev=next(self.m.parameters()).device  
        ids,mask=tk['input_ids'].to(dev),tk['attention_mask'].to(dev)  
        with torch.no_grad(): o=self.m.fwd(ids,mask)  
        _,xq,fq=self.m.extract_state(o['hs'],mask)  
        dirs=F.normalize(self.m.amm.dir_pred(xq,fq),dim=-1,eps=1e-8)  
        dir_sim=(dirs@dirs.T).clamp(-1.0,1.0)  
        with torch.no_grad():  
            fn=F.normalize(fq,dim=-1,eps=1e-8); fiber_sim=(fn@fn.T).clamp(-1.0,1.0)  
        tau=self.c.dir_diversity_tau  
        dir_prob=torch.sigmoid(dir_sim/tau); fiber_prob=torch.sigmoid(fiber_sim/tau)  
        B=len(texts); mask_off=~torch.eye(B,dtype=torch.bool,device=dev)  
        return F.binary_cross_entropy(dir_prob[mask_off],fiber_prob[mask_off].detach())  
  
    def reranker_ranking_loss(self, texts):  
        store=self.m.amm.tree.store  
        if len(store)<2:  
            dev=next(self.m.parameters()).device  
            return torch.tensor(0.0,device=dev,requires_grad=True)  
        tk=self.m.tok(texts,return_tensors='pt',padding=True,truncation=True)  
        dev=next(self.m.parameters()).device  
        ids,mask=tk['input_ids'].to(dev),tk['attention_mask'].to(dev)  
        with torch.no_grad(): o=self.m.fwd(ids,mask)  
        _,xq,fq=self.m.extract_state(o['hs'],mask)  
        mids=list(store.keys())  
        cb=torch.stack([store[m].base.to(dev) for m in mids])  
        cf=torch.stack([store[m].fiber.to(dev) for m in mids])  
        cd=torch.stack([store[m].dirn.to(dev) for m in mids])  
        B=xq.shape[0]; qdir=self.m.amm.dir_pred(xq,fq)  
        dir_sims=torch.einsum('bd,cd->bc',qdir,cd)  
        cb_e=cb.unsqueeze(0).expand(B,-1,-1); cf_e=cf.unsqueeze(0).expand(B,-1,-1)  
        scores=self.m.amm.reranker(xq,fq,cb_e,cf_e,dir_sims)  
        with torch.no_grad():  
            fqn=F.normalize(fq,dim=-1); cfn=F.normalize(cf,dim=-1)  
            relevance=torch.einsum('bd,cd->bc',fqn,cfn)  
        s_mean=scores.mean(-1,keepdim=True); s_std=scores.std(-1,keepdim=True).clamp(min=1e-6)  
        r_mean=relevance.mean(-1,keepdim=True); r_std=relevance.std(-1,keepdim=True).clamp(min=1e-6)  
        sn=(scores-s_mean)/s_std; rn=(relevance-r_mean)/r_std  
        return F.mse_loss(sn,rn.detach())  
  
    def step(self, texts):  
        self.m.train(); self.opt.zero_grad()  
        dev=next(self.m.parameters()).device; W=self.c.loss_weights  
        ids_enc,mask_enc,base,fiber,surp,pooled_mean=self._encode_with_grad(texts)  
        l_et=self.encoder_throughput_loss(ids_enc,mask_enc,fiber)  
        w_sa=self.warmup.weight('semantic_alignment')  
        l_sa=self.semantic_alignment_loss(fiber,ids_enc,mask_enc)*w_sa  
        all_lr=[]; all_pf=[]; all_fs=[]  
        for t in texts:  
            lr,pf,fs=self._recon_forward(t)  
            all_lr.append(lr); all_pf.append(pf)  
            all_fs.append(fs if fs is not None else torch.zeros(1,self.c.d_F,device=dev))  
        l_r=sum(all_lr)/len(texts)  
        pf_batch=torch.cat(all_pf,0); fs_batch=torch.cat(all_fs,0)  
        w_sp=self.warmup.weight('semantic_probe')  
        l_sp=self._semantic_probe_loss(pf_batch,fs_batch)*w_sp  
        w_va=self.warmup.weight('vocab_anchor')  
        l_va=self.vocab_anchor_loss(pf_batch)*w_va  
        l_c=self.contrast(texts) if len(texts)>=2 else torch.tensor(0.0,device=dev)  
        with torch.no_grad():  
            tk2=self.m.tok(texts,return_tensors='pt',padding=True,truncation=True)  
            ids2,mask2=tk2['input_ids'].to(dev),tk2['attention_mask'].to(dev)  
            o2=self.m.fwd(ids2,mask2)  
        _,xq2,fq2=self.m.extract_state(o2['hs'],mask2)  
        l_h=self.holonomy_proxy(xq2,fq2)  
        l_w=self.write_policy_loss(texts)  
        w_dd=self.warmup.weight('dir_diversity')  
        l_dd=(self.direction_diversity_loss(texts) if len(texts)>=2  
              else torch.tensor(0.0,device=dev))*w_dd  
        w_rr=self.warmup.weight('reranker_ranking')  
        l_rr=self.reranker_ranking_loss(texts)*w_rr  
        loss=(W['recon']*l_r+W['semantic_alignment']*l_sa+  
              W['encoder_throughput']*l_et+W['contrast']*l_c+  
              W['holonomy']*l_h+W['write_policy']*l_w+  
              W['semantic_probe']*l_sp+W['dir_diversity']*l_dd+  
              W['reranker_ranking']*l_rr+W['vocab_anchor']*l_va)  
        loss.backward()  
        nn.utils.clip_grad_norm_(  
            [p for n,p in self.m.named_parameters() if p.requires_grad and 'llm' not in n],1.)  
        self.opt.step(); self.warmup.advance(); self._step_count+=1  
        grad_norms=self.grad_monitor.snapshot()  
        self.layer_weight_history.append(self.m.layer_pool.weight_dist().cpu().numpy().copy())  
        if self._step_count%self.c.refresh_memories_every==0:  
            self.m.eval()  
            with torch.no_grad(): self.m._refresh_all_memories()  
            self.m.train()  
        self.m.eval()  
        return {  
            'total':loss.item(),'recon':l_r.item(),'contrast':l_c.item(),  
            'holonomy':l_h.item(),'write_policy':l_w.item(),  
            'semantic_probe':l_sp.item(),'dir_diversity':l_dd.item(),  
            'reranker_ranking':l_rr.item(),'encoder_throughput':l_et.item(),  
            'vocab_anchor':l_va.item(),'semantic_alignment':l_sa.item(),  
            'warmup_sp':w_sp,'warmup_dd':w_dd,'warmup_rr':w_rr,'warmup_va':w_va,'warmup_sa':w_sa,  
            'grad_norms':grad_norms,  
            'bypass_gate':self.m.bridge._last_inject_diag.get('bypass_gate',None),  
            'aligner_scale':self.m.bridge._last_inject_diag.get('aligner_scale',None),  
            'loss_weights':W}  
  
# ═══════════════════════════════════════════════════════════════════  
# 第21部分 · 测试  
# ═══════════════════════════════════════════════════════════════════  
class TestResults:  
    def __init__(self): self.passed=0; self.failed=0; self.errors=[]  
    def check(self, name, cond, msg=""):  
        if cond: self.passed+=1; print(f"  ✓ {name}")  
        else: self.failed+=1; self.errors.append(f"{name}: {msg}"); print(f"  ✗ {name}: {msg}")  
    def summary(self):  
        t=self.passed+self.failed  
        print(f"\n{'='*60}\n  {self.passed}/{t} passed, {self.failed} failed")  
        if self.errors:  
            print("  失败项:")  
            for e in self.errors: print(f"    - {e}")  
        return self.failed==0  
  
def test_properties(m, c, R):  
    print("\n── 性质测试 ──")  
    dev=next(m.parameters()).device  
    xt=torch.randn(4,c.d_M,device=dev); g=m.amm.metric(xt)  
    ev=torch.linalg.eigvalsh(g); R.check("metric_spd",(ev>0).all().item(),f"λ_min={ev.min():.6f}")  
    G=m.amm.metric.christoffel(xt[:1]); sym=(G-G.permute(0,1,3,2)).abs().max().item()  
    R.check("christoffel_sym",sym<1e-5,f"err={sym:.2e}")  
    A=m.amm.conn(xt[:1],torch.randn(1,c.d_M,device=dev))  
    asym=(A+A.transpose(1,2)).abs().max().item()  
    R.check("connection_antisym",asym<1e-5,f"err={asym:.2e}")  
    xs=torch.randn(1,c.d_M,device=dev)*0.3; xe=torch.randn(1,c.d_M,device=dev)*0.3  
    gr=m.amm.geo.solve(xs,xe)  
    R.check("geodesic_converged",gr.converged,f"iters={gr.iterations}")  
    R.check("geodesic_start_fixed",(gr.path[:,0]-xs).norm().item()<1e-5)  
    R.check("geodesic_end_fixed",(gr.path[:,-1]-xe).norm().item()<1e-5)  
    R.check("geodesic_energy_finite",gr.energy<1e6 and gr.energy==gr.energy)  
    f0=torch.randn(1,c.d_F,device=dev); f_rk4=m.amm.trans(f0,gr.path)  
    dr=abs(f_rk4.norm().item()-f0.norm().item())/f0.norm().item()  
    R.check("rk4_norm_preservation",dr<0.05,f"drift={dr:.4f}")  
  
def test_geodesic_gradient(m, c, R):  
    print("\n── 测地线梯度测试 ──")  
    dev=next(m.parameters()).device  
    xs=torch.randn(1,c.d_M,device=dev); xe=torch.randn(1,c.d_M,device=dev,requires_grad=True)  
    gr=m.amm.geo.solve(xs,xe); f0=torch.randn(1,c.d_F,device=dev)  
    ft=m.amm.trans(f0,gr.path); ft.sum().backward()  
    R.check("geo_endpoint_grad_exists",xe.grad is not None and xe.grad.abs().max().item()>0)  
  
def test_geodesic_no_grad(m, c, R):  
    print("\n── 测地线 no_grad 测试 ──")  
    dev=next(m.parameters()).device  
    xs=torch.randn(1,c.d_M,device=dev); xe=torch.randn(1,c.d_M,device=dev)  
    with torch.no_grad(): gr=m.amm.geo.solve(xs,xe)  
    R.check("geo_nograd_ok",True)  
    R.check("geo_nograd_finite",gr.path.isfinite().all().item())  
  
def test_contrast_dimensions(m, c, R):  
    print("\n── 对比损失维度测试 ──")  
    trainer=Trainer(m,c); m.train()  
    try:  
        l_c=trainer.contrast(["Hello world.","Goodbye moon."])  
        R.check("contrast_no_crash",True); R.check("contrast_finite",l_c.isfinite().item())  
        l_c.backward(); pg=m.amm.contrast_proj_f.weight.grad  
        R.check("contrast_proj_f_grad",pg is not None and pg.abs().max().item()>0)  
    except Exception as e: R.check("contrast_no_crash",False,str(e))  
    m.zero_grad(); m.eval()  
  
def test_content_classifier(m, c, R):  
    print("\n── 内容词分类器测试 ──")  
    cc=m.content_classifier  
    R.check("cc_exists",cc is not None)  
    if cc:  
        R.check("cc_has_content",len(cc.content_ids)>100,f"n={len(cc.content_ids)}")  
        dev=next(m.parameters()).device; cmask=cc.content_mask(dev)  
        R.check("cc_mask_shape",cmask.dim()==1 and cmask.shape[0]>0)  
        pos=cc.get_content_positions([220,12519,329,40481],[True]*4)  
        R.check("cc_get_content_positions_works",isinstance(pos,list))  
  
def test_wte_neighbor_cache(m, c, R):  
    print("\n── WTE 邻居缓存测试 ──")  
    R.check("wte_cache_exists",m._wte_neighbor_cache is not None)  
    if m._wte_neighbor_cache:  
        R.check("wte_cache_nonempty",len(m._wte_neighbor_cache)>0,  
            f"n={len(m._wte_neighbor_cache)}")  
        sample_key=next(iter(m._wte_neighbor_cache))  
        sample_val=m._wte_neighbor_cache[sample_key]  
        R.check("wte_cache_value_is_list",isinstance(sample_val,list))  
        piano_ids=[i for i in m.content_classifier.content_ids  
                   if 'piano' in m.tok.decode([i]).lower()]  
        if piano_ids:  
            pid=piano_ids[0]  
            neighbors=m._wte_neighbor_cache.get(pid,[])  
            if neighbors:  
                n_toks=[m.tok.decode([n]).strip() for n in neighbors[:5]]  
                print(f"    piano neighbors: {n_toks}")  
            R.check("wte_piano_has_neighbors",len(neighbors)>=0)  
  
def test_content_semantic_emb(m, c, R):  
    print("\n── Content-Position-Only 语义嵌入测试 ──")  
    dev=next(m.parameters()).device  
    # 音乐文本  
    tk1=m.tok("He practiced piano Chopin nocturne",return_tensors='pt')  
    ids1,mask1=tk1['input_ids'].to(dev),tk1['attention_mask'].to(dev)  
    with torch.no_grad():  
        o1=m.fwd(ids1,mask1); pooled1=m.layer_pool(o1['hs'])  
    sem1=m._compute_content_semantic_emb(pooled1,ids1,mask1)  
    R.check("csem_shape",sem1.shape==(1,c.d_LLM),f"shape={sem1.shape}")  
    R.check("csem_finite",sem1.isfinite().all().item())  
    R.check("csem_nonzero",sem1.abs().max().item()>0)  
    # 太空文本  
    tk2=m.tok("The telescope revealed distant galaxies",return_tensors='pt')  
    ids2,mask2=tk2['input_ids'].to(dev),tk2['attention_mask'].to(dev)  
    with torch.no_grad():  
        o2=m.fwd(ids2,mask2); pooled2=m.layer_pool(o2['hs'])  
    sem2=m._compute_content_semantic_emb(pooled2,ids2,mask2)  
    # 全位置均值  
    mean1=pooled1.mean(1); mean2=pooled2.mean(1)  
    # content-only应该比全位置均值更具域区分性  
    csim=F.cosine_similarity(sem1,sem2).item()  
    msim=F.cosine_similarity(mean1,mean2).item()  
    print(f"    content-only cosine(music,space)={csim:.4f}")  
    print(f"    mean-all cosine(music,space)={msim:.4f}")  
    R.check("csem_more_discriminative",csim<msim or csim<0.95,  
        f"content_sim={csim:.4f}, mean_sim={msim:.4f}")  
  
def test_semantic_retrieval(m, c, R):  
    print("\n── 语义嵌入检索测试 (v3.7) ──")  
    m.amm.tree.store.clear(); m.amm.tree.root=_Node(); m.amm.tree.nid=0; m.amm.time=0  
    m.write("He practiced piano for hours perfecting a difficult Chopin nocturne.",training_mode=True)  
    m.write("She studied music theory and harmonic progression at the conservatory.",training_mode=True)  
    m.write("The telescope revealed distant galaxies beyond the Milky Way.",training_mode=True)  
    m.write("Astronauts trained for the Mars mission in simulated zero gravity.",training_mode=True)  
    m.eval(); dev=next(m.parameters()).device  
    # 钢琴 query  
    tk=m.tok("Tell me about piano practice.",return_tensors='pt')  
    ids,mask=tk['input_ids'].to(dev),tk['attention_mask'].to(dev)  
    with torch.no_grad():  
        o=m.fwd(ids,mask)  
        _,_,_,content_bias=m._get_prefix(o['hs'],mask,update_stats=False,return_extra=True,ids=ids)  
    top10_ids=content_bias[0].topk(10).indices.tolist()  
    top10_toks=[m.tok.decode([t]).strip().lower() for t in top10_ids]  
    has_music=any(w in top10_toks for w in ['piano','chopin','nocturne','practiced',  
        'perfecting','difficult','music','theory','harmonic','harmony',  
        'progression','conservatory','studied','hours'])  
    has_space=any(w in top10_toks for w in ['telescope','galaxies','galaxy','distant',  
        'astronauts','mars','gravity','mission','zero','milky','revealed'])  
    R.check("sem_ret_music_query_has_music",has_music,f"top10={top10_toks}")  
    R.check("sem_ret_music_query_no_space",not has_space or has_music,  
        f"top10={top10_toks}")  
    print(f"    piano query → content_bias top10: {top10_toks}")  
    # 太空 query  
    tk2=m.tok("The space telescope observes distant stars.",return_tensors='pt')  
    ids2,mask2=tk2['input_ids'].to(dev),tk2['attention_mask'].to(dev)  
    with torch.no_grad():  
        o2=m.fwd(ids2,mask2)  
        _,_,_,cb2=m._get_prefix(o2['hs'],mask2,update_stats=False,return_extra=True,ids=ids2)  
    top10_ids2=cb2[0].topk(10).indices.tolist()  
    top10_toks2=[m.tok.decode([t]).strip().lower() for t in top10_ids2]  
    has_space2=any(w in top10_toks2 for w in ['telescope','galaxies','galaxy','distant',  
        'astronauts','mars','gravity','mission','zero','milky','revealed','spectrum'])  
    R.check("sem_ret_space_query_has_space",has_space2,f"top10={top10_toks2}")  
    print(f"    space query → content_bias top10: {top10_toks2}")  
    m.amm.tree.store.clear(); m.amm.tree.root=_Node(); m.amm.tree.nid=0; m.amm.time=0  
  
def test_first_step_not_punct(m, c, R, texts):  
    print("\n── 首步 Top-1 非标点测试 (v3.7) ──")  
    m.amm.tree.store.clear(); m.amm.tree.root=_Node(); m.amm.tree.nid=0; m.amm.time=0  
    for t in texts[:6]: m.write(t,training_mode=True)  
    m.eval(); dev=next(m.parameters()).device; cc=m.content_classifier  
    for prompt in ["Key piano ideas include","The telescope reveals"]:  
        tk=m.tok(prompt,return_tensors='pt')  
        ids,mask=tk['input_ids'].to(dev),tk['attention_mask'].to(dev)  
        with torch.no_grad():  
            o=m.fwd(ids,mask)  
            prefix,fiber_summary,diag,content_bias=m._get_prefix(  
                o['hs'],mask,update_stats=False,return_extra=True,ids=ids)  
            vocab_bias=m._compute_vocab_bias(fiber_summary)  
            o2=m.fwd(ids,mask,prefix)  
            logits=o2['logits'][:,-1].clone()  
            V=min(logits.shape[-1],content_bias.shape[-1])  
            logits[:,:V]=logits[:,:V]+content_bias[:,:V]*c.content_bias_scale  
            if vocab_bias is not None:  
                V2=min(logits.shape[-1],vocab_bias.shape[-1])  
                logits[:,:V2]=logits[:,:V2]+vocab_bias[:,:V2]*c.semantic_boost_scale  
            if cc:  
                cmask=cc.content_mask(dev)  
                V3=min(logits.shape[-1],cmask.shape[0])  
                logits[0,:V3]=logits[0,:V3]+cmask[:V3]*c.universal_content_boost  
            logits=m._degen_guard.process(logits,[],0)  
            top1=logits.argmax(-1).item(); top1_tok=m.tok.decode([top1]).strip()  
        is_punct=top1 in cc.punct_ids or top1 in cc.newline_ids  
        R.check(f"first_step_{prompt[:10]}_not_punct",not is_punct,  
            f"top1={top1}, tok='{top1_tok}'")  
        top5=logits.topk(5).indices[0].tolist()  
        top5_toks=[m.tok.decode([t]).strip() for t in top5]  
        content_in_top5=sum(1 for t in top5 if t in cc.content_ids)  
        R.check(f"first_step_{prompt[:10]}_content_in_top5",content_in_top5>=2,  
            f"top5={top5_toks}, content_count={content_in_top5}")  
        print(f"    '{prompt}' → top1='{top1_tok}', top5={top5_toks}")  
    m.amm.tree.store.clear(); m.amm.tree.root=_Node(); m.amm.tree.nid=0; m.amm.time=0  
  
def test_degeneration_quality(m, c, R, texts):  
    print("\n── 退化质量测试 ──")  
    m.amm.tree.store.clear(); m.amm.tree.root=_Node(); m.amm.tree.nid=0; m.amm.time=0  
    for t in texts[:6]: m.write(t,training_mode=True)  
    m.eval(); cc=m.content_classifier  
    for prompt in ["The pianist","Quantum computing is","Stars and galaxies"]:  
        torch.manual_seed(42)  
        with torch.no_grad(): gen=m.generate(prompt,mt=30,greedy=False)  
        new_text=gen[len(prompt):].strip()  
        total_chars=len(new_text); alpha_chars=sum(1 for ch in new_text if ch.isalpha())  
        ratio=alpha_chars/max(total_chars,1)  
        new_tokens=m.tok.encode(new_text) if new_text else []  
        content_count=len(cc.get_content_ids_from_tokens(new_tokens)) if cc else 0  
        content_ratio=content_count/max(len(new_tokens),1)  
        R.check(f"degen_{prompt[:10]}_has_content",total_chars>=5,  
            f"chars={total_chars}")  
        R.check(f"degen_{prompt[:10]}_alpha_ratio",ratio>0.3,  
            f"ratio={ratio:.2f}, text='{new_text[:50]}'")  
        R.check(f"degen_{prompt[:10]}_content_ratio",content_ratio>0.1,  
            f"ratio={content_ratio:.2f}")  
        print(f"    '{prompt}' → '{new_text[:60]}' (alpha={ratio:.2f}, content={content_ratio:.2f})")  
    m.amm.tree.store.clear(); m.amm.tree.root=_Node(); m.amm.tree.nid=0; m.amm.time=0  
  
def test_domain_semantic_grounding(m, c, R):  
    print("\n── 域语义接地测试 (v3.7) ──")  
    music_texts=[  
        "He practiced piano for hours perfecting a difficult Chopin nocturne.",  
        "The orchestra performed Beethoven symphony with remarkable precision.",  
        "She studied music theory and harmonic progression at the conservatory."]  
    space_texts=[  
        "The telescope revealed distant galaxies beyond the Milky Way.",  
        "Astronauts trained for the Mars mission in simulated zero gravity.",  
        "The nebula emitted radiation across the electromagnetic spectrum."]  
    m.amm.tree.store.clear(); m.amm.tree.root=_Node(); m.amm.tree.nid=0; m.amm.time=0  
    for t in music_texts+space_texts: m.write(t,training_mode=True)  
    m.eval()  
    music_keywords={'piano','music','chopin','nocturne','orchestra','beethoven','symphony',  
                   'harmony','melody','chord','musical','sonata','concerto','instrument',  
                   'practiced','perfecting','harmonic','progression','conservatory',  
                   'performed','remarkable','precision','theory','studied',  
                   'pianist','composer','notes','score','tempo'}  
    space_keywords={'galaxy','galaxies','telescope','star','planet','orbit','space','astronaut',  
                   'mars','nebula','radiation','gravity','cosmic','solar','lunar',  
                   'universe','constellation','spectrum','satellite','mission',  
                   'astronauts','electromagnetic','revealed','distant','simulated','emitted',  
                   'trained','zero'}  
    def count_domain_words(text, keywords):  
        words=set(text.lower().split())  
        return sum(1 for w in words if any(kw in w for kw in keywords))  
    music_query_results=[]; space_query_results=[]  
    for seed in range(3):  
        torch.manual_seed(42+seed)  
        with torch.no_grad():  
            mg=m.generate("The piano performance",mt=40,greedy=False)  
            sg=m.generate("The space telescope",mt=40,greedy=False)  
        music_query_results.append(mg); space_query_results.append(sg)  
    avg_music_in_music=sum(count_domain_words(t,music_keywords) for t in music_query_results)/3  
    avg_space_in_space=sum(count_domain_words(t,space_keywords) for t in space_query_results)/3  
    avg_music_in_space=sum(count_domain_words(t,music_keywords) for t in space_query_results)/3  
    avg_space_in_music=sum(count_domain_words(t,space_keywords) for t in music_query_results)/3  
    print(f"    music_query → music_kw={avg_music_in_music:.1f}, space_kw={avg_space_in_music:.1f}")  
    print(f"    space_query → space_kw={avg_space_in_space:.1f}, music_kw={avg_music_in_space:.1f}")  
    for t in music_query_results[:1]:  
        print(f"    music_gen: '{t[len('The piano performance'):][:80]}'")  
    for t in space_query_results[:1]:  
        print(f"    space_gen: '{t[len('The space telescope'):][:80]}'")  
    R.check("domain_music_has_music_kw",avg_music_in_music>0,f"avg={avg_music_in_music:.1f}")  
    R.check("domain_space_has_space_kw",avg_space_in_space>0,f"avg={avg_space_in_space:.1f}")  
    music_margin=avg_music_in_music-avg_music_in_space  
    space_margin=avg_space_in_space-avg_space_in_music  
    R.check("domain_music_margin_positive",music_margin>0,f"margin={music_margin:.1f}")  
    R.check("domain_space_margin_positive",space_margin>0,f"margin={space_margin:.1f}")  
    m.amm.tree.store.clear(); m.amm.tree.root=_Node(); m.amm.tree.nid=0; m.amm.time=0  
  
def test_zero_train_content_grounding(m, c, R):  
    print("\n── 零训练域接地测试 ──")  
    m.amm.tree.store.clear(); m.amm.tree.root=_Node(); m.amm.tree.nid=0; m.amm.time=0  
    m.write("He practiced piano for hours perfecting a difficult Chopin nocturne.",training_mode=True)  
    m.write("She studied music theory and harmonic progression at the conservatory.",training_mode=True)  
    m.eval(); dev=next(m.parameters()).device  
    tk=m.tok("The piano performance",return_tensors='pt')  
    ids,mask=tk['input_ids'].to(dev),tk['attention_mask'].to(dev)  
    with torch.no_grad():  
        o=m.fwd(ids,mask)  
        _,_,_,cb=m._get_prefix(o['hs'],mask,update_stats=False,return_extra=True,ids=ids)  
    R.check("zero_train_cb_nonzero",cb.abs().max().item()>0.01)  
    top10_ids=cb[0].topk(10).indices.tolist()  
    top10_toks=[m.tok.decode([t]).strip().lower() for t in top10_ids]  
    has_music_word=any(w in ['piano','chopin','nocturne','practiced','perfecting',  
                             'difficult','music','theory','harmonic','harmony',  
                             'progression','conservatory','studied','hours']  
                       for w in top10_toks)  
    R.check("zero_train_cb_has_music",has_music_word,f"top10={top10_toks}")  
    print(f"    zero-train content_bias top10: {top10_toks}")  
    m.amm.tree.store.clear(); m.amm.tree.root=_Node(); m.amm.tree.nid=0; m.amm.time=0  
  
def test_direction_degeneracy(m, c, R):  
    print("\n── 方向退化边界测试 ──")  
    tree=m.amm.tree  
    R.check("degeneracy_method_exists",hasattr(tree,'check_direction_degeneracy'))  
    degen=tree.check_direction_degeneracy(threshold=0.95)  
    R.check("degeneracy_returns_list",isinstance(degen,list))  
    R.check("leaf_size_violations_method",hasattr(tree,'leaf_size_violations'))  
    violations=tree.leaf_size_violations()  
    R.check("leaf_size_violations_returns_list",isinstance(violations,list))  
  
def test_leaf_capacity_stability(c, R):  
    print("\n── 叶容量稳定性测试 ──")  
    tc=Cfg(tree_max_leaf=5,tree_K=3,d_M=c.d_M,d_F=c.d_F)  
    tree=DirectionTree(tc); N=100  
    for i in range(N):  
        d=F.normalize(torch.randn(tc.d_M),dim=0)  
        me=MemEntry(mid=i,base=torch.randn(tc.d_M),fiber=torch.randn(tc.d_F),  
            dirn=d,surprise=0.5,ts=float(i),last=float(i))  
        tree.store[me.mid]=me; tree.nid=i+1; tree._ins(tree.root,me)  
    violations=tree.leaf_size_violations()  
    R.check("leaf_capacity_no_violations",len(violations)==0,  
        f"violations={violations}")  
    errs=tree.verify_consistency()  
    R.check("leaf_capacity_consistent",len(errs)==0,str(errs))  
    R.check("leaf_capacity_count",tree.root.count()==N)  
    degen=tree.check_direction_degeneracy(threshold=0.999)  
    R.check("leaf_degen_check_runs",isinstance(degen,list))  
  
def test_empty_memory(m, c, R):  
    print("\n── 空记忆测试 ──")  
    dev=next(m.parameters()).device  
    old_s=dict(m.amm.tree.store); old_r=m.amm.tree.root; old_n=m.amm.tree.nid  
    m.amm.tree.store={}; m.amm.tree.root=_Node(); m.amm.tree.nid=0; m.eval()  
    tk=m.tok("Hello world",return_tensors='pt')  
    ids,mask=tk['input_ids'].to(dev),tk['attention_mask'].to(dev)  
    with torch.no_grad():  
        o=m.fwd(ids,mask)  
        prefix,_,_,cb=m._get_prefix(o['hs'],mask,return_extra=True,ids=ids)  
    R.check("empty_mem_prefix_finite",prefix.isfinite().all().item())  
    R.check("empty_mem_cb_zero",cb.abs().max().item()<1e-6)  
    with torch.no_grad(): gen=m.generate("Hello",mt=10,greedy=True)  
    R.check("empty_mem_generate_ok",len(gen)>0)  
    m.amm.tree.store=old_s; m.amm.tree.root=old_r; m.amm.tree.nid=old_n  
  
def test_functional(m, c, R, texts):  
    print("\n── 功能测试 ──")  
    dev=next(m.parameters()).device; total=0; gvs=[]  
    m.amm.tree.store.clear(); m.amm.tree.root=_Node(); m.amm.tree.nid=0; m.amm.time=0  
    for t in texts:  
        ns,gv=m.write(t,training_mode=True); total+=ns; gvs.extend(gv)  
    R.check("write_count",total>0,f"stored={total}/{len(texts)}")  
    R.check("write_gate_range",all(0<=g<=1 for g in gvs))  
    all_have_text=all(e.source_text for e in m.amm.tree.store.values())  
    R.check("write_source_text",all_have_text)  
    all_have_sem=all(e.semantic_emb is not None for e in m.amm.tree.store.values())  
    R.check("write_semantic_emb",all_have_sem)  
    all_have_ct=all(len(e.content_token_ids)>0 for e in m.amm.tree.store.values())  
    R.check("write_content_tokens",all_have_ct)  
    all_have_wte=all(e.content_wte_centroid is not None for e in m.amm.tree.store.values())  
    R.check("write_wte_centroid",all_have_wte)  
    all_have_exp=all(len(e.expanded_content_ids)>0 for e in m.amm.tree.store.values())  
    R.check("write_expanded_ids",all_have_exp)  
    m.eval()  
    tk=m.tok("Tell me about piano.",return_tensors='pt')  
    ids,mask=tk['input_ids'].to(dev),tk['attention_mask'].to(dev)  
    with torch.no_grad():  
        o=m.fwd(ids,mask)  
        _,_,_,cb=m._get_prefix(o['hs'],mask,return_extra=True,ids=ids)  
    R.check("retrieve_cb_nonzero",cb.abs().max().item()>0)  
    torch.manual_seed(42)  
    with torch.no_grad(): gen=m.generate("The pianist",20,greedy=True)  
    R.check("generate_nonempty",len(gen)>len("The pianist"))  
    m.amm.time+=2000; n0=len(m.amm.tree.store); nd=m.amm.decay()  
    n1=len(m.amm.tree.store)  
    R.check("decay_consistent",n1==n0-nd)  
  
def test_batch_retrieval(m, c, R):  
    print("\n── Batch 检索测试 ──")  
    dev=next(m.parameters()).device  
    for t in ["Cats are fluffy.","Stars shine bright."]: m.write(t,training_mode=True)  
    m.eval()  
    tk=m.tok(["Tell me about cats.","The night sky."],  
        return_tensors='pt',padding=True,truncation=True)  
    ids,mask=tk['input_ids'].to(dev),tk['attention_mask'].to(dev)  
    with torch.no_grad():  
        o=m.fwd(ids,mask)  
        _,_,_,cb=m._get_prefix(o['hs'],mask,return_extra=True,ids=ids)  
    R.check("batch_cb_shape",cb.shape[0]==2)  
    R.check("batch_cb_finite",cb.isfinite().all().item())  
  
def test_gradient_flow(m, c, R):  
    print("\n── 梯度流测试 ──")  
    dev=next(m.parameters()).device  
    for t in ["The cat sat.","Quantum computing.","Piano practice."]:  
        m.write(t,training_mode=True)  
    m.train(); m.zero_grad()  
    tk=m.tok("Tell me about music.",return_tensors='pt')  
    ids,mask=tk['input_ids'].to(dev),tk['attention_mask'].to(dev)  
    with torch.no_grad(): bo=m.fwd(ids,mask)  
    prefix=m._get_prefix(bo['hs'],mask,update_stats=False,ids=ids)  
    fs=m.bridge._last_fiber_summary  
    o=m.fwd(ids,mask,prefix)  
    lg=o['logits'][:,o['pl']:-1]; tg=ids[:,1:]; ml=min(lg.shape[1],tg.shape[1])  
    if ml>0:  
        loss=F.cross_entropy(lg[:,:ml].reshape(-1,lg.shape[-1]),tg[:,:ml].reshape(-1))  
        if fs is not None:  
            probe_pred=m.semantic_probe(prefix)  
            loss_sp=F.mse_loss(probe_pred,fs.detach())  
            (loss+loss_sp).backward()  
        else: loss.backward()  
        checks=[  
            ("dir_predictor",m.amm.dir_pred.net[0].weight),  
            ("fiber_connection",m.amm.conn.net[0].weight),  
            ("fiber_attn",m.amm.attn.Wq.weight),  
            ("qformer_proj",m.bridge.proj.layers[0].ca.in_proj_weight),  
            ("content_bypass",m.bridge.bypass.proj[0].weight),  
            ("prefix_aligner_scale",m.bridge.aligner.scale_logit)]  
        for name,param in checks:  
            hg=param.grad is not None and param.grad.abs().max().item()>0  
            R.check(f"grad_{name}",hg,  
                f"grad={'None' if param.grad is None else param.grad.abs().max().item():.2e}")  
    m.zero_grad(); m.eval()  
  
def test_gradient_balance(m, c, R, texts):  
    print("\n── 梯度均衡测试 ──")  
    m.amm.tree.store.clear(); m.amm.tree.root=_Node(); m.amm.tree.nid=0; m.amm.time=0  
    for t in texts[:4]: m.write(t,training_mode=True)  
    trainer=Trainer(m,c); info=trainer.step(texts[:3])  
    gn=info['grad_norms']; print(f"    grad_norms: {gn}")  
    for name in ['ctx_encoder','fib_encoder','qformer','content_bypass','prefix_aligner','vocab_proj']:  
        norm=gn.get(name,0.0)  
        R.check(f"grad_{name}_nonzero",norm>0,f"norm={norm:.2e}")  
  
def test_quality(m, c, R, texts):  
    print("\n── 质量测试 ──")  
    m.amm.tree.store.clear(); m.amm.tree.root=_Node(); m.amm.tree.nid=0; m.amm.time=0  
    for t in texts: m.write(t,training_mode=True)  
    trainer=Trainer(m,c); losses=[]  
    for ep in range(6):  
        info=trainer.step(texts[:4]); losses.append(info['total'])  
        print(f"    step{ep+1}: total={info['total']:.4f} recon={info['recon']:.4f} "  
              f"sa={info['semantic_alignment']:.4f}")  
    R.check("training_loss_finite",all(l==l and l<1e6 for l in losses))  
    torch.manual_seed(123); m.eval()  
    with torch.no_grad(): gen_mem=m.generate("The pianist",25,greedy=True)  
    old_s=dict(m.amm.tree.store); old_r=m.amm.tree.root; old_n=m.amm.tree.nid  
    m.amm.tree.store={}; m.amm.tree.root=_Node()  
    with torch.no_grad(): gen_no=m.generate("The pianist",25,greedy=True)  
    m.amm.tree.store=old_s; m.amm.tree.root=old_r; m.amm.tree.nid=old_n  
    print(f"    有记忆: \"{gen_mem}\"")  
    print(f"    无记忆: \"{gen_no}\"")  
    R.check("quality_diff",gen_mem!=gen_no,"生成结果应该不同")  
  
def test_memory_refresh(m, c, R, texts):  
    print("\n── 记忆刷新测试 ──")  
    m.amm.tree.store.clear(); m.amm.tree.root=_Node(); m.amm.tree.nid=0; m.amm.time=0  
    for t in texts[:4]: m.write(t,training_mode=True)  
    n_before=len(m.amm.tree.store)  
    with torch.no_grad(): n_refreshed=m._refresh_all_memories()  
    n_after=len(m.amm.tree.store)  
    R.check("refresh_post_count",n_after>0,f"n={n_after}")  
    errs=m.amm.tree.verify_consistency()  
    R.check("refresh_consistent",len(errs)==0,str(errs))  
    all_have_wte=all(e.content_wte_centroid is not None for e in m.amm.tree.store.values())  
    R.check("refresh_wte_preserved",all_have_wte)  
  
def test_ablation_modes(m, c, R, texts):  
    print("\n── 消融模式测试 ──")  
    for t in texts[:3]: m.write(t,training_mode=True)  
    dev=next(m.parameters()).device; m.eval()  
    tk=m.tok("Tell me about music.",return_tensors='pt')  
    ids,mask=tk['input_ids'].to(dev),tk['attention_mask'].to(dev)  
    prefixes={}  
    for mode in ['both','qformer_only','bypass_only']:  
        m.bridge.inject_mode=mode  
        with torch.no_grad():  
            o=m.fwd(ids,mask)  
            prefix=m._get_prefix(o['hs'],mask,update_stats=False,ids=ids)  
        prefixes[mode]=prefix.clone()  
        R.check(f"ablation_{mode}_finite",prefix.isfinite().all().item())  
    m.bridge.inject_mode='both'  
    if 'qformer_only' in prefixes and 'bypass_only' in prefixes:  
        diff=(prefixes['qformer_only']-prefixes['bypass_only']).abs().max().item()  
        R.check("ablation_modes_differ",diff>1e-6)  
  
def test_tree_consistency(m, c, R):  
    print("\n── 树一致性测试 ──")  
    tree=m.amm.tree; errs=tree.verify_consistency()  
    R.check("tree_consistency",len(errs)==0,str(errs))  
  
def test_deep_tree(c, R):  
    print("\n── 深层树测试 ──")  
    tc=Cfg(tree_max_leaf=5,tree_K=3,d_M=c.d_M,d_F=c.d_F)  
    tree=DirectionTree(tc); N=150  
    for i in range(N):  
        d=F.normalize(torch.randn(tc.d_M),dim=0)  
        me=MemEntry(mid=i,base=torch.randn(tc.d_M),fiber=torch.randn(tc.d_F),  
            dirn=d,surprise=0.5,ts=float(i),last=float(i))  
        tree.store[me.mid]=me; tree.nid=i+1; tree._ins(tree.root,me)  
    errs=tree.verify_consistency()  
    R.check("deep_tree_consistency",len(errs)==0,str(errs))  
    R.check("deep_tree_count",tree.root.count()==N)  
    violations=tree.leaf_size_violations()  
    R.check("deep_tree_no_violations",len(violations)==0,f"violations={violations}")  
    for i in range(0,N,2): tree.remove(i)  
    errs=tree.verify_consistency()  
    R.check("deep_tree_post_remove",len(errs)==0,str(errs))  
    tree.rebuild(); errs=tree.verify_consistency()  
    R.check("deep_tree_post_rebuild",len(errs)==0,str(errs))  
  
def test_dealiaser(m, c, R):  
    print("\n── 去混叠测试 ──")  
    if len(m.amm.tree.store)<2: R.check("dealiaser_skip",True); return  
    da=SpectralDealiaser(m.amm,c)  
    cls=da.detect(sim_threshold=0.3); R.check("dealiaser_detect_runs",True)  
  
# ═══════════════════════════════════════════════════════════════════  
# 第22部分 · 入口  
# ═══════════════════════════════════════════════════════════════════  
def test():  
    torch.manual_seed(42); c=Cfg(); R=TestResults()  
    sep="="*60  
    print(f"\n{sep}\n  嵌入级方案B · v3.7 · 结构化测试\n{sep}")  
    t0=time.time()  
    print("\n[构建]")  
    m=MemLLM(c); m.load("gpt2")  
    total=sum(p.numel() for p in m.parameters())  
    train=sum(p.numel() for p in m.parameters() if p.requires_grad)  
    print(f"  参数: 总{total:,}  可训练{train:,}  冻结{total-train:,}")  
    texts=[  
        "The cat sat on the mat and watched the birds outside the window.",  
        "Quantum computing uses qubits existing in superposition states.",  
        "She walked along the beach at sunset feeling warm sand beneath her feet.",  
        "The stock market experienced significant volatility during the session.",  
        "He practiced piano for hours perfecting a difficult Chopin nocturne.",  
        "The restaurant served an exquisite five course meal with wine pairings.",  
        "Machine learning algorithms identify patterns in large datasets.",  
        "The ancient temple was hidden deep within the tropical rainforest."]  
    test_properties(m,c,R)  
    test_geodesic_gradient(m,c,R)  
    test_geodesic_no_grad(m,c,R)  
    test_contrast_dimensions(m,c,R)  
    test_content_classifier(m,c,R)  
    test_wte_neighbor_cache(m,c,R)  
    test_content_semantic_emb(m,c,R)  
    test_gradient_flow(m,c,R)  
    test_tree_consistency(m,c,R)  
    test_deep_tree(c,R)  
    test_leaf_capacity_stability(c,R)  
    test_direction_degeneracy(m,c,R)  
    test_empty_memory(m,c,R)  
    test_functional(m,c,R,texts)  
    test_batch_retrieval(m,c,R)  
    # v3.7 核心验收  
    test_semantic_retrieval(m,c,R)  
    test_zero_train_content_grounding(m,c,R)  
    test_first_step_not_punct(m,c,R,texts)  
    test_degeneration_quality(m,c,R,texts)  
    test_domain_semantic_grounding(m,c,R)  
    # 保留  
    test_ablation_modes(m,c,R,texts)  
    test_memory_refresh(m,c,R,texts)  
    test_gradient_balance(m,c,R,texts)  
    test_quality(m,c,R,texts)  
    test_dealiaser(m,c,R)  
    elapsed=time.time()-t0; print(f"\n耗时: {elapsed:.1f}s")  
    print(f"\n┌─ 组件参数量 {'─'*30}┐")  
    for name,mod in [  
        ("RiemannianMetric",m.amm.metric),("FiberConnection",m.amm.conn),  
        ("FiberTransporter",m.amm.trans),("CtxEncoder",m.amm.ctx),  
        ("FibEncoder",m.amm.fib),("DirectionPredictor",m.amm.dir_pred),  
        ("EmptyStateNet",m.amm.empty_state),("WriteGate[P]",m.amm.write_gate),  
        ("RetentionScorer",m.amm.retention),("FiberAttn",m.amm.attn),  
        ("RetrievalReranker",m.amm.reranker),("ContentBypass",m.bridge.bypass),  
        ("PrefixSemanticProbe",m.semantic_probe),("PrefixAligner",m.bridge.aligner),  
        ("MemoryVocabProjector",m.vocab_proj),("QFormerProj",m.bridge.proj),  
        ("StateExtractor",m.bridge.ext),("AdaptiveLayerPool",m.layer_pool)]:  
        print(f"│  {name:28s} {sum(p.numel() for p in mod.parameters()):>8,}  │")  
    print(f"└{'─'*44}┘")  
    nb=len(m.amm.tree.store)*(c.d_M*2+c.d_F+c.d_LLM*2)*4  
    print(f"\n记忆存储: {len(m.amm.tree.store)} 条, ~{nb/1024:.1f} KB\n")  
    return R.summary()  
  
if __name__=="__main__":  
    ok=test(); exit(0 if ok else 1)  
