#!/usr/bin/env python3
"""
嵌入级方案B · v3.21
════════════════════════════════════════════════════════════════════════

v3.21 变更摘要 (相对 v3.20)
─────────────────────────

[P0-RETRIEVE] Token-Level MaxSim Retrieval
  替代 WTE centroid 均值比较
  score_maxsim(query, memory) =
    mean over q_tok in query_content_tokens of:
      max over m_tok in memory_content_tokens of:
        cosine(WTE[q_tok], WTE[m_tok])
  "piano" query vs music memory: maxsim ≈ 1.0 (piano↔piano exact match)
  "piano" query vs space memory: maxsim ≈ 0.2 (no token close to piano)
  评分权重: 0.05*dir + 0.10*semantic + 0.85*maxsim
  当 query 无内容词时自适应回退: 0.2*dir + 0.8*semantic

[P0-DECODE] Query-Weighted Per-Token Content Bias
  记忆中每个 token 按与 query token 的 max cosine 加权:
    relevance(m_tok) = max over q_tok of cosine(WTE[m_tok], WTE[q_tok])
    bias[m_tok] += retrieval_weight * relevance(m_tok)
  "piano"(rel=1.0) 得满权, "hours"(rel=0.15) 得低权
  content_bias_scale 降至 10.0 (检索更精确, 无需暴力 boost)

[P0-DECODE] Generated Token Decay + Structural Rhythm
  每生成一个 token, 其 content_bias *= 0.15^count
  "piano" 生成一次后 bias 降为 15%, 两次后降为 2.25%
  连续 2+ 个 content token 后, 临时降低 content_bias_scale * 0.25
  并对 function words 施加 +3.0 boost, 恢复句法结构
  消除 "piano pianist piano guitar piano" 堆词

[P0-PREFIX] Content WTE Direct Injection
  检索到的域词 WTE 向量按 query 相关度加权平均
  直接加到 prefix embedding (post-aligner)
  scale=0.3, 约为 prefix 幅度的 30%
  绕过 QFormerProj/ContentBypass 的未收敛学习路径
  GPT-2 注意力直接看到域词嵌入 → 首步 logit 向域词偏移

[P1-RETRIEVE] Reranker Correction Clip
  clip correction to [-0.2, +0.2]
  防止未收敛的 reranker 翻转 MaxSim 排序

[REMOVED] content_wte_centroid (被 MaxSim 完全替代)
[REMOVED] ret_wte_weight (被 ret_maxsim_weight 替代)

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
    degen_min_tokens: int = 5; degen_repeat_penalty: float = 1.4
    degen_max_consec_punct: int = 2
    probe_contrastive_tau: float = 0.1
    contrast_tau: float = 0.5
    # ── decode/prefix ──
    prefix_init_scale: float = 0.5
    degen_early_punct_penalty: float = 80.0
    degen_early_newline_penalty: float = 80.0
    early_content_steps: int = 5
    universal_content_boost: float = 2.0
    universal_content_boost_steps: int = 5
    content_bias_scale: float = 15.0
    content_bias_decay: float = 0.02
    content_bias_floor: float = 0.4
    generated_token_decay: float = 0.15
    structural_rhythm_threshold: int = 2
    structural_boost: float = 3.0
    content_repeat_penalty: float = 5.0
    first_step_content_multiplier: float = 6.0
    first_step_penalty_multiplier: float = 3.0
    step0_filler_penalty: float = 5.0
    domain_anchor_k: int = 8
    domain_anchor_boost: float = 10.0
    domain_anchor_start_step: int = 0
    domain_anchor_coverage_threshold: float = 0.15
    # ── v3.16 retrieval ──
    ret_sem_weight: float = 0.40
    ret_bidi_min_weight: float = 0.25
    ret_forward_maxsim_weight: float = 0.20
    ret_dir_weight: float = 0.15
    ret_sem_gate_ratio: float = 0.60
    reranker_clip: float = 0.2
    forward_maxsim_hard_threshold: float = 0.20
    bidi_hard_threshold: float = 0.20
    bidi_relative_ratio: float = 0.60
    fwd_coherence_ratio: float = 0.55
    score_keep_ratio: float = 0.80
    retrieval_weight_temperature: float = 0.05
    consol_maxsim_min: float = 0.40
    # ── v3.18 AND-style dual gate ──
    gate_sem_ratio: float = 0.65
    gate_bidi_ratio: float = 0.70
    gate_sem_floor: float = 0.10
    gate_bidi_floor: float = 0.10
    gate_bidi_hard_min: float = 0.12
    # diagnostic-only backward compat
    gate_sem_weight: float = 0.50
    gate_bidi_weight: float = 0.50
    gate_ratio: float = 0.70
    gate_floor: float = 0.05
    bidi_absolute_gap: float = 0.15
    # ── v3.19 content bias ──
    content_bias_relevance_floor: float = 0.05
    content_bias_concentration: float = 2.0
    # ── v3.17 retrieval expanded ids ──
    retrieval_use_expanded_ids: bool = True
    # ── prefix injection ──
    content_inject_scale: float = 1.0
    prefix_inject_last_ratio: float = 0.25
    prefix_inject_last_multiplier: float = 6.0
    prefix_inject_other_multiplier: float = 1.0
    prefix_target_multiplier: float = 3.0
    content_wte_topk_for_inject: int = 5
    use_word_starter_filter: bool = True
    bpe_echo_window: int = 3
    bpe_echo_penalty: float = 4.0
    post_starter_nonstarter_penalty: float = 3.0
    use_dominance_filter: bool = True
    dominance_margin: float = 1.25
    dominance_sem_floor: float = 0.18
    dominance_jaccard_threshold: float = 0.20
    dominance_min_label_size: int = 3
    use_first_step_lexical: bool = True
    first_step_lexical_scale: float = 45.0
    first_step_lexical_topk: int = 12
    first_step_lexical_decay_steps: int = 1
    use_tfidf_weighting: bool = True
    tfidf_smoothing: float = 1.0
    use_idf_retrieval: bool = True
    idf_floor: float = 0.1
    use_idf_dominance: bool = True
    dominance_idf_margin: float = 1.5
    dominance_idf_top1_floor: float = 0.25
    prefix_anchor_replace: bool = True
    prefix_anchor_scale: float = 3.0
    prefix_anchor_use_pe: bool = True
    # ── preserved ──
    semantic_boost_scale: float = 0.5
    semantic_boost_decay: float = 0.06
    semantic_boost_floor: float = 0.2
    semantic_align_temp: float = 0.3
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
# 第5部分 · 检索重排序 (v3.8: correction clip)
# ═══════════════════════════════════════════════════════════════════
class RetrievalReranker(nn.Module):
    def __init__(self, d_M, d_F, clip=0.2):
        super().__init__()
        self.clip=clip
        inp=2*d_M+2*d_F+1
        self.net=nn.Sequential(nn.Linear(inp,128),nn.SiLU(),nn.LayerNorm(128),
                               nn.Linear(128,64),nn.SiLU(),nn.LayerNorm(64),nn.Linear(64,1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self, xq, fq, xc, fc, dir_sim):
        B,C=xc.shape[:2]
        xq_e=xq.unsqueeze(1).expand(-1,C,-1); fq_e=fq.unsqueeze(1).expand(-1,C,-1)
        inp=torch.cat([xq_e,fq_e,xc,fc,dir_sim.unsqueeze(-1)],-1)
        correction=self.net(inp).squeeze(-1)
        correction=correction.clamp(-self.clip,self.clip)
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
    def __init__(self, d_LLM, init_scale=0.5):
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
# 第9部分 · ContentTokenClassifier (v3.19: +word_starter_ids)
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
    FILLER_WORDS = frozenset({
        'include','includes','including','included',
        'also','just','however','moreover','furthermore',
        'nevertheless','therefore','thus','hence','accordingly',
        'meanwhile','instead','rather','otherwise','additionally',
        'basically','essentially','actually','obviously','clearly',
        'simply','certainly','indeed','probably','perhaps',
        'apparently','presumably','supposedly','regardless',
        'nonetheless','conversely','alternatively','specifically',
        'generally','typically','usually','often','sometimes',
        'particularly','especially','notably'
    })
    def __init__(self, tokenizer, min_len=3):
        self.content_ids: Set[int] = set()
        self.function_ids: Set[int] = set()
        self.punct_ids: Set[int] = set()
        self.newline_ids: Set[int] = set()
        self.filler_ids: Set[int] = set()
        self.word_starter_ids: Set[int] = set()
        self.content_starter_ids: Set[int] = set()
        vocab_size = getattr(tokenizer, 'vocab_size', 50257)
        for i in range(min(vocab_size, 50300)):
            try:
                tok_text = tokenizer.decode([i])
                is_word_starter = len(tok_text) > 0 and tok_text[0] in (' ', '\t')
                stripped = tok_text.strip().lower()
                cleaned = ''.join(c for c in stripped if c.isalpha())
                if is_word_starter:
                    self.word_starter_ids.add(i)
                if '\n' in tok_text:
                    self.newline_ids.add(i); self.function_ids.add(i)
                elif stripped == '' or all(not c.isalnum() for c in stripped):
                    self.punct_ids.add(i); self.function_ids.add(i)
                elif len(cleaned) >= min_len and cleaned not in self.STOPWORDS:
                    self.content_ids.add(i)
                    if is_word_starter:
                        self.content_starter_ids.add(i)
                else:
                    self.function_ids.add(i)
                if cleaned in self.FILLER_WORDS:
                    self.filler_ids.add(i)
            except:
                self.function_ids.add(i)
        self._content_tensor = None
        self._content_starter_tensor = None
        self.starter_ids: Set[int] = set()
        starters_words = {'the','a','an','it','this','that','there','here','its','my',
                          'our','his','her','their','we','they','he','she','one'}
        for i in range(min(vocab_size, 50300)):
            try:
                tok_text = tokenizer.decode([i]).strip().lower()
                cleaned = ''.join(c for c in tok_text if c.isalpha())
                if cleaned in starters_words:
                    self.starter_ids.add(i)
            except:
                pass

    def content_mask(self, device):
        if self._content_tensor is None or self._content_tensor.device != device:
            V = max(max(self.content_ids, default=0), max(self.function_ids, default=0),
                    max(self.punct_ids, default=0), max(self.newline_ids, default=0)) + 1
            m = torch.zeros(V, device=device)
            for i in self.content_ids:
                if i < V: m[i] = 1.0
            self._content_tensor = m
        return self._content_tensor

    def content_starter_mask(self, device):
        if self._content_starter_tensor is None or self._content_starter_tensor.device != device:
            V = max(max(self.content_ids, default=0), max(self.function_ids, default=0),
                    max(self.punct_ids, default=0), max(self.newline_ids, default=0)) + 1
            m = torch.zeros(V, device=device)
            for i in self.content_starter_ids:
                if i < V: m[i] = 1.0
            self._content_starter_tensor = m
        return self._content_starter_tensor

    def get_content_ids_from_tokens(self, token_ids):
        return [t for t in token_ids if t in self.content_ids]

    def get_content_positions(self, token_ids, mask=None):
        positions = []
        for pos, tid in enumerate(token_ids):
            if mask is not None and pos < len(mask) and not mask[pos]:
                continue
            if tid in self.content_ids:
                positions.append(pos)
        return positions

# ═══════════════════════════════════════════════════════════════════
# 第10部分 · MemoryVocabProjector
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
# 第11部分 · MemEntry + DirectionTree (v3.16: 移除 content_words)
# ═══════════════════════════════════════════════════════════════════
@dataclass
class MemEntry:
    mid: int; base: torch.Tensor; fiber: torch.Tensor; dirn: torch.Tensor
    surprise: float; ts: float; last: float; cnt: int = 0; version: int = 0
    source_text: str = ""
    content_token_ids: List[int] = field(default_factory=list)
    semantic_emb: Optional[torch.Tensor] = None
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
# 第13部分 · QFormer + 嵌入桥 (v3.19: +content_target_wte)
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
        self.c=c
        self.proj=QFormerProj(c); self.ext=StateExtractor(c)
        self.pe=nn.Parameter(torch.randn(c.L_mem,c.d_LLM)*0.02)
        self.bypass=ContentBypass(c.d_F,c.d_LLM,gate_bias=c.bypass_init_gate_bias)
        self.aligner=PrefixAligner(c.d_LLM,c.prefix_init_scale)
        self.content_inject_scale=c.content_inject_scale
        self.prefix_inject_last_ratio=c.prefix_inject_last_ratio
        self.prefix_inject_last_multiplier=c.prefix_inject_last_multiplier
        self.prefix_inject_other_multiplier=c.prefix_inject_other_multiplier
        self.prefix_target_multiplier=c.prefix_target_multiplier
        self.inject_mode='both'
        self._last_inject_diag={}
        self._last_fiber_summary=None
    def inject(self, fibers, mem_mask=None, fiber_summary=None,
               content_wte_mean=None, content_target_wte=None):
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
        anchor_replace=(self.c.prefix_anchor_replace
                       and content_target_wte is not None
                       and content_target_wte.abs().max().item()>1e-6)
        cwm_applied=False
        if content_wte_mean is not None:
            cwm=content_wte_mean
            if cwm.dim()==2:
                cwm=cwm.unsqueeze(1)
            L=qf_out.shape[1]
            n_last=max(1,int(L*self.prefix_inject_last_ratio))
            pos_scale=torch.ones(L,device=qf_out.device)
            pos_scale[:L-n_last]=self.prefix_inject_other_multiplier
            pos_scale[L-n_last:]=self.prefix_inject_last_multiplier
            if anchor_replace:
                pos_scale[-1]=0.0
            pos_scale=pos_scale.view(1,-1,1)
            qf_out=qf_out+cwm*self.content_inject_scale*pos_scale
            cwm_applied=True
        target_applied=False
        anchor_norm_val=0.0
        if anchor_replace:
            ctw=content_target_wte
            anchor_slot=ctw*self.c.prefix_anchor_scale
            if self.c.prefix_anchor_use_pe:
                anchor_slot=anchor_slot+self.pe[-1].unsqueeze(0)
            qf_out=torch.cat([qf_out[:,:-1,:],anchor_slot.unsqueeze(1)],dim=1)
            target_applied=True
            anchor_norm_val=anchor_slot.norm(dim=-1).mean().item()
        elif content_target_wte is not None:
            ctw=content_target_wte
            if ctw.dim()==2:
                ctw=ctw.unsqueeze(1)
            target_scale=torch.zeros(qf_out.shape[1],device=qf_out.device)
            target_scale[-1]=self.prefix_target_multiplier
            qf_out=qf_out+ctw*target_scale.view(1,-1,1)
            target_applied=True
        self._last_fiber_summary=fiber_summary.detach() if fiber_summary is not None else None
        self._last_inject_diag={
            'bypass_gate':gate_val.mean().item() if gate_val is not None else None,
            'qf_norm':qf_out.norm().item(),
            'bypass_norm':bp_out.norm().item() if bp_out is not None else 0.0,
            'aligner_scale':torch.sigmoid(self.aligner.scale_logit).item()*self.aligner._target_std.item(),
            'cwm_applied':cwm_applied,
            'target_applied':target_applied,
            'anchor_replace':anchor_replace,
            'anchor_norm':anchor_norm_val}
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
# 第15部分 · DegenerationGuard (v3.8: 更强的重复检测)
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
    def process(self, logits, generated_ids, step, first_step_penalty_mult=1.0):
        self._build()
        punct_pen = self.cfg.degen_early_punct_penalty
        newline_pen = self.cfg.degen_early_newline_penalty
        if step == 0:
            punct_pen *= first_step_penalty_mult
            newline_pen *= first_step_penalty_mult
        if step<self.cfg.early_content_steps:
            for pid in self._punct_ids:
                if pid<logits.shape[-1]: logits[0,pid]-=punct_pen
            for nid in self._newline_ids:
                if nid<logits.shape[-1]: logits[0,nid]-=newline_pen
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
    top_sem_sim: float = 0.0
    top_forward_maxsim: float = 0.0
    top_backward_maxsim: float = 0.0
    top_bidi_min: float = 0.0
    top_gate_affinity: float = 0.0
    gate_threshold: float = 0.0
    n_gate_pass: int = 0
    n_candidates_initial: int = 0
    n_after_hard_filter: int = 0
    n_after_score_filter: int = 0
    n_after_coherence_filter: int = 0
    n_after_bidi_gap_filter: int = 0
    dominance_triggered: bool = False
    dominant_memory_id: Optional[int] = None
    dominance_margin_observed: float = 0.0
    n_after_dominance_filter: int = 0
    batch_mem_weights: List[List[Tuple[int, float]]] = field(default_factory=list)
    per_memory_forward_maxsim: Dict[int, float] = field(default_factory=dict)
    per_memory_bidi_min: Dict[int, float] = field(default_factory=dict)
    per_memory_sem_sim: Dict[int, float] = field(default_factory=dict)
    per_memory_gate_affinity: Dict[int, float] = field(default_factory=dict)
    dominant_per_batch: List[Optional[int]] = field(default_factory=list)
    idf_applied: bool = False
    top_forward_maxsim_idf: float = 0.0
    top_bidi_min_idf: float = 0.0
    per_memory_forward_maxsim_idf: Dict[int, float] = field(default_factory=dict)
    dominance_idf_margin_observed: float = 0.0

# ═══════════════════════════════════════════════════════════════════
# 第17部分 · AMM (v3.19: +bidi absolute gap filter)
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
        self.reranker=RetrievalReranker(c.d_M,c.d_F,clip=c.reranker_clip)
        self.tree=DirectionTree(c); self.time=0.
        self.wte_normed: Optional[torch.Tensor] = None

    def surprise_proxy(self, logits, tgt):
        nll=-F.log_softmax(logits,-1).gather(2,tgt.unsqueeze(-1)).squeeze(-1)
        T=nll.shape[1]
        if T==0: return logits.new_zeros(logits.shape[0])
        w=torch.linspace(0.5,1.5,T,**_dev(nll)); w=w/w.sum()*T
        return (nll*w.unsqueeze(0)).mean(-1)

    def _compute_dirn(self, base, fiber):
        with torch.no_grad():
            return self.dir_pred(base.unsqueeze(0),fiber.unsqueeze(0)).squeeze(0)

    def _get_mem_scoring_ids(self, mem):
        if self.c.retrieval_use_expanded_ids and mem.expanded_content_ids:
            return mem.expanded_content_ids
        return mem.content_token_ids

    @staticmethod
    def _mem_label_set(mem, content_classifier) -> FrozenSet[int]:
        if content_classifier is None:
            return frozenset(mem.content_token_ids)
        return frozenset(t for t in mem.content_token_ids
                         if t in content_classifier.content_starter_ids)

    @staticmethod
    def _jaccard(s1: FrozenSet[int], s2: FrozenSet[int]) -> float:
        if not s1 or not s2:
            return 0.0
        inter = len(s1 & s2)
        union = len(s1 | s2)
        return inter / union if union > 0 else 0.0

    def _compute_corpus_idf(self, content_classifier) -> Dict[int, float]:
        s=self.c.tfidf_smoothing
        N=len(self.tree.store)
        if N==0:
            return {}
        df={}
        for mem in self.tree.store.values():
            if content_classifier is not None:
                label_set=set(t for t in mem.content_token_ids
                              if t in content_classifier.content_starter_ids)
            else:
                label_set=set(mem.content_token_ids)
            for t in label_set:
                df[t]=df.get(t,0)+1
        return {t: math.log((N+s)/(d+s))+1.0 for t,d in df.items()}

    @staticmethod
    def _compute_forward_maxsim(query_ids, mem_ids, wte_normed,
                                query_idf=None, idf_floor=0.1):
        if not query_ids or not mem_ids:
            return 0.0
        V = wte_normed.shape[0]
        q_valid = [i for i in query_ids if i < V]
        m_valid = [i for i in mem_ids if i < V]
        if not q_valid or not m_valid:
            return 0.0
        q_vecs = wte_normed[q_valid]
        m_vecs = wte_normed[m_valid]
        sim = q_vecs @ m_vecs.T
        max_per_q = sim.max(dim=1).values
        if query_idf is not None:
            weights=torch.tensor(
                [max(query_idf.get(q, idf_floor), idf_floor) for q in q_valid],
                device=wte_normed.device, dtype=sim.dtype)
            total=weights.sum().clamp(min=1e-8)
            return ((max_per_q*weights).sum()/total).item()
        return max_per_q.mean().item()

    @staticmethod
    def _compute_backward_maxsim(query_ids, mem_ids, wte_normed,
                                 query_idf=None, idf_floor=0.1):
        if not query_ids or not mem_ids:
            return 0.0
        V = wte_normed.shape[0]
        q_valid = [i for i in query_ids if i < V]
        m_valid = [i for i in mem_ids if i < V]
        if not q_valid or not m_valid:
            return 0.0
        q_vecs = wte_normed[q_valid]
        m_vecs = wte_normed[m_valid]
        sim = q_vecs @ m_vecs.T
        max_per_m_vals,max_per_m_idx=sim.max(dim=0)
        if query_idf is not None:
            q_weights=torch.tensor(
                [max(query_idf.get(q, idf_floor), idf_floor) for q in q_valid],
                device=wte_normed.device, dtype=sim.dtype)
            matched=q_weights[max_per_m_idx]
            total=matched.sum().clamp(min=1e-8)
            return ((max_per_m_vals*matched).sum()/total).item()
        return max_per_m_vals.mean().item()

    @staticmethod
    def _compute_maxsim_bidi(ids_a, ids_b, wte_normed,
                             query_idf=None, idf_floor=0.1):
        fwd = AMM._compute_forward_maxsim(ids_a, ids_b, wte_normed, query_idf, idf_floor)
        bwd = AMM._compute_backward_maxsim(ids_a, ids_b, wte_normed, query_idf, idf_floor)
        return 0.5 * fwd + 0.5 * bwd

    def _check_consolidation_compatible(self, existing_content_ids, new_content_ids):
        if not existing_content_ids or not new_content_ids:
            return True
        if self.wte_normed is None:
            return True
        maxsim = self._compute_maxsim_bidi(
            existing_content_ids, new_content_ids, self.wte_normed)
        return maxsim >= self.c.consol_maxsim_min

    def store_mem(self, h, surp, training_mode=False, source_text="",
                  content_token_ids=None,
                  content_semantic_emb=None, expanded_content_ids=None):
        dev=h.device; h2=h.unsqueeze(0)
        x=self.ctx(h2).squeeze(0).detach()
        s=surp if isinstance(surp,torch.Tensor) else torch.tensor(surp,**_dev(h))
        sv=s.view(1) if s.dim()<=1 else s
        f=self.fib(h2,x.unsqueeze(0),sv).squeeze(0).detach()
        d=self._compute_dirn(x,f)
        sem_emb=content_semantic_emb if content_semantic_emb is not None else h.detach().clone()
        ct_ids=content_token_ids or []
        exp_ids=expanded_content_ids or []
        if self.tree.store:
            scored=self.tree.retrieve(d.detach(),bw=1)[:5]
            for mid,_ in scored:
                if mid in self.tree.store:
                    ex=self.tree.store[mid]
                    dist=self.metric.midpoint_approx_distance(
                        x.unsqueeze(0),ex.base.unsqueeze(0).to(dev)).item()
                    if dist<self.c.consol_dist*self.c.consol_conflict_ratio:
                        if not self._check_consolidation_compatible(
                                ex.content_token_ids, ct_ids):
                            continue
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
                        self.time+=1; return ex
        m=MemEntry(mid=self.tree.nid,base=x.detach().clone(),fiber=f.detach().clone(),
                   dirn=d.detach().clone(),surprise=s.item(),ts=self.time,last=self.time,
                   source_text=source_text,content_token_ids=ct_ids,
                   semantic_emb=sem_emb.detach().clone() if sem_emb is not None else None,
                   expanded_content_ids=exp_ids)
        self.tree.nid+=1; self.tree.insert(m); self.time+=1; return m

    def retrieve_multi(self, xq, fq, topk=None, bw=None, update_stats=True,
                       query_semantic_emb=None,
                       query_content_ids_per_batch=None,
                       wte_normed=None,
                       content_classifier=None):
        B=xq.shape[0]; dev=xq.device
        topk=topk or self.c.retrieval_topk; bw=bw or self.c.retrieval_beam
        recall_k=int(topk*self.c.retrieval_recall_factor)
        flat_thresh=self.c.flat_scan_threshold_factor*topk
        qdir=self.dir_pred(xq,fq)
        diag=RetrievalDiag()
        corpus_idf=self._compute_corpus_idf(content_classifier) if self.c.use_idf_retrieval else None
        diag.idf_applied=corpus_idf is not None
        idf_floor=self.c.idf_floor
        if not self.tree.store:
            empty=self.empty_state(xq,fq)
            mask=torch.ones(B,1,**_dev(xq))
            summary=empty.mean(1) if empty.dim()==3 else empty
            diag.fiber_summary_norm=summary.norm().item()
            diag.batch_mem_weights=[[] for _ in range(B)]
            diag.dominant_per_batch=[None for _ in range(B)]
            return empty.unsqueeze(1),mask,summary,diag
        all_results=[]; all_masks=[]; all_biases=[]; all_summaries=[]; all_batch_mw=[]; all_dominant=[]
        for b in range(B):
            n_store=len(self.tree.store)
            if n_store<=flat_thresh:
                mids=list(self.tree.store.keys()); diag.was_flat_scan=True
            else:
                scored=self.tree.retrieve(qdir[b].detach(),bw)
                mids=[s[0] for s in scored[:recall_k]]
            mems=[self.tree.store[i] for i in mids if i in self.tree.store]
            diag.recall_count=len(mems)
            diag.n_candidates_initial=len(mems)
            if not mems:
                empty=self.empty_state(xq[b:b+1],fq[b:b+1])
                all_results.append(empty.squeeze(0).unsqueeze(0))
                all_masks.append(torch.ones(1,**_dev(xq)))
                all_biases.append(torch.zeros(1,**_dev(xq)))
                all_summaries.append(empty.squeeze(0))
                all_batch_mw.append([]); all_dominant.append(None)
                continue
            C=len(mems)
            sb=torch.stack([m.base.to(dev) for m in mems])
            sf=torch.stack([m.fiber.to(dev) for m in mems])
            md=torch.stack([m.dirn.to(dev) for m in mems])
            raw_dir_sim=torch.einsum('d,cd->c',qdir[b],md)
            diag.top_dir_sim=raw_dir_sim.max().item()

            sem_sims=[]
            if query_semantic_emb is not None:
                for mem in mems:
                    if mem.semantic_emb is not None:
                        s=F.cosine_similarity(
                            query_semantic_emb[b:b+1],
                            mem.semantic_emb.unsqueeze(0).to(dev),dim=-1).squeeze()
                        sem_sims.append(s)
                    else:
                        sem_sims.append(raw_dir_sim.new_tensor(0.0))
                sem_sim_t=torch.stack(sem_sims)
                diag.top_sem_sim=sem_sim_t.max().item()
            else:
                sem_sim_t=torch.zeros(C,device=dev)

            q_content_ids=(query_content_ids_per_batch[b]
                           if query_content_ids_per_batch and b<len(query_content_ids_per_batch)
                           else [])
            wn=wte_normed if wte_normed is not None else self.wte_normed
            if q_content_ids and wn is not None:
                forward_scores=[]; backward_scores=[]; forward_idf_scores=[]
                for mem in mems:
                    scoring_ids=self._get_mem_scoring_ids(mem)
                    fwd=self._compute_forward_maxsim(q_content_ids,scoring_ids,wn,corpus_idf,idf_floor)
                    bwd=self._compute_backward_maxsim(q_content_ids,scoring_ids,wn,corpus_idf,idf_floor)
                    forward_scores.append(fwd); backward_scores.append(bwd); forward_idf_scores.append(fwd)
                forward_t=torch.tensor(forward_scores,device=dev)
                backward_t=torch.tensor(backward_scores,device=dev)
                bidi_min_t=torch.minimum(forward_t,backward_t)
                forward_idf_t=torch.tensor(forward_idf_scores,device=dev)
                diag.top_forward_maxsim=forward_t.max().item()
                diag.top_backward_maxsim=backward_t.max().item()
                diag.top_bidi_min=bidi_min_t.max().item()
                diag.top_forward_maxsim_idf=forward_idf_t.max().item()
                diag.top_bidi_min_idf=bidi_min_t.max().item()
            else:
                forward_t=torch.zeros(C,device=dev)
                backward_t=torch.zeros(C,device=dev)
                bidi_min_t=torch.zeros(C,device=dev)
                forward_idf_t=torch.zeros(C,device=dev)

            combined_sim=(self.c.ret_sem_weight*sem_sim_t
                          +self.c.ret_bidi_min_weight*bidi_min_t
                          +self.c.ret_forward_maxsim_weight*forward_t
                          +self.c.ret_dir_weight*raw_dir_sim)

            top_sem=sem_sim_t.max().item() if C>0 else 0.0
            top_bidi=bidi_min_t.max().item() if C>0 else 0.0
            sem_thresh=max(self.c.gate_sem_floor, top_sem*self.c.gate_sem_ratio)
            bidi_thresh=max(self.c.gate_bidi_floor, top_bidi*self.c.gate_bidi_ratio, self.c.gate_bidi_hard_min)
            hard_mask=(sem_sim_t>=sem_thresh) & (bidi_min_t>=bidi_thresh)
            gate_affinity=(self.c.gate_sem_weight*sem_sim_t
                           +self.c.gate_bidi_weight*bidi_min_t)
            diag.top_gate_affinity=gate_affinity.max().item() if C>0 else 0.0
            diag.gate_threshold=max(sem_thresh, bidi_thresh)
            diag.n_gate_pass=int(hard_mask.sum().item())
            if hard_mask.sum().item()==0:
                and_score=torch.minimum(sem_sim_t,bidi_min_t)
                hard_mask[and_score.argmax()]=True
            diag.n_after_hard_filter=int(hard_mask.sum().item())
            for mi,mem in enumerate(mems):
                diag.per_memory_gate_affinity[mem.mid]=gate_affinity[mi].item()

            keep_indices=hard_mask.nonzero(as_tuple=True)[0]
            if keep_indices.numel()>0 and keep_indices.numel()<C:
                mems=[mems[i] for i in keep_indices.tolist()]
                sb=sb[keep_indices]; sf=sf[keep_indices]
                combined_sim=combined_sim[keep_indices]
                raw_dir_sim=raw_dir_sim[keep_indices]
                forward_t=forward_t[keep_indices]
                bidi_min_t=bidi_min_t[keep_indices]
                sem_sim_t=sem_sim_t[keep_indices]
                forward_idf_t=forward_idf_t[keep_indices]
                C=len(mems)

            rerank_scores=self.reranker(
                xq[b:b+1],fq[b:b+1],sb.unsqueeze(0),sf.unsqueeze(0),
                combined_sim.unsqueeze(0)).squeeze(0)
            diag.reranker_delta_mean=(rerank_scores-combined_sim).abs().mean().item()
            diag.top_reranker_score=rerank_scores.max().item()

            if C>1:
                top_score=rerank_scores.max()
                score_thresh=top_score*self.c.score_keep_ratio
                score_mask=rerank_scores>=score_thresh
                if score_mask.sum().item()<1:
                    score_mask[rerank_scores.argmax()]=True
                score_keep=score_mask.nonzero(as_tuple=True)[0]
                diag.n_after_score_filter=score_keep.numel()
                if score_keep.numel()<C:
                    mems=[mems[i] for i in score_keep.tolist()]
                    sb=sb[score_keep]; sf=sf[score_keep]
                    rerank_scores=rerank_scores[score_keep]
                    forward_t=forward_t[score_keep]
                    bidi_min_t=bidi_min_t[score_keep]
                    sem_sim_t=sem_sim_t[score_keep]
                    forward_idf_t=forward_idf_t[score_keep]
                    C=len(mems)
            else:
                diag.n_after_score_filter=C

            if C>1 and forward_t.max().item()>0:
                top_fwd_here=forward_t.max()
                coherence_mask=forward_t>=top_fwd_here*self.c.fwd_coherence_ratio
                if coherence_mask.sum()>=1:
                    coherence_keep=coherence_mask.nonzero(as_tuple=True)[0]
                    diag.n_after_coherence_filter=coherence_keep.numel()
                    if coherence_keep.numel()<C:
                        mems=[mems[i] for i in coherence_keep.tolist()]
                        sb=sb[coherence_keep]; sf=sf[coherence_keep]
                        rerank_scores=rerank_scores[coherence_keep]
                        forward_t=forward_t[coherence_keep]
                        bidi_min_t=bidi_min_t[coherence_keep]
                        sem_sim_t=sem_sim_t[coherence_keep]
                        forward_idf_t=forward_idf_t[coherence_keep]
                        C=len(mems)
                else:
                    diag.n_after_coherence_filter=C
            else:
                diag.n_after_coherence_filter=C

            if C>1 and bidi_min_t.max().item()>0:
                top_bidi_here=bidi_min_t.max().item()
                gap_mask=bidi_min_t>=(top_bidi_here-self.c.bidi_absolute_gap)
                if gap_mask.sum()>=1:
                    gap_keep=gap_mask.nonzero(as_tuple=True)[0]
                    diag.n_after_bidi_gap_filter=gap_keep.numel()
                    if gap_keep.numel()<C:
                        mems=[mems[i] for i in gap_keep.tolist()]
                        sb=sb[gap_keep]; sf=sf[gap_keep]
                        rerank_scores=rerank_scores[gap_keep]
                        forward_t=forward_t[gap_keep]
                        bidi_min_t=bidi_min_t[gap_keep]
                        sem_sim_t=sem_sim_t[gap_keep]
                        forward_idf_t=forward_idf_t[gap_keep]
                        C=len(mems)
                else:
                    diag.n_after_bidi_gap_filter=C
            else:
                diag.n_after_bidi_gap_filter=C

            dominant_mid=None
            if self.c.use_idf_dominance and C>=2 and forward_idf_t.max().item()>0:
                fwd_sorted,fwd_sort_idx=torch.sort(forward_idf_t,descending=True)
                top1_idx=fwd_sort_idx[0].item()
                top1_fwd=fwd_sorted[0].item()
                top2_fwd=fwd_sorted[1].item()
                idf_margin=top1_fwd/max(top2_fwd,1e-6)
                diag.dominance_idf_margin_observed=idf_margin
                if top1_fwd>=self.c.dominance_idf_top1_floor and idf_margin>=self.c.dominance_idf_margin:
                    diag.dominance_triggered=True
                    dominant_mid=mems[top1_idx].mid
                    keep_thresh=top1_fwd/self.c.dominance_idf_margin
                    keep_mask=forward_idf_t>=keep_thresh
                    keep_mask[top1_idx]=True
                    keep_local=keep_mask.nonzero(as_tuple=True)[0]
                    if keep_local.numel()<C:
                        mems=[mems[i] for i in keep_local.tolist()]
                        sb=sb[keep_local]; sf=sf[keep_local]
                        rerank_scores=rerank_scores[keep_local]
                        forward_t=forward_t[keep_local]
                        bidi_min_t=bidi_min_t[keep_local]
                        sem_sim_t=sem_sim_t[keep_local]
                        forward_idf_t=forward_idf_t[keep_local]
                        C=len(mems)

            if self.c.use_dominance_filter and C>=2 and content_classifier is not None:
                dominance_scores=forward_idf_t if forward_idf_t.max().item()>0 else rerank_scores
                sorted_idx=torch.argsort(dominance_scores,descending=True)
                top1_local=sorted_idx[0].item()
                top2_local=sorted_idx[1].item()
                top1_score=dominance_scores[top1_local].item()
                top2_score=dominance_scores[top2_local].item()
                margin=top1_score/max(abs(top2_score),1e-6) if top2_score>0 else float('inf')
                diag.dominance_margin_observed=margin
                top1_sem=sem_sim_t[top1_local].item()
                top1_mem=mems[top1_local]
                top1_label=self._mem_label_set(top1_mem,content_classifier)
                if (len(top1_label)>=self.c.dominance_min_label_size
                        and top1_sem>=self.c.dominance_sem_floor
                        and margin>=self.c.dominance_margin):
                    diag.dominance_triggered=True
                    if dominant_mid is None:
                        dominant_mid=top1_mem.mid
                    keep_local=[]
                    for i,mem in enumerate(mems):
                        if i==top1_local:
                            keep_local.append(i); continue
                        mem_label=self._mem_label_set(mem,content_classifier)
                        if self._jaccard(top1_label,mem_label)>=self.c.dominance_jaccard_threshold:
                            keep_local.append(i)
                    if len(keep_local)<C:
                        kt=torch.tensor(keep_local,device=dev,dtype=torch.long)
                        mems=[mems[i] for i in keep_local]
                        sb=sb[kt]; sf=sf[kt]
                        rerank_scores=rerank_scores[kt]
                        forward_t=forward_t[kt]
                        bidi_min_t=bidi_min_t[kt]
                        sem_sim_t=sem_sim_t[kt]
                        forward_idf_t=forward_idf_t[kt]
                        C=len(mems)
            diag.n_after_dominance_filter=C

            if not self.training and C>topk:
                _,top_idx=rerank_scores.topk(topk)
                mems=[mems[i] for i in top_idx.cpu().tolist()]
                sb=sb[top_idx]; sf=sf[top_idx]; rerank_scores=rerank_scores[top_idx]
                forward_t=forward_t[top_idx]
                bidi_min_t=bidi_min_t[top_idx]
                sem_sim_t=sem_sim_t[top_idx]
                forward_idf_t=forward_idf_t[top_idx]
                C=topk

            for mi,mem in enumerate(mems):
                diag.per_memory_forward_maxsim[mem.mid]=forward_t[mi].item()
                diag.per_memory_bidi_min[mem.mid]=bidi_min_t[mi].item()
                diag.per_memory_sem_sim[mem.mid]=sem_sim_t[mi].item()
                diag.per_memory_forward_maxsim_idf[mem.mid]=forward_idf_t[mi].item()

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
                for m in mems:
                    m.last=self.time; m.cnt+=1
            final_scores=0.5*rerank_scores+0.5*forward_idf_t if (self.c.use_idf_retrieval and forward_idf_t.max().item()>0) else rerank_scores
            w=F.softmax(final_scores/self.c.retrieval_weight_temperature,dim=0)
            fs=(transported*w.unsqueeze(-1)).sum(0)
            batch_mw=[(m.mid,w[mi].item()) for mi,m in enumerate(mems)]
            all_batch_mw.append(batch_mw)
            all_dominant.append(dominant_mid)
            all_results.append(transported); all_masks.append(torch.ones(C,**_dev(xq)))
            all_biases.append(final_scores/self.c.tau); all_summaries.append(fs)

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
        diag.dominant_per_batch=all_dominant
        if diag.dominant_per_batch and diag.dominant_per_batch[0] is not None:
            diag.dominant_memory_id=diag.dominant_per_batch[0]
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
                    if not self._check_consolidation_compatible(
                            ms[i].content_token_ids, ms[j].content_token_ids):
                        continue
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
                    merged.add(ms[j].mid)
        for mid in merged: del self.tree.store[mid]
        if merged: self.tree.rebuild()
        return len(merged)

# ═══════════════════════════════════════════════════════════════════
# 第18部分 · MemLLM (v3.16: 移除 GPT-2 特化, 语义门控)
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
        self._wte_normed: Optional[torch.Tensor] = None

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
        self._wte_normed=F.normalize(
            self.llm.transformer.wte.weight.detach(),dim=-1,eps=1e-8)
        self.amm.wte_normed=self._wte_normed
        self._build_wte_neighbor_cache()

    def _build_wte_neighbor_cache(self):
        if self.llm is None or self.content_classifier is None: return
        wte_n=self._wte_normed
        cc=self.content_classifier
        content_list=sorted(cc.content_ids)
        valid=[t for t in content_list if t<wte_n.shape[0]]
        self._wte_neighbor_cache={}
        K=self.c.wte_neighbor_k; thresh=self.c.wte_neighbor_threshold
        batch_size=500
        for start in range(0,len(valid),batch_size):
            batch_ids=valid[start:start+batch_size]
            batch_t=torch.tensor(batch_ids,device=wte_n.device)
            batch_vecs=wte_n[batch_t]
            sims=batch_vecs@wte_n.T
            topk_vals,topk_ids=sims.topk(K+1,dim=-1)
            for i,tid in enumerate(batch_ids):
                neighbors=[]
                for v_val,nid in zip(topk_vals[i],topk_ids[i]):
                    nid_int=nid.item()
                    if nid_int==tid: continue
                    if v_val.item()>=thresh and nid_int in cc.content_ids:
                        neighbors.append(nid_int)
                self._wte_neighbor_cache[tid]=neighbors

    def _expand_content_ids(self, content_ids: List[int]) -> List[int]:
        if not self._wte_neighbor_cache: return content_ids
        expanded=set(content_ids)
        for tid in content_ids:
            neighbors=self._wte_neighbor_cache.get(tid,[])
            expanded.update(neighbors)
        return list(expanded)

    def _compute_content_semantic_emb(self, hidden_states, ids, mask):
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

    def _compute_tfidf_idf(self) -> Dict[int,float]:
        cc=self.content_classifier
        if cc is None:
            return {}
        return self.amm._compute_corpus_idf(cc)

    def _compute_tfidf_weights(self, diag, query_content_ids_per_batch, dominant_only=True):
        cc=self.content_classifier
        if cc is None:
            return []
        V=self.c.vocab_size
        wte_n=self._wte_normed
        idf=self._compute_tfidf_idf() if self.c.use_tfidf_weighting else {}
        B=len(diag.batch_mem_weights)
        result=[]
        for b in range(B):
            q_ids=(query_content_ids_per_batch[b]
                   if query_content_ids_per_batch and b<len(query_content_ids_per_batch)
                   else [])
            q_valid=[i for i in q_ids if i<wte_n.shape[0]]
            if q_valid and idf:
                q_idf_weights=torch.tensor(
                    [max(idf.get(i, self.c.idf_floor), self.c.idf_floor) for i in q_valid],
                    device=wte_n.device)
                q_vecs=wte_n[q_valid]*q_idf_weights.unsqueeze(1)
            else:
                q_vecs=wte_n[q_valid] if q_valid else None
            target_mids=[]
            if (dominant_only and diag.dominant_per_batch and b<len(diag.dominant_per_batch)
                    and diag.dominant_per_batch[b] is not None):
                target_mids=[(diag.dominant_per_batch[b],1.0)]
            elif b<len(diag.batch_mem_weights) and diag.batch_mem_weights[b]:
                sorted_mw=sorted(diag.batch_mem_weights[b],key=lambda x:-x[1])
                target_mids=[sorted_mw[0]]
            token_weights={}
            for mid,base_w in target_mids:
                if mid not in self.amm.tree.store:
                    continue
                mem=self.amm.tree.store[mid]
                label_ids=[t for t in mem.content_token_ids
                           if t in cc.content_starter_ids and t<V and t<wte_n.shape[0]]
                if not label_ids:
                    continue
                label_t=torch.tensor(label_ids,device=wte_n.device)
                label_vecs=wte_n[label_t]
                if q_vecs is not None:
                    sim=label_vecs@q_vecs.T
                    sem_rel=sim.max(dim=1).values.clamp(min=0)
                else:
                    sem_rel=torch.ones(len(label_ids),device=wte_n.device)
                idf_vals=torch.tensor([idf.get(t,1.0) for t in label_ids],device=wte_n.device)
                final=base_w*sem_rel*idf_vals
                for i,t in enumerate(label_ids):
                    v=final[i].item()
                    if v>1e-8:
                        token_weights[t]=token_weights.get(t,0.0)+v
            if token_weights:
                mx=max(token_weights.values())
                if mx>1e-8:
                    token_weights={t:v/mx for t,v in token_weights.items()}
            result.append(token_weights)
        return result

    def _build_first_step_lexical_bias(self, diag, query_content_ids_per_batch):
        V=self.c.vocab_size; dev=next(self.parameters()).device
        B=len(diag.batch_mem_weights)
        bias=torch.zeros(B,V,device=dev)
        if not self.c.use_first_step_lexical:
            return bias
        weights_per_batch=self._compute_tfidf_weights(diag,query_content_ids_per_batch,dominant_only=True)
        K=self.c.first_step_lexical_topk
        for b in range(B):
            tw=weights_per_batch[b] if b<len(weights_per_batch) else {}
            if not tw:
                continue
            sorted_items=sorted(tw.items(),key=lambda x:-x[1])[:K]
            for tid,w in sorted_items:
                if 0<=tid<V:
                    bias[b,tid]=w
        return bias

    def _build_content_bias(self, diag, query_content_ids_per_batch):
        V=self.c.vocab_size; dev=next(self.parameters()).device
        B=len(diag.batch_mem_weights)
        bias=torch.zeros(B,V,device=dev)
        wte_n=self._wte_normed
        cc=self.content_classifier
        floor=self.c.content_bias_relevance_floor
        concentration=self.c.content_bias_concentration
        use_starter=self.c.use_word_starter_filter
        for b,mem_weights in enumerate(diag.batch_mem_weights):
            q_ids=(query_content_ids_per_batch[b]
                   if query_content_ids_per_batch and b<len(query_content_ids_per_batch)
                   else [])
            q_valid=[i for i in q_ids if i<wte_n.shape[0]]
            if q_valid:
                q_vecs=wte_n[q_valid]
            for mid,weight in mem_weights:
                if mid not in self.amm.tree.store: continue
                mem=self.amm.tree.store[mid]
                bidi_w = diag.per_memory_bidi_min.get(mid, 0.5)
                adjusted_weight = weight * (bidi_w ** 2)
                scoring_ids=self.amm._get_mem_scoring_ids(mem)
                if use_starter:
                    valid_ids=[t for t in scoring_ids
                               if t<V and t<wte_n.shape[0]
                               and cc is not None and t in cc.content_starter_ids]
                else:
                    valid_ids=[t for t in scoring_ids
                               if t<V and t<wte_n.shape[0]
                               and cc is not None and t in cc.content_ids]
                if not valid_ids: continue
                if q_valid:
                    m_vecs=wte_n[valid_ids]
                    sim=m_vecs@q_vecs.T
                    relevance=sim.max(dim=1).values.clamp(min=0)
                    relevance=relevance.pow(concentration)
                    relevance=relevance*(1.0-floor)+floor
                    for i,tid in enumerate(valid_ids):
                        bias[b,tid]+=adjusted_weight*relevance[i].item()
                else:
                    for tid in valid_ids:
                        bias[b,tid]+=adjusted_weight
            bmax=bias[b].max()
            if bmax>1e-8: bias[b]/=bmax
        return bias

    def _compute_content_wte_topk(self, diag, query_content_ids_per_batch):
        dev=next(self.parameters()).device
        wte=self.llm.transformer.wte.weight.detach()
        wte_n=self._wte_normed
        B=len(diag.batch_mem_weights)
        cc=self.content_classifier
        floor=self.c.content_bias_relevance_floor
        concentration=self.c.content_bias_concentration
        use_starter=self.c.use_word_starter_filter
        K=self.c.content_wte_topk_for_inject
        idf=self._compute_tfidf_idf() if self.c.use_tfidf_weighting else {}
        mean_results=[]; target_results=[]
        for b in range(B):
            q_ids=(query_content_ids_per_batch[b]
                   if query_content_ids_per_batch and b<len(query_content_ids_per_batch)
                   else [])
            q_valid=[i for i in q_ids if i<wte_n.shape[0]]
            weight_map={}
            dom_mid=(diag.dominant_per_batch[b]
                     if diag.dominant_per_batch and b<len(diag.dominant_per_batch)
                     else None)
            if dom_mid is not None and dom_mid in self.amm.tree.store:
                source_items=[(dom_mid,1.0)]
            else:
                source_items=diag.batch_mem_weights[b] if b<len(diag.batch_mem_weights) else []
            for mid,w in source_items:
                if mid not in self.amm.tree.store: continue
                mem=self.amm.tree.store[mid]
                bidi_w = diag.per_memory_bidi_min.get(mid, 0.5)
                adjusted_w = w * (bidi_w ** 2)
                scoring_ids=self.amm._get_mem_scoring_ids(mem)
                for tid in scoring_ids:
                    if tid>=wte.shape[0] or cc is None:
                        continue
                    if use_starter and tid not in cc.content_starter_ids:
                        continue
                    if (not use_starter) and tid not in cc.content_ids:
                        continue
                    weight_map[tid]=weight_map.get(tid,0.0)+adjusted_w
            if not weight_map:
                zero=torch.zeros(self.c.d_LLM,device=dev)
                mean_results.append(zero); target_results.append(zero.clone()); continue
            tids=list(weight_map.keys())
            tids_t=torch.tensor(tids,device=dev)
            weights_t=torch.tensor([weight_map[t] for t in tids],device=dev)
            if q_valid:
                q_vecs=wte_n[q_valid]
                m_vecs_n=wte_n[tids_t]
                sim=m_vecs_n@q_vecs.T
                relevance=sim.max(dim=1).values.clamp(min=0)
                relevance=relevance.pow(concentration)
                relevance=relevance*(1.0-floor)+floor
                weights_t=weights_t*relevance
            if idf:
                idf_t=torch.tensor([idf.get(t,1.0) for t in tids],device=dev)
                weights_t=weights_t*idf_t
            k_eff=min(K, tids_t.numel())
            top_vals, top_idx=weights_t.topk(k_eff)
            top_tids=tids_t[top_idx]
            total=top_vals.sum()
            if total>1e-8:
                top_wte=wte[top_tids]
                mean_results.append((top_wte*top_vals.unsqueeze(1)).sum(0)/total)
            else:
                mean_results.append(wte[top_tids].mean(0))
            target_tid=tids_t[weights_t.argmax()]
            target_results.append(wte[target_tid])
        return torch.stack(mean_results), torch.stack(target_results)

    def _compute_domain_anchors(self, content_bias, k=None):
        k=k or self.c.domain_anchor_k
        B=content_bias.shape[0]
        anchors=[]
        for b in range(B):
            vals,ids=content_bias[b].topk(min(k,content_bias.shape[1]))
            anchor_set=[]
            for v,tid in zip(vals,ids):
                if v.item()>1e-6:
                    anchor_set.append(tid.item())
            anchors.append(anchor_set)
        return anchors

    def _get_prefix(self, hs, mask=None, pl=0, update_stats=True,
                    return_extra=False, ids=None):
        pooled,xq,fq=self.extract_state(hs,mask,pl)
        trimmed_mask=mask[:,pl:] if mask is not None and pl>0 else mask
        if trimmed_mask is not None and pooled.shape[1]!=trimmed_mask.shape[1]:
            trimmed_mask=None
        query_content_ids_per_batch=[]
        if ids is not None and self.content_classifier is not None:
            for b in range(ids.shape[0]):
                b_ids=ids[b].tolist()
                b_exact=list(set(self.content_classifier.get_content_ids_from_tokens(b_ids)))
                query_content_ids_per_batch.append(b_exact)
        if ids is not None and self.content_classifier is not None:
            query_sem=self._compute_content_semantic_emb(pooled,ids,trimmed_mask)
        else:
            query_sem=pooled.mean(1)
        wte_n=self._wte_normed
        fibers,mem_mask,fiber_summary,diag=self.amm.retrieve_multi(
            xq,fq,update_stats=update_stats,
            query_semantic_emb=query_sem,
            query_content_ids_per_batch=query_content_ids_per_batch,
            wte_normed=wte_n,
            content_classifier=self.content_classifier)
        content_wte_mean, content_target_wte=self._compute_content_wte_topk(
            diag,query_content_ids_per_batch)
        has_cwm=content_wte_mean.abs().max().item()>1e-6
        has_tgt=content_target_wte.abs().max().item()>1e-6
        prefix=self.bridge.inject(fibers,mem_mask,fiber_summary=fiber_summary,
                                  content_wte_mean=content_wte_mean if has_cwm else None,
                                  content_target_wte=content_target_wte if has_tgt else None)
        if return_extra:
            content_bias=self._build_content_bias(diag,query_content_ids_per_batch)
            first_step_bias=self._build_first_step_lexical_bias(diag,query_content_ids_per_batch)
            return prefix,fiber_summary,diag,content_bias,first_step_bias
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
        content_sem=self._compute_content_semantic_emb(hs_pooled,ids,mask)
        raw_ids=self.tok.encode(text)
        cc=self.content_classifier
        content_ids=list(set(cc.get_content_ids_from_tokens(raw_ids))) if cc else []
        expanded_ids=self._expand_content_ids(content_ids)
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
            prefix,fiber_summary,_,content_bias,first_step_bias=self._get_prefix(
                o['hs'],mask,update_stats=True,return_extra=True,ids=ids)
            vocab_bias=self._compute_vocab_bias(fiber_summary)
        has_content=content_bias is not None and content_bias.abs().max().item()>0.01
        has_first_step=first_step_bias is not None and first_step_bias.abs().max().item()>1e-6
        cc=self.content_classifier
        domain_anchors=self._compute_domain_anchors(content_bias) if has_content else [[]]
        anchors_for_b0=set(domain_anchors[0]) if domain_anchors else set()
        generated_anchors=set()
        generated_ids=[]
        generated_content_counts: Dict[int,int] = {}
        consecutive_content=0
        recent_starters: List[Tuple[int,int]] = []
        for i in range(mt):
            if i>0 and i%self.c.retrieval_interval==0:
                with torch.no_grad():
                    o=self.fwd(ids,mask,prefix); pl=o['pl']
                    prefix,fiber_summary,_,content_bias,first_step_bias=self._get_prefix(
                        o['hs'],o['mask'],pl,update_stats=True,return_extra=True,ids=ids)
                    vocab_bias=self._compute_vocab_bias(fiber_summary)
                    has_content=content_bias is not None and content_bias.abs().max().item()>0.01
                    has_first_step=first_step_bias is not None and first_step_bias.abs().max().item()>1e-6
                    if has_content:
                        domain_anchors=self._compute_domain_anchors(content_bias)
                        anchors_for_b0=set(domain_anchors[0]) if domain_anchors else set()
            with torch.no_grad():
                o=self.fwd(ids,mask,prefix); lg=o['logits'][:,-1:].squeeze(1).clone()
                step_scale_content=max(self.c.content_bias_floor,
                                       1.0-i*self.c.content_bias_decay)
                step_scale_learned=max(self.c.semantic_boost_floor,
                                       1.0-i*self.c.semantic_boost_decay)
                if i==0:
                    effective_content_scale=step_scale_content*self.c.first_step_content_multiplier
                elif consecutive_content>=self.c.structural_rhythm_threshold:
                    effective_content_scale=step_scale_content*0.25
                    if cc:
                        for fid in list(cc.function_ids)[:5000]:
                            if fid<lg.shape[-1]:
                                lg[0,fid]+=self.c.structural_boost
                else:
                    effective_content_scale=step_scale_content
                if has_first_step and i<self.c.first_step_lexical_decay_steps:
                    V_fs=min(lg.shape[-1],first_step_bias.shape[-1])
                    lg[:,:V_fs]=lg[:,:V_fs]+first_step_bias[:,:V_fs]*self.c.first_step_lexical_scale
                if has_content:
                    cb_adjusted=content_bias.clone()
                    for tid,count in generated_content_counts.items():
                        if tid<cb_adjusted.shape[-1]:
                            decay=self.c.generated_token_decay**count
                            cb_adjusted[0,tid]*=decay
                    V=min(lg.shape[-1],cb_adjusted.shape[-1])
                    lg[:,:V]=lg[:,:V]+cb_adjusted[:,:V]*self.c.content_bias_scale*effective_content_scale
                if vocab_bias is not None:
                    V2=min(lg.shape[-1],vocab_bias.shape[-1])
                    lg[:,:V2]=lg[:,:V2]+vocab_bias[:,:V2]*self.c.semantic_boost_scale*step_scale_learned
                if i==0 and cc is not None:
                    cmask=(cc.content_starter_mask(dev)
                           if self.c.use_word_starter_filter else cc.content_mask(dev))
                    V3=min(lg.shape[-1],cmask.shape[0])
                    lg[0,:V3]=lg[0,:V3]+cmask[:V3]*self.c.universal_content_boost
                elif i<self.c.universal_content_boost_steps and cc is not None and has_content:
                    cmask=(cc.content_starter_mask(dev)
                           if self.c.use_word_starter_filter else cc.content_mask(dev))
                    V3=min(lg.shape[-1],cmask.shape[0])
                    boost_scale=1.0-i/self.c.universal_content_boost_steps
                    lg[0,:V3]=lg[0,:V3]+cmask[:V3]*self.c.universal_content_boost*boost_scale
                if (i>=self.c.domain_anchor_start_step and anchors_for_b0 and has_content):
                    coverage=len(generated_anchors)/max(len(anchors_for_b0),1)
                    if coverage<self.c.domain_anchor_coverage_threshold:
                        unvisited=anchors_for_b0-generated_anchors
                        for tid in unvisited:
                            if tid<lg.shape[-1]:
                                lg[0,tid]+=self.c.domain_anchor_boost
                if cc:
                    for tid,count in generated_content_counts.items():
                        if tid in cc.content_ids and tid<lg.shape[-1]:
                            lg[0,tid]-=self.c.content_repeat_penalty*count
                if cc and self._wte_neighbor_cache is not None and recent_starters:
                    for prev_tid,_prev_step in recent_starters:
                        for nid in self._wte_neighbor_cache.get(prev_tid,[]):
                            if nid in cc.word_starter_ids:
                                continue
                            if nid<lg.shape[-1]:
                                lg[0,nid]-=self.c.bpe_echo_penalty
                if cc and generated_ids and generated_ids[-1] in cc.content_starter_ids:
                    for tid in cc.content_ids:
                        if tid not in cc.word_starter_ids and tid<lg.shape[-1]:
                            lg[0,tid]-=self.c.post_starter_nonstarter_penalty
                if self._degen_guard is not None:
                    penalty_mult=self.c.first_step_penalty_multiplier if i==0 else 1.0
                    lg=self._degen_guard.process(
                        lg,generated_ids,i,first_step_penalty_mult=penalty_mult)
                if i<self.c.early_content_steps and cc is not None:
                    for pid in cc.punct_ids:
                        if pid<lg.shape[-1]: lg[0,pid]=-float('inf')
                    for nid in cc.newline_ids:
                        if nid<lg.shape[-1]: lg[0,nid]=-float('inf')
                if i==0 and cc is not None:
                    for fid in cc.filler_ids:
                        if fid<lg.shape[-1]:
                            lg[0,fid]-=self.c.step0_filler_penalty
                if greedy:
                    nxt=lg.argmax(-1,keepdim=True)
                else:
                    lg=lg/self.c.gen_temp; p=F.softmax(lg,-1)
                    sp,si=torch.sort(p,descending=True); cs=torch.cumsum(sp,-1)
                    rm=cs-sp>self.c.gen_top_p; sp[rm]=0
                    total=sp.sum(-1,keepdim=True)
                    if (total<1e-10).any(): sp[:,0]=1.0; total=sp.sum(-1,keepdim=True)
                    sp=sp/total; nxt=si.gather(-1,torch.multinomial(sp,1))
            nxt_id=nxt.item()
            if nxt_id==self.tok.eos_token_id and len(generated_ids)>=self.c.degen_min_tokens: break
            generated_ids.append(nxt_id)
            if cc and nxt_id in cc.content_ids:
                generated_content_counts[nxt_id]=generated_content_counts.get(nxt_id,0)+1
                consecutive_content+=1
                if nxt_id in anchors_for_b0:
                    generated_anchors.add(nxt_id)
                if nxt_id in cc.word_starter_ids:
                    recent_starters.append((nxt_id,i))
            else:
                consecutive_content=0
            recent_starters=[(t,s) for (t,s) in recent_starters if (i-s)<self.c.bpe_echo_window]
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
                'semantic_emb':m.semantic_emb.cpu() if m.semantic_emb is not None else None}
        torch.save(data,path)

    def load_memory(self, path):
        data=torch.load(path,weights_only=False)
        self.amm.tree.store.clear(); self.amm.tree.root=_Node()
        self.amm.tree.nid=data['nid']; self.amm.time=data['time']
        dev=next(self.parameters()).device
        for mid,d in data['store'].items():
            sem=d.get('semantic_emb',None)
            if sem is not None: sem=sem.to(dev)
            m=MemEntry(mid=mid,base=d['base'].to(dev),fiber=d['fiber'].to(dev),
                dirn=d['dirn'].to(dev),surprise=d['surprise'],ts=d['ts'],
                last=d['last'],cnt=d['cnt'],version=d['version'],
                source_text=d.get('source_text',''),
                content_token_ids=d.get('content_token_ids',[]),
                expanded_content_ids=d.get('expanded_content_ids',[]),
                semantic_emb=sem)
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

    def recon(self, text):
        return self._recon_forward(text)

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
