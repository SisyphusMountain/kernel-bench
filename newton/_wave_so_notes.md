# STATUS: steps 1-4 DONE+gated (e_step_so ~3e-9, wave_so ~3e-9 split+leaf, dts_so ~3e-9 eq1+ge2;
# gates: python -m newton.verify {e_so,wave_so,dts_so} small). NEXT: compose hvp(u) in
# newton/hvp_exact.py::make_exact_hvp:
# 1. tangent fwd: param_jvp_uniform -> dP; e_tangent_fixed_point -> dE*,dEs1,dEs2,dEbar;
#    jvp_root_scores REFACTORED to return full dPi/dPibar buffers + per-wave d_dts (+dcst from
#    _wave_tangent_constants: dDL,dEbar,dE,dSL1,dSL2,dMC,dleaf,dpd_param,dps_param).
# 2. d_seed at root rows = -ln2*(q*t - q*(q.t)) with t=dPi[root]; d_rhs [C,S] buffer.
# 3. reverse sweep over cache['waves'] (already reverse order): (a) wave_backward_so -> d_Av +
#    d_aw contraction; (b) dv = wave_backward_uniform_fused(seed=d_rhs_k + d_Av, SAME cached
#    active_mask/settings) -> (dv, aw(dv) = B^T dv); (c) d_aw_total = contraction + aw(dv),
#    scatter_accum mapping same as primal (pD+=aw0, pS+=aw345, E+=aw0+aw2, Ebar+=aw1, Es1+=aw4,
#    Es2+=aw3, mt+=aw2); (d) dts: primal dts_cross_backward+tree applied to dv (C^T dv into
#    d_rhs, params, using cached active_mask) PLUS dts_backward_so at fixed v_k (already writes
#    d_rhs/params/tree-contribs).
# 4. E-side: dq_E = d_grad_E_acc + bwd(x; dg=(0,d_gEs1,d_gEs2,d_gEbar)) [e_step_triton_autograd
#    with tangent cotangents] + e_step_backward_so(x; g=(0,gEs1,gEs2,gEbar); dx)[grad_E out]
#    + d[norm-term grad] closed form: g_norm_j = -n_fam*2^{E_j}/(S*norm), d = -n_fam*ln2*2^{E_j}
#    dE_j/(S*norm) + n_fam*2^{E_j}*dnorm/(S*norm^2), dnorm=-ln2*mean(2^E dE).
# 5. dwE = _bicgstab(SAME AG_flat operator, rhs = dq_E + e_step_so(x; g=(wE,0,0,0); dx)[grad_E]).
# 6. theta head (smooth, autograd OK): primal grad_theta = (dP/dtheta)^T cot_P with cot_P =
#    (grad_pS,grad_pD,grad_pL?,grad_mt,grad_col buckets) + e-step bwd at (E*.detach(),P) with
#    cotangents (g_new=wE, g_ebar=grad_Ebar_acc) param outputs. Tangent: d_grad_theta =
#    jvp of theta->(dP/dtheta)^T cot_P at fixed cot_P (torch.func.jvp, smooth log_softmax only)
#    + (dP/dtheta)^T d[cot_P], d[cot_P] = wave-side d accumulators + e-step bwd with tangent
#    cotangents (g_new=dwE, g_ebar=d_gEbar) + e_step_so(g=(wE,0,0,grad_Ebar_acc); dx) params.
# 7. Gates: fp64 analytic vs fp64 FD-of-grad HVP <=1e-6; symmetry; fp32 vs fp64 ~1e-3; then
#    integration --hvp exact (newton_lanczos/pipeline/driver) + benchmarks + 1007x64 fp32.
# NOTE dx for e-step pieces = (dE*, dE_new=dE* at fixed point? NO: E_new input is Phi(E*) whose
# tangent at the fixed point = dE* as well since dE* solves the tangent fixed point; pass
# dE_new=dE*, dE_s1/dE_s2 = child gathers of dE* (e_tangent returns these), dEbar from e_tangent).

# wave_so derivation notes (extracted from wave_backward.py — for the exact-HVP step 3/4)

Forward terms per (w,s) (wave_step.py:789-809): t0=dl+Pi[s], t1=Pi[s]+ebar, t2=pibar[s]+e,
t3=sl1+Pi[c1], t4=sl2+Pi[c2], t5=leaf (leaf_hit only), t6=dts_r; Pi_new=lse2(t0..t6);
pibar[s]=log2(row_sum−anc_or_self_sum(2^Pi))+row_max+mc.

## Backward precompute (wave_backward.py:283-483), at Pi*
e_k=2^{t_k−m} over t0..t5, inv_sum=1/Σe_k(0..5); dts_l=log2(Σ)+m; w_L=2^{dts_l−Pi_new} (1 if no
splits). Coeffs: diag_wt=w_L(e0+e1)inv_sum; pibar_u_coeff=w_L·e2·inv_sum·inv_denom
(inv_denom=1/(row_sum−anc_sum)); sl1_wt=w_L·e3·inv_sum; sl2_wt=w_L·e4·inv_sum;
p_prime=2^{(col+)Pi−row_max}.

## J^T action per Neumann iter (486-637): given v:
u_d[s]=v[s]·pibar_u_coeff[s]; A=Σ_s u_d; subtree_sum[s]=Σ_{j∈subtree(s)} u_d[j] (DFS cumsum or
level walk); result[s]= v[s]·diag_wt[s] + p_prime[s]·(A−subtree_sum[s]) + v[parent[s]]·edge_wt[s]
(or scatter v[s]·sl1_wt→c1, v[s]·sl2_wt→c2). Neumann: v←rhs+result.

## aw outputs (1049-1268): alpha=v_k·w_L; aw_k=alpha·e_k·inv_sum (k=0..5), aw345=aw3+aw4+aw5.
Maps: grad_log_pD+=Σ_w aw0; grad_log_pS+=Σ_w aw345; grad_E[s]+=Σ_w(aw0+aw2); grad_Ebar+=aw1;
grad_E_s1+=aw4; grad_E_s2+=aw3; grad_mt+=aw2. Leaf t5 → inside aw345 only.

## dts backward (1868-2285): per split (l,r,parent w): d0=log_pD+Pi_l+Pi_r, d1=Pi_l+Pibar_r,
d2=Pi_r+Pibar_l, d3=log_pS+Pi_l[c1]+Pi_r[c2], d4=log_pS+Pi_r[c1]+Pi_l[c2];
w_k=2^{lsp+d_k−Pi_parent_new[w,s]} (NOTE: normalized by parent Pi_new directly — the v·w6 factor
is folded in: vd_k = v_k[w,s]·w_k). rhs scatter: Pi_l+=vd0+vd1; Pi_r+=vd0+vd2; Pi_l[c1]+=vd3;
Pi_r[c1]+=vd4; Pi_r[c2]+=vd3; Pi_l[c2]+=vd4. Pibar cotangents (ud-staged):
ud_l=vd2·2^{row_max_l+mt_l−Pibar_l}, ud_r=vd1·2^{row_max_r+mt_r−Pibar_r}; pibar_A=Σ_s ud.
Params: grad_log_pD+=Σvd0; grad_log_pS+=Σ(vd3+vd4); grad_mt+=vd1+vd2 (on l/r rows).

## pibar tree VJP (2569-2746): given ud, A per child row: subtree reduce ud; contrib[s] =
p_prime_child[s]·(A−subtree_sum[s]) with p_prime_child=2^{(col+)Pi_child−pibar_row_max};
accumulated_rhs[child,s]+=contrib; grad_col[s]+=contrib.

## Second-order strategy (per plan): outputs are LINEAR in v/cotangents ⇒
d[J^T v] = J^T(dv-part handled by reusing solve with modified seed) + dJ^T·v (new contraction).
Contraction needs d of all coeffs at fixed v: de_k=ln2·e_k·(dt_k−dm→use dΨ form), better:
dw̃_k where w̃_k=e_k·inv_sum=2^{t_k−dts_l}: dw̃_k=ln2·w̃_k·(dt_k−d_dts_l), d_dts_l=Σw̃_j dt_j.
dw_L=ln2·w_L·(d_dts_l−dPi_new), dPi_new=tangent of full lse incl dts: =w_L·d_dts_l+(1−w_L)·d_dts_r.
dp_prime=ln2·p_prime·(dcol+dPi) (row_max frozen — normalizer invariance, same as e_step_so).
d_inv_denom=−inv_denom²·d(denom), d(denom)=Σ_j ln2·p_prime-like dPi_j terms (row scope).
dt_k tangents: dt0=dDL+dPi[s], dt1=dPi[s]+dEbar_c, dt2=dpibar[s]+dE_c, dt3=dSL1+dPi[c1],
dt4=dSL2+dPi[c2], dt5=dleaf, dt6=d_dts_r; dpibar from forward tangent (dPibar buffer).
Then: d(result)[s] = v[s]·d(diag_wt) + dp_prime·(A−sub) + p_prime·(dA−dsub) [dA from
du_d=v·d(pibar_u_coeff)] + v[parent]·d(edge_wt). d(aw_k)=v·d(w_L·e_k·inv_sum)=v·(dw_L·w̃_k+w_L·dw̃_k).
dts: d(vd_k)=v·dw_k with dw_k=ln2·w_k·(dlsp=0+dd_k−dPi_parent_new); d(ud)=dvd·2^{...}+vd·ln2·2^{...}·
(drow_max frozen+dmt−dPibar). Gate trick: FD the frozen kernels under perturbed (Pi*,Pibar*,
consts,dts_r) with FIXED seed rhs/v.

## Caches available (hvp_exact.build_point_cache): per wave (reverse order) v_k, dts_r,
active_mask, meta; accum grads; e_side q_E/wE/aux_to_e. Gate passed (golden grad 9e-6).

## Verified details from precompute kernel (wave_backward.py:283-482):
- p_prime = 2^{(col+)Pi − pibar_row_max[row]} (NORMALIZER = saved pibar_row_max, NOT a live max
  → freeze its tangent; outputs invariant: pibar_u_coeff·p_prime products cancel the 2^{±rm}).
- No ln2 anywhere: log2-space lse derivative is exactly w̃_k = e_k·inv_sum; and
  ∂pibar_s/∂Π_j = p_prime_j·inv_denom_s for j∉path(s) (ln2 from d2^x cancels ln2 from dlog2).
- subtree_sum[j] = Σ_{s∈subtree(j)} u_d_s (transpose of ancestor walk → implement via
  MAX_ANCESTOR_DEPTH atomic walk scattering u_d_s up path(s), like e_step_so excluded_u).
- ancestor_sum (for denom) = ancestor-OR-SELF path sums of p_prime (binary lifting w/ jump
  table from wave_step._get_jumps).
- Child edges: result contribution scatters v_s·sl1_wt_s → c1[s], v_s·sl2_wt_s → c2[s].
- CONTRACTION IS APPLIED ONCE at converged cached v_k: dv = (I−Aᵀ)⁻¹[d_rhs + d(Aᵀ)v_k];
  the reused solve handles (I−Aᵀ)⁻¹ and its aw* outputs give Bᵀdv.
- Tangent formulas (m and pibar_row_max frozen): dw̃_k = ln2·w̃_k·(dt_k − dlse), dlse=Σw̃_j dt_j;
  dw_L = ln2·w_L·(1−w_L)·(dlse − d_dts_r); dp_prime = ln2·p_prime·dPi;
  ddenom_s = Σ_{row} ln2·p_prime_j·dPi_j − Σ_{path(s)} ln2·p_prime_a·dPi_a;
  d(pibar_u_coeff) = [dw_L·w̃2 + w_L·dw̃2]·inv_denom − pibar_u_coeff·inv_denom·ddenom;
  d(diag_wt) = dw_L(w̃0+w̃1) + w_L(dw̃0+dw̃1); d(sl_wt) analogous.
  d(Aᵀv)[j] = v_j·d(diag_wt_j) + dp_prime_j·(A−sub_j) + p_prime_j·(dA−dsub_j)
             + Σ_{s: c1[s]=j} v_s·d(sl1_wt_s) + Σ_{s: c2[s]=j} v_s·d(sl2_wt_s)
    with u_d=v·pibar_u_coeff, du_d=v·d(pibar_u_coeff), A=Σu_d, dA=Σdu_d, sub/dsub subtree sums.
  d_aw_k = v·[dw_L·w̃_k + w_L·dw̃_k]  (param tangents; same bucket mapping as primal aw_k).
- dt_k: dt0=dDL+dPi_s, dt1=dPi_s+dEbar_c, dt2=dPibar_s+dE_c, dt3=dSL1+dPi_{c1},
  dt4=dSL2+dPi_{c2}, dt5=dleaf (leaf_hit), d_dts_r given. NOTE dPibar_s here is the tangent of
  the SAVED Pibar entering t2 (from forward-tangent buffer), distinct from pibar recomputed
  from Π within the same wave — the backward uses saved Pibar_star for t2 but routes the pibar
  cotangent through Π of the SAME row via p_prime/u_d (self-loop). Both views consistent at the
  fixed point since Pibar* = pibar(Π*).
