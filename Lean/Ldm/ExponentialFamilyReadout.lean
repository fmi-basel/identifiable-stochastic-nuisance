import Mathlib

set_option linter.style.header false
set_option linter.style.haveILetI false
set_option linter.unusedSectionVars false
set_option linter.unusedFintypeInType false
set_option linter.unusedDecidableInType false

/-!
# Exponential-family statistic readout with view-private nuisance

This file proves the manuscript's end-to-end nuisance-LDM theorem.  It starts
from the four conditional independences used there in place of the
mutual-information equality.  A common forward kernel and a common reverse
kernel cancel from the Radon--Nikodym ratios.  Finitely many source
natural-parameter differences that span the source statistic space then give
an affine readout of the source sufficient statistic from the learned
sufficient statistic.  The auxiliary cancellation and rebasing results are
private implementation details of this theorem.
-/

open Function MeasureTheory ProbabilityTheory Module
open scoped RealInnerProductSpace

namespace Ldm

variable {ι κ : Type*} [Fintype ι] [Fintype κ]
  [DecidableEq ι] [DecidableEq κ]

private noncomputable def naturalExpWeight
    {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (partition : E → ℝ) (a x : E) : ℝ :=
  Real.exp (inner ℝ a x - partition a)

/-- The natural exponential-family measure obtained by tilting a carrier
measure through a sufficient statistic. -/
noncomputable def statisticNaturalExpMeasure
    {X E : Type*} [MeasurableSpace X]
    [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (carrier : Measure X) (statistic : X → E) (partition : E → ℝ)
    (parameter : E) : Measure X :=
  carrier.withDensity fun x ↦
    ENNReal.ofReal (naturalExpWeight partition parameter (statistic x))

private lemma measurable_naturalExpWeight
    {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    [MeasurableSpace E] [BorelSpace E]
    (partition : E → ℝ) (a : E) :
    Measurable (naturalExpWeight partition a) := by
  unfold naturalExpWeight
  fun_prop

private lemma map_compProd_fst_measurableEquiv
    {A B C : Type*}
    [MeasurableSpace A] [MeasurableSpace B] [MeasurableSpace C]
    (e : A ≃ᵐ B) (μ : Measure A) (κ : Kernel A C)
    [SFinite μ] [IsSFiniteKernel κ] :
    Measure.map (Prod.map e id) (μ ⊗ₘ κ) =
      Measure.map e μ ⊗ₘ κ.comap e.symm e.symm.measurable := by
  ext s hs
  rw [Measure.map_apply (by fun_prop) hs, Measure.compProd_apply
      (hs.preimage (by fun_prop)), Measure.compProd_apply hs,
    MeasureTheory.lintegral_map_equiv]
  congr with a
  rw [Kernel.comap_apply]
  rw [e.symm_apply_apply]
  congr 1

private lemma eventuallyEq_comap_measurableEquiv
    {A B C : Type*}
    [MeasurableSpace A] [MeasurableSpace B] [MeasurableSpace C]
    (e : A ≃ᵐ B) (μ : Measure A) (κ η : Kernel B C)
    (h : κ =ᵐ[μ.map e] η) :
    κ.comap e e.measurable =ᵐ[μ] η.comap e e.measurable := by
  rw [Filter.EventuallyEq, e.measurableEmbedding.ae_map_iff,
    ← Filter.EventuallyEq] at h
  change (fun x ↦ κ (e x)) =ᵐ[μ] fun x ↦ η (e x)
  exact h

private lemma compProd_eventuallyEq_of_eventuallyEq
    {U A B : Type*}
    [MeasurableSpace U] [MeasurableSpace A] [MeasurableSpace B]
    (μ : Measure U) (κ : Kernel U A) (η η' : Kernel (U × A) B)
    [SFinite μ] [IsSFiniteKernel κ]
    [IsSFiniteKernel η] [IsSFiniteKernel η']
    (h : η =ᵐ[μ ⊗ₘ κ] η') :
    κ ⊗ₖ η =ᵐ[μ] κ ⊗ₖ η' := by
  have h_ae_ae : ∀ᵐ u ∂μ, ∀ᵐ a ∂κ u, η (u, a) = η' (u, a) :=
    Measure.ae_ae_of_ae_compProd h
  filter_upwards [h_ae_ae] with u hu
  rw [Kernel.compProd_apply_eq_compProd_sectR,
    Kernel.compProd_apply_eq_compProd_sectR]
  exact Measure.compProd_congr hu

private lemma condDistrib_prod_swap_ae
    {Ω A B C : Type*}
    [MeasurableSpace Ω]
    [MeasurableSpace A] [MeasurableSpace B]
    [MeasurableSpace C] [StandardBorelSpace C] [Nonempty C]
    (P : Measure Ω) [IsFiniteMeasure P]
    (X : Ω → A) (Y : Ω → B) (Z : Ω → C)
    (hX : Measurable X) (hY : Measurable Y) (hZ : Measurable Z) :
    condDistrib Z (fun ω ↦ (X ω, Y ω)) P =ᵐ[
        P.map (fun ω ↦ (X ω, Y ω))]
      (condDistrib Z (fun ω ↦ (Y ω, X ω)) P).comap
        MeasurableEquiv.prodComm MeasurableEquiv.prodComm.measurable := by
  apply condDistrib_ae_eq_of_measure_eq_compProd
      (fun ω ↦ (X ω, Y ω)) hZ.aemeasurable
  let e : B × A ≃ᵐ A × B := MeasurableEquiv.prodComm
  let e' : (B × A) × C ≃ᵐ (A × B) × C :=
    e.prodCongr (MeasurableEquiv.refl C)
  calc
    P.map (fun ω ↦ ((X ω, Y ω), Z ω)) =
        Measure.map e' (P.map (fun ω ↦ ((Y ω, X ω), Z ω))) := by
      rw [Measure.map_map (by fun_prop) (by fun_prop)]
      rfl
    _ = Measure.map e'
        (P.map (fun ω ↦ (Y ω, X ω)) ⊗ₘ
          condDistrib Z (fun ω ↦ (Y ω, X ω)) P) := by
      rw [compProd_map_condDistrib hZ.aemeasurable]
    _ = P.map (fun ω ↦ (X ω, Y ω)) ⊗ₘ
        (condDistrib Z (fun ω ↦ (Y ω, X ω)) P).comap
          MeasurableEquiv.prodComm MeasurableEquiv.prodComm.measurable := by
      change Measure.map (Prod.map e id)
          (P.map (fun ω ↦ (Y ω, X ω)) ⊗ₘ
            condDistrib Z (fun ω ↦ (Y ω, X ω)) P) = _
      rw [map_compProd_fst_measurableEquiv]
      congr 1
      rw [Measure.map_map (by fun_prop) (by fun_prop)]
      rfl

private lemma condDistrib_prod_ae_eq_compProd
    {Ω U A B : Type*}
    [MeasurableSpace Ω] [MeasurableSpace U]
    [MeasurableSpace A] [StandardBorelSpace A] [Nonempty A]
    [MeasurableSpace B] [StandardBorelSpace B] [Nonempty B]
    (P : Measure Ω) [IsFiniteMeasure P]
    (history : Ω → U) (X : Ω → A) (Y : Ω → B)
    (hhistory : Measurable history) (hX : Measurable X) (hY : Measurable Y) :
    condDistrib (fun ω ↦ (X ω, Y ω)) history P =ᵐ[P.map history]
      condDistrib X history P ⊗ₖ
        condDistrib Y (fun ω ↦ (history ω, X ω)) P := by
  apply condDistrib_ae_eq_of_measure_eq_compProd history
      (hX.prodMk hY).aemeasurable
  calc
    P.map (fun ω ↦ (history ω, (X ω, Y ω))) =
        Measure.map MeasurableEquiv.prodAssoc
          (P.map (fun ω ↦ ((history ω, X ω), Y ω))) := by
      rw [Measure.map_map (by fun_prop) (by fun_prop)]
      rfl
    _ = Measure.map MeasurableEquiv.prodAssoc
        (P.map (fun ω ↦ (history ω, X ω)) ⊗ₘ
          condDistrib Y (fun ω ↦ (history ω, X ω)) P) := by
      rw [compProd_map_condDistrib hY.aemeasurable]
    _ = Measure.map MeasurableEquiv.prodAssoc
        ((P.map history ⊗ₘ condDistrib X history P) ⊗ₘ
          condDistrib Y (fun ω ↦ (history ω, X ω)) P) := by
      rw [compProd_map_condDistrib hX.aemeasurable]
    _ = P.map history ⊗ₘ
        (condDistrib X history P ⊗ₖ
          condDistrib Y (fun ω ↦ (history ω, X ω)) P) := by
      rw [Measure.compProd_assoc']

private theorem condDistrib_prod_ae_eq_compProd_of_condIndep
    {Ω U A B : Type*}
    [MeasurableSpace Ω] [StandardBorelSpace Ω]
    [MeasurableSpace U] [StandardBorelSpace U] [Nonempty U]
    [MeasurableSpace A] [StandardBorelSpace A] [Nonempty A]
    [MeasurableSpace B] [StandardBorelSpace B] [Nonempty B]
    (P : Measure Ω) [IsFiniteMeasure P]
    (history : Ω → U) (X : Ω → A) (Y : Ω → B)
    (hhistory : Measurable history) (hX : Measurable X) (hY : Measurable Y)
    (hIndep : Y ⟂ᵢ[X, hX; P] history) :
    condDistrib (fun ω ↦ (X ω, Y ω)) history P =ᵐ[P.map history]
      condDistrib X history P ⊗ₖ
        (condDistrib Y X P).prodMkLeft U := by
  have hchain := condDistrib_prod_ae_eq_compProd
    P history X Y hhistory hX hY
  have hswap := condDistrib_prod_swap_ae
    P history X Y hhistory hX hY
  have hgiven :
      condDistrib Y (fun ω ↦ (X ω, history ω)) P =ᵐ[
          P.map (fun ω ↦ (X ω, history ω))]
        (condDistrib Y X P).prodMkRight U :=
    (condIndepFun_iff_condDistrib_prod_ae_eq_prodMkRight
      hY hhistory hX).mp hIndep.symm
  let e : U × A ≃ᵐ A × U := MeasurableEquiv.prodComm
  have hmap :
      (P.map (fun ω ↦ (history ω, X ω))).map e =
        P.map (fun ω ↦ (X ω, history ω)) := by
    rw [Measure.map_map (by fun_prop) (by fun_prop)]
    rfl
  rw [← hmap] at hgiven
  have hgivenComap := eventuallyEq_comap_measurableEquiv e
    (P.map (fun ω ↦ (history ω, X ω))) _ _ hgiven
  have hignore :
      (condDistrib Y (fun ω ↦ (X ω, history ω)) P).comap
          e e.measurable =ᵐ[P.map (fun ω ↦ (history ω, X ω))]
        (condDistrib Y X P).prodMkLeft U := by
    filter_upwards [hgivenComap] with ux hux
    rcases ux with ⟨u, x⟩
    change condDistrib Y (fun ω ↦ (X ω, history ω)) P (x, u) =
      condDistrib Y X P x at hux
    exact hux
  have heta :
      condDistrib Y (fun ω ↦ (history ω, X ω)) P =ᵐ[
          P.map (fun ω ↦ (history ω, X ω))]
        (condDistrib Y X P).prodMkLeft U :=
    hswap.trans hignore
  have hpair :
      P.map history ⊗ₘ condDistrib X history P =
        P.map (fun ω ↦ (history ω, X ω)) :=
    compProd_map_condDistrib hX.aemeasurable
  rw [← hpair] at heta
  exact hchain.trans
    (compProd_eventuallyEq_of_eventuallyEq
      (P.map history) (condDistrib X history P)
      (condDistrib Y (fun ω ↦ (history ω, X ω)) P)
      ((condDistrib Y X P).prodMkLeft U) heta)

private theorem conditionalJoint_common_forward_reverse_of_condIndep
    {Ω U E F : Type*}
    [MeasurableSpace Ω] [StandardBorelSpace Ω]
    [MeasurableSpace U] [StandardBorelSpace U] [Nonempty U]
    [MeasurableSpace E] [StandardBorelSpace E] [Nonempty E]
    [MeasurableSpace F] [StandardBorelSpace F] [Nonempty F]
    (P : Measure Ω) [IsFiniteMeasure P]
    (history : Ω → U) (signal : Ω → E) (code : Ω → F)
    (hhistory : Measurable history) (hsignal : Measurable signal)
    (hcode : Measurable code)
    (signal_history_given_code : signal ⟂ᵢ[code, hcode; P] history)
    (code_history_given_signal : code ⟂ᵢ[signal, hsignal; P] history) :
    let K := condDistrib code signal P
    let R := condDistrib signal code P
    ∀ᵐ u ∂P.map history,
      condDistrib (fun ω ↦ (signal ω, code ω)) history P u =
          condDistrib signal history P u ⊗ₘ K ∧
      condDistrib (fun ω ↦ (signal ω, code ω)) history P u =
          Measure.map MeasurableEquiv.prodComm
            (condDistrib code history P u ⊗ₘ R) ∧
      condDistrib signal history P u ⊗ₘ K =
          Measure.map MeasurableEquiv.prodComm
            (condDistrib code history P u ⊗ₘ R) := by
  dsimp only
  have hfwdKernel := condDistrib_prod_ae_eq_compProd_of_condIndep
    P history signal code hhistory hsignal hcode code_history_given_signal
  have hrevKernel := condDistrib_prod_ae_eq_compProd_of_condIndep
    P history code signal hhistory hcode hsignal signal_history_given_code
  have hjointSwap :
      condDistrib (fun ω ↦ (signal ω, code ω)) history P =ᵐ[P.map history]
        (condDistrib (fun ω ↦ (code ω, signal ω)) history P).map
          MeasurableEquiv.prodComm := by
    have hfun :
        MeasurableEquiv.prodComm ∘ (fun ω ↦ (code ω, signal ω)) =
          (fun ω ↦ (signal ω, code ω)) := by
      funext ω
      rfl
    rw [← hfun]
    exact condDistrib_comp history (hcode.prodMk hsignal).aemeasurable
      MeasurableEquiv.prodComm.measurable (μ := P)
  have hfwdMeasure :
      ∀ᵐ u ∂P.map history,
        condDistrib (fun ω ↦ (signal ω, code ω)) history P u =
          condDistrib signal history P u ⊗ₘ condDistrib code signal P := by
    filter_upwards [hfwdKernel] with u hu
    rw [hu, Kernel.compProd_apply_eq_compProd_sectR,
      Kernel.sectR_prodMkLeft]
  have hrevMeasure :
      ∀ᵐ u ∂P.map history,
        condDistrib (fun ω ↦ (signal ω, code ω)) history P u =
          Measure.map MeasurableEquiv.prodComm
            (condDistrib code history P u ⊗ₘ condDistrib signal code P) := by
    filter_upwards [hjointSwap, hrevKernel] with u hswap hrev
    rw [hswap, Kernel.map_apply _ MeasurableEquiv.prodComm.measurable,
      hrev, Kernel.compProd_apply_eq_compProd_sectR,
      Kernel.sectR_prodMkLeft]
  filter_upwards [hfwdMeasure, hrevMeasure] with u hfwd hrev
  exact ⟨hfwd, hrev, hfwd.symm.trans hrev⟩

private lemma condDistrib_history_eq_view_of_condIndep_kernel
    {Ω H V F : Type*}
    [MeasurableSpace Ω] [StandardBorelSpace Ω]
    [MeasurableSpace H] [StandardBorelSpace H] [Nonempty H]
    [MeasurableSpace V] [StandardBorelSpace V] [Nonempty V]
    [MeasurableSpace F] [StandardBorelSpace F] [Nonempty F]
    (P : Measure Ω) [IsProbabilityMeasure P]
    (history : Ω → H) (view : Ω → V) (code : Ω → F)
    (hhistory : Measurable history) (hview : Measurable view)
    (hcode : Measurable code)
    (code_history_given_view : code ⟂ᵢ[view, hview; P] history)
    (code_view_given_history : code ⟂ᵢ[history, hhistory; P] view) :
    ∀ᵐ uv ∂P.map (fun ω ↦ (history ω, view ω)),
      condDistrib code history P uv.1 = condDistrib code view P uv.2 := by
  let HV := fun ω ↦ (history ω, view ω)
  let VH := fun ω ↦ (view ω, history ω)
  let e : H × V ≃ᵐ V × H := MeasurableEquiv.prodComm
  have hHV : Measurable HV := hhistory.prodMk hview
  have hVH : Measurable VH := hview.prodMk hhistory
  have hcondHistory :
      condDistrib code HV P =ᵐ[P.map HV]
        (condDistrib code history P).prodMkRight V :=
    (condIndepFun_iff_condDistrib_prod_ae_eq_prodMkRight
      hcode hview hhistory).mp code_view_given_history.symm
  have hcondView :
      condDistrib code VH P =ᵐ[P.map VH]
        (condDistrib code view P).prodMkRight H :=
    (condIndepFun_iff_condDistrib_prod_ae_eq_prodMkRight
      hcode hhistory hview).mp code_history_given_view.symm
  have hlaw : (P.map HV).map e = P.map VH := by
    rw [Measure.map_map]
    · rfl
    · exact e.measurable
    · exact hHV
  have hcondView' :
      (condDistrib code VH P).comap e e.measurable =ᵐ[P.map HV]
        ((condDistrib code view P).prodMkRight H).comap e e.measurable := by
    rw [← hlaw] at hcondView
    exact eventuallyEq_comap_measurableEquiv e (P.map HV) _ _ hcondView
  have hswap := condDistrib_prod_swap_ae
    P history view code hhistory hview hcode
  filter_upwards [hcondHistory, hcondView', hswap] with uv hu hv hs
  rcases uv with ⟨u, v⟩
  change condDistrib code history P u = condDistrib code view P v
  change condDistrib code HV P (u, v) = condDistrib code history P u at hu
  change condDistrib code VH P (v, u) = condDistrib code view P v at hv
  change condDistrib code HV P (u, v) = condDistrib code VH P (v, u) at hs
  exact hu.symm.trans (hs.trans hv)

private lemma eventually_condDistrib_history_mem_family_of_view
    {Ω H V F M : Type*}
    [MeasurableSpace Ω] [StandardBorelSpace Ω]
    [MeasurableSpace H] [StandardBorelSpace H] [Nonempty H]
    [MeasurableSpace V] [StandardBorelSpace V] [Nonempty V]
    [MeasurableSpace F] [StandardBorelSpace F] [Nonempty F]
    (P : Measure Ω) [IsProbabilityMeasure P]
    (history : Ω → H) (view : Ω → V) (code : Ω → F)
    (hhistory : Measurable history) (hview : Measurable view)
    (hcode : Measurable code)
    (code_history_given_view : code ⟂ᵢ[view, hview; P] history)
    (code_view_given_history : code ⟂ᵢ[history, hhistory; P] view)
    (family : M → Measure F) (learnedParameter : V → M)
    (hlearned : condDistrib code view P =ᵐ[P.map view]
      fun v ↦ family (learnedParameter v)) :
    ∀ᵐ u ∂P.map history, ∃ q, condDistrib code history P u = family q := by
  let HV := fun ω ↦ (history ω, view ω)
  have hHV : Measurable HV := hhistory.prodMk hview
  have hkernel := condDistrib_history_eq_view_of_condIndep_kernel
    P history view code hhistory hview hcode
      code_history_given_view code_view_given_history
  have hdisint : P.map history ⊗ₘ condDistrib view history P = P.map HV :=
    compProd_map_condDistrib hview.aemeasurable
  have hkernelNested :
      ∀ᵐ u ∂P.map history, ∀ᵐ v ∂condDistrib view history P u,
        condDistrib code history P u = condDistrib code view P v := by
    rw [← hdisint] at hkernel
    exact Measure.ae_ae_of_ae_compProd hkernel
  have hviewMarg : Measure.map Prod.snd (P.map HV) = P.map view := by
    rw [Measure.map_map]
    · rfl
    · exact measurable_snd
    · exact hHV
  have hlearnedJoint :
      ∀ᵐ hv ∂P.map HV,
        condDistrib code view P hv.2 = family (learnedParameter hv.2) := by
    have hmapped :
        ∀ᵐ v ∂Measure.map Prod.snd (P.map HV),
          condDistrib code view P v = family (learnedParameter v) := by
      rw [hviewMarg]
      exact hlearned
    exact ae_of_ae_map measurable_snd.aemeasurable hmapped
  have hlearnedNested :
      ∀ᵐ u ∂P.map history, ∀ᵐ v ∂condDistrib view history P u,
        condDistrib code view P v = family (learnedParameter v) := by
    rw [← hdisint] at hlearnedJoint
    exact Measure.ae_ae_of_ae_compProd hlearnedJoint
  filter_upwards [hkernelNested, hlearnedNested] with u hu hu'
  obtain ⟨v, hv, hv'⟩ := (hu.and hu').exists
  exact ⟨learnedParameter v, hv.trans hv'⟩

private noncomputable def likelihoodCoordinatesCLM
    (v : ι → EuclideanSpace ℝ κ) :
    EuclideanSpace ℝ κ →L[ℝ] EuclideanSpace ℝ ι :=
  (PiLp.continuousLinearEquiv 2 ℝ (fun _ : ι ↦ ℝ)).symm.toContinuousLinearMap.comp
    (ContinuousLinearMap.pi fun i ↦ innerSL ℝ (v i))

@[simp] private lemma likelihoodCoordinatesCLM_apply
    (v : ι → EuclideanSpace ℝ κ) (x : EuclideanSpace ℝ κ) (i : ι) :
    likelihoodCoordinatesCLM v x i = inner ℝ (v i) x := by
  rfl

private lemma likelihoodCoordinatesCLM_injective
    (v : ι → EuclideanSpace ℝ ι)
    (hv : Submodule.span ℝ (Set.range v) = ⊤) :
    Function.Injective (likelihoodCoordinatesCLM v) := by
  intro x y hxy
  have hall : ∀ i, inner ℝ (v i) (x - y) = 0 := by
    intro i
    have hi := congrArg (fun z : EuclideanSpace ℝ ι ↦ z i) hxy
    simp only [likelihoodCoordinatesCLM_apply] at hi
    rw [inner_sub_right, sub_eq_zero]
    exact hi
  have hortho : ∀ z ∈ Submodule.span ℝ (Set.range v),
      inner ℝ z (x - y) = 0 := by
    intro z hz
    refine Submodule.span_induction ?_ ?_ ?_ ?_ hz
    · intro w hw
      rcases hw with ⟨i, rfl⟩
      exact hall i
    · simp
    · intro a b _ha_mem _hb_mem ha hb
      rw [inner_add_left, ha, hb, add_zero]
    · intro a z _hz_mem hz
      rw [inner_smul_left, hz, mul_zero]
  have hmem : x - y ∈ Submodule.span ℝ (Set.range v) := by
    rw [hv]
    exact Submodule.mem_top
  exact sub_eq_zero.mp (inner_self_eq_zero.mp (hortho _ hmem))

private noncomputable def likelihoodCoordinates
    (v : ι → EuclideanSpace ℝ ι)
    (hv : Submodule.span ℝ (Set.range v) = ⊤) :
    EuclideanSpace ℝ ι ≃L[ℝ] EuclideanSpace ℝ ι := by
  let T := likelihoodCoordinatesCLM v
  have hinj : Function.Injective T := likelihoodCoordinatesCLM_injective v hv
  have hsurj : Function.Surjective T :=
    (LinearMap.injective_iff_surjective_of_finrank_eq_finrank rfl).mp hinj
  exact ContinuousLinearEquiv.ofBijective T
    (LinearMap.ker_eq_bot.mpr hinj) (LinearMap.range_eq_top.mpr hsurj)

@[simp] private lemma likelihoodCoordinates_apply
    (v : ι → EuclideanSpace ℝ ι)
    (hv : Submodule.span ℝ (Set.range v) = ⊤)
    (x : EuclideanSpace ℝ ι) (i : ι) :
    likelihoodCoordinates v hv x i = inner ℝ (v i) x := by
  rfl

private lemma affineReadout_of_likelihoodSufficiency
    {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω)
    (signal : Ω → EuclideanSpace ℝ ι)
    (code : Ω → EuclideanSpace ℝ κ)
    (signalCoordinates : EuclideanSpace ℝ ι ≃L[ℝ] EuclideanSpace ℝ ι)
    (codeCoordinates : EuclideanSpace ℝ κ →L[ℝ] EuclideanSpace ℝ ι)
    (offset : EuclideanSpace ℝ ι)
    (likelihoodSufficiency :
      (fun ω ↦ signalCoordinates (signal ω)) =ᵐ[P]
        (fun ω ↦ codeCoordinates (code ω) + offset)) :
    ∃ (L : EuclideanSpace ℝ κ →L[ℝ] EuclideanSpace ℝ ι)
        (c : EuclideanSpace ℝ ι),
      signal =ᵐ[P] (fun ω ↦ L (code ω) + c) := by
  let L := signalCoordinates.symm.toContinuousLinearMap.comp codeCoordinates
  let c := signalCoordinates.symm offset
  refine ⟨L, c, ?_⟩
  filter_upwards [likelihoodSufficiency] with ω hω
  apply signalCoordinates.injective
  simp only [L, c, ContinuousLinearMap.comp_apply,
    ContinuousLinearEquiv.apply_symm_apply, map_add]
  simpa using hω

private lemma rnDeriv_eq_of_common_forward_reverse
    {E F : Type*} [MeasurableSpace E] [MeasurableSpace F]
    (P P₀ : Measure E) (Q Q₀ : Measure F)
    (K : Kernel E F) (R : Kernel F E)
    [IsFiniteMeasure P] [IsFiniteMeasure P₀]
    [IsFiniteMeasure Q] [IsFiniteMeasure Q₀]
    [IsFiniteKernel K] [IsFiniteKernel R]
    (hP : P ⊗ₘ K = Measure.map MeasurableEquiv.prodComm (Q ⊗ₘ R))
    (hP₀ : P₀ ⊗ₘ K = Measure.map MeasurableEquiv.prodComm (Q₀ ⊗ₘ R)) :
    (fun x : E × F ↦ P.rnDeriv P₀ x.1) =ᵐ[P₀ ⊗ₘ K]
      (fun x ↦ Q.rnDeriv Q₀ x.2) := by
  have hfwd := rnDeriv_measure_compProd_left P P₀ K
  have hrev₀ := rnDeriv_measure_compProd_left Q Q₀ R
  let e : F × E ≃ᵐ E × F := MeasurableEquiv.prodComm
  have hmap := e.measurableEmbedding.rnDeriv_map (Q ⊗ₘ R) (Q₀ ⊗ₘ R)
  have hrev :
      (Measure.map e (Q ⊗ₘ R)).rnDeriv
          (Measure.map e (Q₀ ⊗ₘ R)) =ᵐ[Measure.map e (Q₀ ⊗ₘ R)]
        (fun x : E × F ↦ Q.rnDeriv Q₀ x.2) := by
    rw [Filter.EventuallyEq, e.measurableEmbedding.ae_map_iff,
      ← Filter.EventuallyEq]
    filter_upwards [hmap, hrev₀] with x hx hy
    exact hx.trans hy
  rw [← hP, ← hP₀] at hrev
  exact hfwd.symm.trans hrev

/-- Rebasing a statistic natural exponential family at a nonzero parameter preserves its members.
The new direction is added to the reference parameter.
The new log partition is the corresponding partition difference. -/
private lemma statisticNaturalExpMeasure_rebase
    {X : Type*} [MeasurableSpace X]
    (base : Measure X)
    (statistic : X → EuclideanSpace ℝ ι)
    (hstatistic : Measurable statistic)
    (partition : EuclideanSpace ℝ ι → ℝ)
    (reference direction : EuclideanSpace ℝ ι) :
    statisticNaturalExpMeasure
        (statisticNaturalExpMeasure base statistic partition reference)
        statistic
        (fun a ↦ partition (reference + a) - partition reference)
        direction =
      statisticNaturalExpMeasure base statistic partition
        (reference + direction) := by
  unfold statisticNaturalExpMeasure
  let f : X → ENNReal := fun x ↦
    ENNReal.ofReal (naturalExpWeight partition reference (statistic x))
  let g : X → ENNReal := fun x ↦ ENNReal.ofReal
    (naturalExpWeight
      (fun a ↦ partition (reference + a) - partition reference)
      direction (statistic x))
  have hf : AEMeasurable f base := by
    simpa only [f, Function.comp_apply] using
      (((measurable_naturalExpWeight partition reference).comp
        hstatistic).ennreal_ofReal.aemeasurable)
  have hg : AEMeasurable g base := by
    simpa only [g, Function.comp_apply] using
      (((measurable_naturalExpWeight
        (fun a ↦ partition (reference + a) - partition reference)
        direction).comp hstatistic).ennreal_ofReal.aemeasurable)
  change (base.withDensity f).withDensity g = _
  rw [← withDensity_mul₀ hf hg]
  congr 1
  funext x
  dsimp only [f, g, Pi.mul_apply]
  simp only [naturalExpWeight]
  rw [← ENNReal.ofReal_mul (Real.exp_nonneg _)]
  congr 1
  rw [← Real.exp_add]
  congr 1
  rw [inner_add_left]
  ring

/-- This absolute-continuity direction transports almost-everywhere identities back to the carrier.
The reverse absolute continuity is automatic for every `withDensity` measure. -/
private lemma absolutelyContinuous_statisticNaturalExpMeasure
    {X : Type*} [MeasurableSpace X]
    (base : Measure X)
    (statistic : X → EuclideanSpace ℝ ι)
    (hstatistic : Measurable statistic)
    (partition : EuclideanSpace ℝ ι → ℝ)
    (parameter : EuclideanSpace ℝ ι) :
    base ≪ statisticNaturalExpMeasure base statistic partition parameter := by
  apply withDensity_absolutelyContinuous'
    (((measurable_naturalExpWeight partition parameter).comp
      hstatistic).ennreal_ofReal.aemeasurable)
  filter_upwards with x
  exact ENNReal.ofReal_ne_zero_iff.mpr (Real.exp_pos _)

/-- The Radon--Nikodym derivative of a statistic-tilted measure with respect to
its base measure is the explicit exponential-family weight. -/
private lemma rnDeriv_statisticNaturalExpMeasure
    {X : Type*} [MeasurableSpace X]
    (μ : Measure X) [SigmaFinite μ]
    (statistic : X → EuclideanSpace ℝ ι) (hstatistic : Measurable statistic)
    (partition : EuclideanSpace ℝ ι → ℝ) (a : EuclideanSpace ℝ ι) :
    (statisticNaturalExpMeasure μ statistic partition a).rnDeriv μ =ᵐ[μ]
      fun x ↦ ENNReal.ofReal (naturalExpWeight partition a (statistic x)) := by
  simpa only [statisticNaturalExpMeasure, Function.comp_apply] using
    Measure.rnDeriv_withDensity μ
      ((measurable_naturalExpWeight partition a).comp hstatistic).ennreal_ofReal

/-- Finite-factorization form of the exponential-family readout principle.

`sourceBase` and `codeBase` are the conditional laws at one reference history.
The selected conditional laws are written as normalized natural-exponential
tilts of those reference laws.  `hjoint` and `hjointBase` state that all of
these laws use the same forward channel `K` and reverse channel `R`.

For a family originally parameterized by `alpha`, use
`sourceDirection i = alpha (u i) - alpha u₀` and the rebased log-partition
`a ↦ psi (alpha u₀ + a) - psi (alpha u₀)`, and analogously on the learned
side.  Thus the reference bases need not correspond to zero parameters in the
original family.

No invertibility of the observation map or of either sufficient statistic is
assumed.  The only diversity assumption is the explicit spanning condition on
the finitely many selected source natural-parameter differences. -/
private theorem exponentialFamilyReadout_of_factorizations
    {X Y : Type*}
    [MeasurableSpace X] [MeasurableSpace Y]
    (sourceBase : Measure X) (codeBase : Measure Y)
    [IsFiniteMeasure sourceBase] [IsFiniteMeasure codeBase]
    (sourceStatistic : X → EuclideanSpace ℝ ι)
    (codeStatistic : Y → EuclideanSpace ℝ κ)
    (hsourceStatistic : Measurable sourceStatistic)
    (hcodeStatistic : Measurable codeStatistic)
    (sourcePartition : EuclideanSpace ℝ ι → ℝ)
    (codePartition : EuclideanSpace ℝ κ → ℝ)
    (sourceDirection : ι → EuclideanSpace ℝ ι)
    (codeDirection : ι → EuclideanSpace ℝ κ)
    (hsourceSpan :
      Submodule.span ℝ (Set.range sourceDirection) = ⊤)
    (hsourceNoAffineRedundancy :
      ∀ (v : EuclideanSpace ℝ ι) (a : ℝ),
        (fun x ↦ inner ℝ v (sourceStatistic x)) =ᵐ[sourceBase]
            (fun _ ↦ a) →
          v = 0)
    (K : Kernel X Y) (R : Kernel Y X)
    [IsMarkovKernel K] [IsMarkovKernel R]
    (hsourceFinite : ∀ i, IsFiniteMeasure
      (statisticNaturalExpMeasure sourceBase sourceStatistic sourcePartition
        (sourceDirection i)))
    (hcodeFinite : ∀ i, IsFiniteMeasure
      (statisticNaturalExpMeasure codeBase codeStatistic codePartition
        (codeDirection i)))
    (hjoint : ∀ i,
      statisticNaturalExpMeasure sourceBase sourceStatistic sourcePartition
          (sourceDirection i) ⊗ₘ K =
        Measure.map MeasurableEquiv.prodComm
          (statisticNaturalExpMeasure codeBase codeStatistic codePartition
            (codeDirection i) ⊗ₘ R))
    (hjointBase :
      sourceBase ⊗ₘ K =
        Measure.map MeasurableEquiv.prodComm (codeBase ⊗ₘ R)) :
    ∃ (L : EuclideanSpace ℝ κ →L[ℝ] EuclideanSpace ℝ ι)
        (c : EuclideanSpace ℝ ι),
      (fun x : X × Y ↦ sourceStatistic x.1) =ᵐ[sourceBase ⊗ₘ K]
          (fun x ↦ L (codeStatistic x.2) + c) ∧
      Function.Surjective L ∧
      Module.finrank ℝ (EuclideanSpace ℝ ι) ≤
        Module.finrank ℝ (EuclideanSpace ℝ κ) := by
  let sourceCoordinates :
      EuclideanSpace ℝ ι ≃L[ℝ] EuclideanSpace ℝ ι :=
    likelihoodCoordinates sourceDirection hsourceSpan
  let codeCoordinates :
      EuclideanSpace ℝ κ →L[ℝ] EuclideanSpace ℝ ι :=
    likelihoodCoordinatesCLM codeDirection
  have hratio (i : ι) :
      (fun x : X × Y ↦
        (statisticNaturalExpMeasure sourceBase sourceStatistic sourcePartition
          (sourceDirection i)).rnDeriv sourceBase x.1) =ᵐ[sourceBase ⊗ₘ K]
      (fun x ↦
        (statisticNaturalExpMeasure codeBase codeStatistic codePartition
          (codeDirection i)).rnDeriv codeBase x.2) := by
    letI := hsourceFinite i
    letI := hcodeFinite i
    exact rnDeriv_eq_of_common_forward_reverse
      (statisticNaturalExpMeasure sourceBase sourceStatistic sourcePartition
        (sourceDirection i)) sourceBase
      (statisticNaturalExpMeasure codeBase codeStatistic codePartition
        (codeDirection i)) codeBase K R (hjoint i) hjointBase
  have hsourceRN (i : ι) := rnDeriv_statisticNaturalExpMeasure
    sourceBase sourceStatistic hsourceStatistic sourcePartition
      (sourceDirection i)
  have hcodeRN (i : ι) := rnDeriv_statisticNaturalExpMeasure
    codeBase codeStatistic hcodeStatistic codePartition (codeDirection i)
  have hsourceJoint (i : ι) :
      (fun x : X × Y ↦
        (statisticNaturalExpMeasure sourceBase sourceStatistic sourcePartition
          (sourceDirection i)).rnDeriv sourceBase x.1) =ᵐ[sourceBase ⊗ₘ K]
      (fun x ↦ ENNReal.ofReal (Real.exp
        ((sourceCoordinates (sourceStatistic x.1)) i -
          sourcePartition (sourceDirection i)))) := by
    have hraw := Measure.ae_eq_compProd_of_ae_eq_fst K
      (Measure.measurable_rnDeriv _ _)
      (((measurable_naturalExpWeight sourcePartition
        (sourceDirection i)).comp hsourceStatistic).ennreal_ofReal)
      (hsourceRN i)
    filter_upwards [hraw] with x hx
    simpa [naturalExpWeight, sourceCoordinates,
      likelihoodCoordinates_apply] using hx
  have hcodeJoint (i : ι) :
      (fun x : X × Y ↦
        (statisticNaturalExpMeasure codeBase codeStatistic codePartition
          (codeDirection i)).rnDeriv codeBase x.2) =ᵐ[sourceBase ⊗ₘ K]
      (fun x ↦ ENNReal.ofReal (Real.exp
        ((codeCoordinates (codeStatistic x.2)) i -
          codePartition (codeDirection i)))) := by
    rw [hjointBase]
    let e : Y × X ≃ᵐ X × Y := MeasurableEquiv.prodComm
    rw [Filter.EventuallyEq, e.measurableEmbedding.ae_map_iff,
      ← Filter.EventuallyEq]
    change (fun x : Y × X ↦
      (statisticNaturalExpMeasure codeBase codeStatistic codePartition
        (codeDirection i)).rnDeriv codeBase x.1) =ᵐ[codeBase ⊗ₘ R]
      (fun x ↦ ENNReal.ofReal (Real.exp
        ((codeCoordinates (codeStatistic x.1)) i -
          codePartition (codeDirection i))))
    have hraw := Measure.ae_eq_compProd_of_ae_eq_fst R
      (Measure.measurable_rnDeriv _ _)
      (((measurable_naturalExpWeight codePartition
        (codeDirection i)).comp hcodeStatistic).ennreal_ofReal)
      (hcodeRN i)
    filter_upwards [hraw] with z hz
    simpa [naturalExpWeight, codeCoordinates,
      likelihoodCoordinatesCLM_apply] using hz
  have hweights (i : ι) :
      (fun x : X × Y ↦ ENNReal.ofReal (Real.exp
        ((sourceCoordinates (sourceStatistic x.1)) i -
          sourcePartition (sourceDirection i)))) =ᵐ[sourceBase ⊗ₘ K]
      (fun x ↦ ENNReal.ofReal (Real.exp
        ((codeCoordinates (codeStatistic x.2)) i -
          codePartition (codeDirection i)))) :=
    (hsourceJoint i).symm.trans ((hratio i).trans (hcodeJoint i))
  let offset : EuclideanSpace ℝ ι := WithLp.toLp 2 fun i ↦
    sourcePartition (sourceDirection i) - codePartition (codeDirection i)
  have hcoordinates :
      (fun x : X × Y ↦ sourceCoordinates (sourceStatistic x.1)) =ᵐ[
        sourceBase ⊗ₘ K]
      (fun x ↦ codeCoordinates (codeStatistic x.2) + offset) := by
    filter_upwards [Filter.eventually_all.2 hweights] with x hx
    ext i
    have hi := hx i
    simp only [ENNReal.ofReal_eq_ofReal_iff, Real.exp_nonneg] at hi
    apply_fun Real.log at hi
    simp only [Real.log_exp] at hi
    change (sourceCoordinates (sourceStatistic x.1)) i =
      (codeCoordinates (codeStatistic x.2)) i +
        (sourcePartition (sourceDirection i) -
          codePartition (codeDirection i))
    linarith
  obtain ⟨L, c, hreadout⟩ := affineReadout_of_likelihoodSufficiency
    (sourceBase ⊗ₘ K) (fun x ↦ sourceStatistic x.1)
      (fun x ↦ codeStatistic x.2) sourceCoordinates codeCoordinates offset
      hcoordinates
  have hadjoint : Function.Injective L.adjoint := by
    intro u v huv
    have hker : L.adjoint (u - v) = 0 := by
      rw [map_sub, huv, sub_self]
    have hinnerJoint :
        (fun x : X × Y ↦ inner ℝ (u - v) (sourceStatistic x.1)) =ᵐ[
          sourceBase ⊗ₘ K]
        (fun _ ↦ inner ℝ (u - v) c) := by
      filter_upwards [hreadout] with x hx
      rw [hx, inner_add_right, ← L.adjoint_inner_left, hker]
      simp
    have hinnerBase :
        (fun x : X ↦ inner ℝ (u - v) (sourceStatistic x)) =ᵐ[sourceBase]
          (fun _ ↦ inner ℝ (u - v) c) := by
      rw [← Measure.fst_compProd sourceBase K]
      apply (ae_map_iff measurable_fst.aemeasurable
        (measurableSet_eq_fun
          (Measurable.const_inner hsourceStatistic) measurable_const)).2
      exact hinnerJoint
    exact sub_eq_zero.mp
      (hsourceNoAffineRedundancy (u - v) (inner ℝ (u - v) c) hinnerBase)
  have hsurjective : Function.Surjective L := by
    change Function.Surjective L.toLinearMap
    rw [← LinearMap.range_eq_top]
    apply Submodule.eq_top_of_finrank_eq
    calc
      Module.finrank ℝ L.range = Module.finrank ℝ L.adjoint.range :=
        L.toLinearMap.finrank_range_adjoint.symm
      _ = Module.finrank ℝ (EuclideanSpace ℝ ι) :=
        LinearMap.finrank_range_of_inj hadjoint
  exact ⟨L, c, hreadout, hsurjective,
    LinearMap.finrank_le_finrank_of_surjective hsurjective⟩

/-- End-to-end exponential-family statistic recovery for nuisance LDM.

The four conditional-independence hypotheses encode the manuscript's Markov and information steps:
`signal ⟂ history | code`, `code ⟂ history | signal`,
`code ⟂ history | view`, and `code ⟂ view | history`.
`hsourceConditional` and `hlearnedConditional` give the two exact exponential-family laws.
`hhistoryDiversity` is essential parameter coverage.
`hsourceNoAffineRedundancy` excludes a source statistic supported on a proper affine hyperplane.
The conclusion holds under `P`, not only under the selected reference-history law.
It includes a measurable decoder for an injective source statistic.
It also includes bijectivity of the readout when the statistic dimensions agree. -/
theorem statisticReadout_of_exponentialFamily_nuisanceLDM
    {Ω H V X Y : Type*}
    [MeasurableSpace Ω] [StandardBorelSpace Ω]
    [MeasurableSpace H] [StandardBorelSpace H] [Nonempty H]
    [MeasurableSpace V] [StandardBorelSpace V] [Nonempty V]
    [MeasurableSpace X] [StandardBorelSpace X] [Nonempty X]
    [MeasurableSpace Y] [StandardBorelSpace Y] [Nonempty Y]
    (P : Measure Ω) [IsProbabilityMeasure P]
    (history : Ω → H) (view : Ω → V)
    (signal : Ω → X) (code : Ω → Y)
    (hhistory : Measurable history) (hview : Measurable view)
    (hsignal : Measurable signal) (hcode : Measurable code)
    (signal_history_given_code : signal ⟂ᵢ[code, hcode; P] history)
    (code_history_given_signal : code ⟂ᵢ[signal, hsignal; P] history)
    (code_history_given_view : code ⟂ᵢ[view, hview; P] history)
    (code_view_given_history : code ⟂ᵢ[history, hhistory; P] view)
    (sourceCarrier : Measure X) (codeCarrier : Measure Y)
    (sourceStatistic : X → EuclideanSpace ℝ ι)
    (codeStatistic : Y → EuclideanSpace ℝ κ)
    (hsourceStatistic : Measurable sourceStatistic)
    (hcodeStatistic : Measurable codeStatistic)
    (sourcePartition : EuclideanSpace ℝ ι → ℝ)
    (codePartition : EuclideanSpace ℝ κ → ℝ)
    (sourceParameter : H → EuclideanSpace ℝ ι)
    (learnedParameter : V → EuclideanSpace ℝ κ)
    (hsourceConditional :
      condDistrib signal history P =ᵐ[P.map history]
        fun u ↦ statisticNaturalExpMeasure sourceCarrier sourceStatistic
          sourcePartition (sourceParameter u))
    (hlearnedConditional :
      condDistrib code view P =ᵐ[P.map view]
        fun v ↦ statisticNaturalExpMeasure codeCarrier codeStatistic
          codePartition (learnedParameter v))
    (hhistoryDiversity :
      ∀ good : Set H, (∀ᵐ u ∂P.map history, u ∈ good) →
        ∃ (u₀ : H) (u : ι → H),
          u₀ ∈ good ∧
          (∀ i, u i ∈ good) ∧
          Submodule.span ℝ
            (Set.range fun i ↦ sourceParameter (u i) - sourceParameter u₀) = ⊤)
    (hsourceNoAffineRedundancy :
      ∀ (w : EuclideanSpace ℝ ι) (a : ℝ),
        (fun x ↦ inner ℝ w (sourceStatistic x)) =ᵐ[sourceCarrier]
            (fun _ ↦ a) →
          w = 0) :
    ∃ (L : EuclideanSpace ℝ κ →L[ℝ] EuclideanSpace ℝ ι)
        (c : EuclideanSpace ℝ ι),
      (fun ω ↦ sourceStatistic (signal ω)) =ᵐ[P]
          (fun ω ↦ L (codeStatistic (code ω)) + c) ∧
      Function.Surjective L ∧
      Module.finrank ℝ (EuclideanSpace ℝ ι) ≤
        Module.finrank ℝ (EuclideanSpace ℝ κ) ∧
      (Function.Injective sourceStatistic →
        ∃ decoder : Y → X, Measurable decoder ∧
          signal =ᵐ[P] fun ω ↦ decoder (code ω)) ∧
      (Module.finrank ℝ (EuclideanSpace ℝ κ) =
          Module.finrank ℝ (EuclideanSpace ℝ ι) →
        Function.Bijective L) := by
  let sourceFamily : EuclideanSpace ℝ ι → Measure X := fun a ↦
    statisticNaturalExpMeasure sourceCarrier sourceStatistic sourcePartition a
  let codeFamily : EuclideanSpace ℝ κ → Measure Y := fun b ↦
    statisticNaturalExpMeasure codeCarrier codeStatistic codePartition b
  let K : Kernel X Y := condDistrib code signal P
  let R : Kernel Y X := condDistrib signal code P
  have hcodeConditionalExists :
      ∀ᵐ u ∂P.map history, ∃ q : EuclideanSpace ℝ κ,
        condDistrib code history P u = codeFamily q := by
    simpa only [codeFamily] using
      (eventually_condDistrib_history_mem_family_of_view
        P history view code hhistory hview hcode
          code_history_given_view code_view_given_history
          (fun q ↦ statisticNaturalExpMeasure codeCarrier codeStatistic
            codePartition q)
          learnedParameter hlearnedConditional)
  have hcommon := conditionalJoint_common_forward_reverse_of_condIndep
    P history signal code hhistory hsignal hcode
      signal_history_given_code code_history_given_signal
  let Good : Set H := {u |
    condDistrib signal history P u = sourceFamily (sourceParameter u) ∧
    ∃ q : EuclideanSpace ℝ κ,
      condDistrib code history P u = codeFamily q ∧
      sourceFamily (sourceParameter u) ⊗ₘ K =
        Measure.map MeasurableEquiv.prodComm (codeFamily q ⊗ₘ R)}
  have hGood : ∀ᵐ u ∂P.map history, u ∈ Good := by
    filter_upwards [hsourceConditional, hcodeConditionalExists, hcommon]
      with u hs hq hc
    obtain ⟨q, hq⟩ := hq
    refine ⟨hs, q, hq, ?_⟩
    simpa only [sourceFamily, codeFamily, K, R, hs, hq] using hc.2.2
  obtain ⟨u₀, u, hu₀, hu, hsourceSpan⟩ :=
    hhistoryDiversity Good hGood
  let q₀ : EuclideanSpace ℝ κ := hu₀.2.choose
  let q : ι → EuclideanSpace ℝ κ := fun i ↦ (hu i).2.choose
  have hsourceReference :
      condDistrib signal history P u₀ = sourceFamily (sourceParameter u₀) :=
    hu₀.1
  have hcodeReference :
      condDistrib code history P u₀ = codeFamily q₀ :=
    hu₀.2.choose_spec.1
  have hjointReference :
      sourceFamily (sourceParameter u₀) ⊗ₘ K =
        Measure.map MeasurableEquiv.prodComm (codeFamily q₀ ⊗ₘ R) :=
    hu₀.2.choose_spec.2
  have hsourceSelected (i : ι) :
      condDistrib signal history P (u i) =
        sourceFamily (sourceParameter (u i)) :=
    (hu i).1
  have hcodeSelected (i : ι) :
      condDistrib code history P (u i) = codeFamily (q i) :=
    (hu i).2.choose_spec.1
  have hjointSelected (i : ι) :
      sourceFamily (sourceParameter (u i)) ⊗ₘ K =
        Measure.map MeasurableEquiv.prodComm (codeFamily (q i) ⊗ₘ R) :=
    (hu i).2.choose_spec.2
  let sourceBase : Measure X := sourceFamily (sourceParameter u₀)
  let codeBase : Measure Y := codeFamily q₀
  let sourceDirection : ι → EuclideanSpace ℝ ι := fun i ↦
    sourceParameter (u i) - sourceParameter u₀
  let codeDirection : ι → EuclideanSpace ℝ κ := fun i ↦
    q i - q₀
  let sourceRebasedPartition : EuclideanSpace ℝ ι → ℝ := fun a ↦
    sourcePartition (sourceParameter u₀ + a) -
      sourcePartition (sourceParameter u₀)
  let codeRebasedPartition : EuclideanSpace ℝ κ → ℝ := fun b ↦
    codePartition (q₀ + b) - codePartition q₀
  have hsourceRebase (i : ι) :
      statisticNaturalExpMeasure sourceBase sourceStatistic
          sourceRebasedPartition (sourceDirection i) =
        sourceFamily (sourceParameter (u i)) := by
    have h := statisticNaturalExpMeasure_rebase sourceCarrier sourceStatistic
      hsourceStatistic sourcePartition (sourceParameter u₀)
        (sourceDirection i)
    have hadd : sourceParameter u₀ + sourceDirection i =
        sourceParameter (u i) := by
      dsimp only [sourceDirection]
      abel
    rw [hadd] at h
    simpa only [sourceBase, sourceFamily, sourceRebasedPartition] using h
  have hcodeRebase (i : ι) :
      statisticNaturalExpMeasure codeBase codeStatistic
          codeRebasedPartition (codeDirection i) =
        codeFamily (q i) := by
    have h := statisticNaturalExpMeasure_rebase codeCarrier codeStatistic
      hcodeStatistic codePartition q₀ (codeDirection i)
    have hadd : q₀ + codeDirection i = q i := by
      dsimp only [codeDirection]
      abel
    rw [hadd] at h
    simpa only [codeBase, codeFamily, codeRebasedPartition] using h
  letI : IsFiniteMeasure sourceBase :=
    ⟨by
      change sourceFamily (sourceParameter u₀) Set.univ < ⊤
      rw [← hsourceReference]
      exact measure_lt_top _ _⟩
  letI : IsFiniteMeasure codeBase :=
    ⟨by
      change codeFamily q₀ Set.univ < ⊤
      rw [← hcodeReference]
      exact measure_lt_top _ _⟩
  have hsourceFinite (i : ι) : IsFiniteMeasure
      (statisticNaturalExpMeasure sourceBase sourceStatistic
        sourceRebasedPartition (sourceDirection i)) :=
    ⟨by rw [hsourceRebase i, ← hsourceSelected i]; exact measure_lt_top _ _⟩
  have hcodeFinite (i : ι) : IsFiniteMeasure
      (statisticNaturalExpMeasure codeBase codeStatistic
        codeRebasedPartition (codeDirection i)) :=
    ⟨by rw [hcodeRebase i, ← hcodeSelected i]; exact measure_lt_top _ _⟩
  have hsourceNoAffineRedundancyBase :
      ∀ (w : EuclideanSpace ℝ ι) (a : ℝ),
        (fun x ↦ inner ℝ w (sourceStatistic x)) =ᵐ[sourceBase]
            (fun _ ↦ a) →
          w = 0 := by
    intro w a hwa
    apply hsourceNoAffineRedundancy w a
    apply (absolutelyContinuous_statisticNaturalExpMeasure sourceCarrier
      sourceStatistic hsourceStatistic sourcePartition
        (sourceParameter u₀)).ae_eq
    simpa only [sourceBase, sourceFamily] using hwa
  obtain ⟨L, c, hreadoutReference, hL, hfinrank⟩ :=
    exponentialFamilyReadout_of_factorizations
      sourceBase codeBase sourceStatistic codeStatistic hsourceStatistic
      hcodeStatistic sourceRebasedPartition codeRebasedPartition
      sourceDirection codeDirection hsourceSpan
      hsourceNoAffineRedundancyBase K R hsourceFinite hcodeFinite
      (by
        intro i
        rw [hsourceRebase i, hcodeRebase i]
        exact hjointSelected i)
      (by
        simpa only [sourceBase, codeBase] using hjointReference)
  have hcarrierToReference : sourceCarrier ≪ sourceBase := by
    simpa only [sourceBase, sourceFamily] using
      (absolutelyContinuous_statisticNaturalExpMeasure sourceCarrier
        sourceStatistic hsourceStatistic sourcePartition
          (sourceParameter u₀))
  have hfamilyToReference (a : EuclideanSpace ℝ ι) :
      sourceFamily a ≪ sourceBase := by
    exact (withDensity_absolutelyContinuous sourceCarrier _).trans
      hcarrierToReference
  have hconditionalReadout :
      ∀ᵐ a ∂P.map history,
        (fun x : X × Y ↦ sourceStatistic x.1) =ᵐ[
          condDistrib (fun ω ↦ (signal ω, code ω)) history P a]
        (fun x ↦ L (codeStatistic x.2) + c) := by
    filter_upwards [hcommon, hsourceConditional] with a ha hs
    rw [ha.1, hs]
    apply ((hfamilyToReference (sourceParameter a)).compProd_left K).ae_eq
    simpa only [sourceBase] using hreadoutReference
  have htriple :
      ∀ᵐ x ∂(P.map history ⊗ₘ
          condDistrib (fun ω ↦ (signal ω, code ω)) history P),
        sourceStatistic x.2.1 = L (codeStatistic x.2.2) + c := by
    apply Measure.ae_compProd_of_ae_ae
      (measurableSet_eq_fun
        (hsourceStatistic.comp measurable_snd.fst)
        (((L.measurable.comp (hcodeStatistic.comp measurable_snd.snd))).add_const c))
    simpa only [Filter.EventuallyEq, Function.comp_apply] using
      hconditionalReadout
  rw [compProd_map_condDistrib (hsignal.prodMk hcode).aemeasurable] at htriple
  have hpull := ae_of_ae_map
    (hhistory.prodMk (hsignal.prodMk hcode)).aemeasurable htriple
  have hreadout :
      (fun ω ↦ sourceStatistic (signal ω)) =ᵐ[P]
        (fun ω ↦ L (codeStatistic (code ω)) + c) := by
    simpa only [Filter.EventuallyEq] using hpull
  refine ⟨L, c, hreadout, hL, hfinrank, ?_, ?_⟩
  · intro hsourceInjective
    let sourceEmbedding : MeasurableEmbedding sourceStatistic :=
      hsourceStatistic.measurableEmbedding hsourceInjective
    let decoder : Y → X := fun y ↦
      sourceEmbedding.invFun (L (codeStatistic y) + c)
    refine ⟨decoder, ?_, ?_⟩
    · exact sourceEmbedding.measurable_invFun.comp
        ((L.measurable.comp hcodeStatistic).add_const c)
    · filter_upwards [hreadout] with ω hω
      dsimp only [decoder]
      rw [← hω]
      exact (sourceEmbedding.leftInverse_invFun (signal ω)).symm
  · intro hdim
    exact ⟨
      (LinearMap.injective_iff_surjective_of_finrank_eq_finrank hdim).mpr hL,
      hL⟩

end Ldm
