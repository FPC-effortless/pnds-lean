import PNDS.SourceAlignedDifferentiation

/-! Reification of the complete default neural graph into the scalar AD
language. Interpretation lemmas are kernel proofs of graph correspondence;
this compiler does not read or trust a Torch trace. -/
namespace PNDS.SourceAlignedPrograms
noncomputable section
open PNDS.SourceAlignedModel PNDS.SourceAlignedMemory
open PNDS.SourceAlignedDifferentiation

variable {n : ℕ}
abbrev EV (n k : ℕ) := Fin k → Expr n
abbrev EM (n m k : ℕ) := Fin m → Fin k → Expr n

instance : Add (Expr n) := ⟨Expr.add⟩
instance : Neg (Expr n) := ⟨Expr.neg⟩
instance : Sub (Expr n) := ⟨fun a b => .add a (.neg b)⟩
instance : Mul (Expr n) := ⟨Expr.mul⟩
instance : Div (Expr n) := ⟨fun a b => .mul a (.inv b)⟩
instance (k : ℕ) : OfNat (Expr n) k := ⟨.constant k⟩

@[simp] theorem eval_add (a b : Expr n) (rho) : eval (a+b) rho = eval a rho+eval b rho := rfl
@[simp] theorem eval_mul (a b : Expr n) (rho) : eval (a*b) rho = eval a rho*eval b rho := rfl
@[simp] theorem eval_sub (a b : Expr n) (rho) : eval (a-b) rho = eval a rho-eval b rho := rfl
@[simp] theorem eval_div (a b : Expr n) (rho) : eval (a/b) rho = eval a rho/eval b rho := rfl
@[simp] theorem eval_neg (a : Expr n) (rho) : eval (-a) rho = -eval a rho := rfl
@[simp] theorem eval_nat (k : ℕ) (rho : Fin n → ℝ) : eval (OfNat.ofNat k : Expr n) rho = k := rfl
@[simp] theorem eval_zero (rho : Fin n → ℝ) : eval (0 : Expr n) rho = 0 := by norm_num [eval]
@[simp] theorem eval_one (rho : Fin n → ℝ) : eval (1 : Expr n) rho = 1 := by norm_num [eval]
@[simp] theorem eval_two (rho : Fin n → ℝ) : eval (2 : Expr n) rho = 2 := by norm_num [eval]
@[simp] theorem eval_four (rho : Fin n → ℝ) : eval (4 : Expr n) rho = 4 := by norm_num [eval]
@[simp] theorem eval_constant (c : ℝ) (rho : Fin n → ℝ) : eval (.constant c) rho = c := rfl
@[simp] theorem eval_exp (a : Expr n) (rho) : eval (.exp a) rho = Real.exp (eval a rho) := rfl
@[simp] theorem eval_log (a : Expr n) (rho) : eval (.log a) rho = Real.log (eval a rho) := rfl
@[simp] theorem eval_sqrt (a : Expr n) (rho) : eval (.sqrt a) rho = Real.sqrt (eval a rho) := rfl
@[simp] theorem eval_max (a b : Expr n) (rho) : eval (.max a b) rho = max (eval a rho) (eval b rho) := rfl
@[simp] theorem eval_sin (a : Expr n) (rho) : eval (.sin a) rho = Real.sin (eval a rho) := rfl
@[simp] theorem eval_cos (a : Expr n) (rho) : eval (.cos a) rho = Real.cos (eval a rho) := rfl

def iv {k : ℕ} (v : EV n k) (rho : Fin n → ℝ) : Vec k := fun i => eval (v i) rho
def im {m k : ℕ} (v : EM n m k) (rho : Fin n → ℝ) : Mat m k := fun i j => eval (v i j) rho

def sumE : {k : ℕ} → EV n k → Expr n
  | 0, _ => 0
  | k+1, v => v 0 + sumE (fun i => v i.succ)

@[simp] theorem eval_sumE {k : ℕ} (v : EV n k) (rho : Fin n → ℝ) :
    eval (sumE v) rho = ∑ i, eval (v i) rho := by
  induction k with
  | zero => simp [sumE, eval]
  | succ k ih => simp [sumE, ih, Fin.sum_univ_succ]

structure SDense (n i o : ℕ) where
  weight : EM n o i
  bias : EV n o

def interpDense {i o : ℕ} (w : SDense n i o) (rho : Fin n → ℝ) : Dense i o :=
  ⟨im w.weight rho, iv w.bias rho⟩

def lin {i o : ℕ} (w : EM n o i) (x : EV n i) : EV n o := fun j =>
  sumE (fun k => w j k*x k)
def den {i o : ℕ} (w : SDense n i o) (x : EV n i) : EV n o :=
  fun j => lin w.weight x j+w.bias j
def sc {m k : ℕ} (x : EV n m) (y : EV n k) : EV n (m+k) := Fin.append x y
def smap {k : ℕ} (f : Expr n → Expr n) (x : EV n k) : EV n k := fun i => f (x i)
def sf {k : ℕ} (x : EV n (k+k)) : EV n k := fun i => x (Fin.castAdd k i)
def ss {k : ℕ} (x : EV n (k+k)) : EV n k := fun i => x (Fin.natAdd k i)
def mixE {k : ℕ} (x y g : EV n k) : EV n k := fun i => x i+g i*(y i-x i)
def sigE (x : Expr n) : Expr n := 1/(1+Expr.exp (-x))
def tanhE (x : Expr n) : Expr n := (Expr.exp (2*x)-1)/(Expr.exp (2*x)+1)
def siluE (x : Expr n) : Expr n := x*sigE x

def softE {k : ℕ} (x : EV n k) : EV n k :=
  fun i => Expr.exp (x i)/sumE (fun j => Expr.exp (x j))
def normE {k : ℕ} (x : EV n k) : EV n k :=
  fun i => x i / Expr.sqrt (Expr.max (sumE (fun j => x j*x j))
    (.constant ((1/1000000 : ℝ)^2)))

@[simp] theorem sqrt_clamped_square (s : ℝ) :
    Real.sqrt (max s ((1/1000000 : ℝ)^2)) = max (Real.sqrt s) (1/1000000) := by
  rw [(show Monotone Real.sqrt from fun _ _ h => Real.sqrt_le_sqrt h).map_max,
    Real.sqrt_sq (by norm_num : (0:ℝ) ≤ 1/1000000)]

@[simp] theorem interp_lin {i o : ℕ} (w : EM n o i) (x : EV n i) (rho) :
    iv (lin w x) rho = linear (im w rho) (iv x rho) := by
  funext j
  simp [iv, lin, linear, im]

@[simp] theorem interp_den {i o : ℕ} (w : SDense n i o) (x : EV n i) (rho) :
    iv (den w x) rho = dense (interpDense w rho) (iv x rho) := by
  funext j
  simp [iv, den, dense, interpDense, lin, linear, im]

@[simp] theorem interp_cat {i o : ℕ} (x : EV n i) (y : EV n o) (rho) :
    iv (sc x y) rho = cat (iv x rho) (iv y rho) := by
  funext j
  refine Fin.addCases ?_ ?_ j <;> intro k <;> simp [iv, sc, cat, Fin.append]

@[simp] theorem interp_norm {k : ℕ} (x : EV n k) (rho) :
    iv (normE x) rho = normalize (iv x rho) := by
  funext i
  simp only [iv,normE,PNDS.SourceAlignedModel.normalize,eval_div,eval_sqrt,eval_max,
    eval_sumE,eval_mul,eval_constant,sqrt_clamped_square]

structure SParameters (n : ℕ) where
  encoder : SDense n 28 48
  timeProjection : EM n 48 7
  address : SDense n 48 8
  content : SDense n 48 8
  fastMap : SDense n 144 176
  slowMap : SDense n 184 176
  memoryGates : SDense n 224 2
  readoutIn : SDense n 232 132
  readoutOut : SDense n 132 14
  residualIn : SDense n 232 12
  residualOut : SDense n 12 14
  residualLogit : Expr n
  rotations : Fin 2 → EM n 8 8
  sourceContent : SDense n 48 16
  writeRouter : SDense n 48 2
  readRouter : SDense n 8 2
  retention : SDense n 48 2
  writeStrength : SDense n 48 2
  historicalContent : SDense n 16 14
  predictiveMemory : EM n 14 16
  predictiveLogit : Expr n
  deltaScale : EV n 14

def interpParameters (s : SParameters n) (rho : Fin n → ℝ) : Parameters :=
  {encoder := interpDense s.encoder rho
   timeProjection := im s.timeProjection rho
   address := interpDense s.address rho
   content := interpDense s.content rho
   fastMap := interpDense s.fastMap rho
   slowMap := interpDense s.slowMap rho
   memoryGates := interpDense s.memoryGates rho
   readoutIn := interpDense s.readoutIn rho
   readoutOut := interpDense s.readoutOut rho
   residualIn := interpDense s.residualIn rho
   residualOut := interpDense s.residualOut rho
   residualLogit := eval s.residualLogit rho
   rotations := fun h => im (s.rotations h) rho
   sourceContent := interpDense s.sourceContent rho
   writeRouter := interpDense s.writeRouter rho
   readRouter := interpDense s.readRouter rho
   retention := interpDense s.retention rho
   writeStrength := interpDense s.writeStrength rho
   historicalContent := interpDense s.historicalContent rho
   predictiveMemory := im s.predictiveMemory rho
   predictiveLogit := eval s.predictiveLogit rho
   deltaScale := iv s.deltaScale rho}

structure SState (n : ℕ) where
  fastSum : EV n 88
  memory : EM n 8 8
  elapsedClock : Expr n
  lastWriteClock : Expr n
  writeSeen : Bool
  count : ℕ
  aligned : Fin 2 → EM n 16 8
  owner : ℕ
  lastOrdinal : Option ℕ

def interpState (s : SState n) (rho : Fin n → ℝ) : State :=
  {fastSum := iv s.fastSum rho
   memory := im s.memory rho
   elapsedClock := eval s.elapsedClock rho
   lastWriteClock := eval s.lastWriteClock rho
   writeSeen := s.writeSeen
   count := s.count
   aligned := fun h => im (s.aligned h) rho
   owner := s.owner
   lastOrdinal := s.lastOrdinal}

structure SInput (n : ℕ) where
  observation : EV n 14
  observedAction : EV n 14
  elapsedDelta : Expr n
  admitted : Bool
  owner : ℕ
  ordinal : ℕ
  query : Option ℕ

def interpInput (s : SInput n) (rho : Fin n → ℝ) : Input :=
  {observation := iv s.observation rho
   observedAction := iv s.observedAction rho
   elapsedDelta := eval s.elapsedDelta rho
   admitted := s.admitted
   owner := s.owner
   ordinal := s.ordinal
   query := s.query}

structure SPrepared (n : ℕ) where
  embedding : EV n 48
  basis : EV n 88
  proposalFast : EV n 88
  proposalSlow : EV n 88
  nativeKey : EV n 8
  nativeValue : EV n 8
  nativeAlpha : Expr n
  nativeBeta : Expr n
  sourceKey : Fin 2 → EV n 8
  sourceValue : EV n 16
  sourceAlpha : EV n 2
  sourceBeta : EV n 2

def interpPrepared (s : SPrepared n) (rho : Fin n → ℝ) : Prepared :=
  {embedding := iv s.embedding rho
   basis := iv s.basis rho
   proposalFast := iv s.proposalFast rho
   proposalSlow := iv s.proposalSlow rho
   nativeKey := iv s.nativeKey rho
   nativeValue := iv s.nativeValue rho
   nativeAlpha := eval s.nativeAlpha rho
   nativeBeta := eval s.nativeBeta rho
   sourceKey := fun h => iv (s.sourceKey h) rho
   sourceValue := iv s.sourceValue rho
   sourceAlpha := iv s.sourceAlpha rho
   sourceBeta := iv s.sourceBeta rho}

@[simp] theorem interp_first {k : ℕ} (x : EV n (k+k)) (rho) :
    iv (sf x) rho = first (iv x rho) := rfl
@[simp] theorem interp_second {k : ℕ} (x : EV n (k+k)) (rho) :
    iv (ss x) rho = second (iv x rho) := rfl
@[simp] theorem interp_vadd {k : ℕ} (x y : EV n k) (rho) :
    iv (x+y) rho = iv x rho+iv y rho := rfl
@[simp] theorem interp_zero {k : ℕ} (rho : Fin n → ℝ) :
    iv (0 : EV n k) rho = 0 := by
  funext i
  simp [iv,eval]
@[simp] theorem interp_mix {k : ℕ} (x y g : EV n k) (rho) :
    iv (mixE x y g) rho = mix (iv x rho) (iv y rho) (iv g rho) := rfl
@[simp] theorem interp_sigmap {k : ℕ} (x : EV n k) (rho) :
    iv (smap sigE x) rho = map sig (iv x rho) := by
  funext i
  simp [iv, smap, sigE, sig, PNDS.SourceAlignedModel.map, eval, div_eq_mul_inv,sub_eq_add_neg]
@[simp] theorem interp_tanhmap {k : ℕ} (x : EV n k) (rho) :
    iv (smap tanhE x) rho = map th (iv x rho) := by
  funext i
  simp [iv, smap, tanhE, th, PNDS.SourceAlignedModel.map, eval, div_eq_mul_inv,sub_eq_add_neg]
@[simp] theorem interp_silumap {k : ℕ} (x : EV n k) (rho) :
    iv (smap siluE x) rho = map silu (iv x rho) := by
  funext i
  simp [iv, smap, siluE, silu, sigE, sig, PNDS.SourceAlignedModel.map, eval, div_eq_mul_inv,sub_eq_add_neg]
@[simp] theorem interp_quartermap {k : ℕ} (x : EV n k) (rho) :
    iv (smap (fun a => sigE a/4) x) rho = map (fun a => sig a/4) (iv x rho) := by
  funext i
  simp [iv, smap, sigE, sig, PNDS.SourceAlignedModel.map, eval, div_eq_mul_inv,sub_eq_add_neg]
@[simp] theorem interp_soft {k : ℕ} (x : EV n k) (rho) :
    iv (softE x) rho = softmax (iv x rho) := by
  funext i
  simp [iv, softE, softmax, eval, div_eq_mul_inv]

@[simp] theorem interp_den_at {i o : ℕ} (w : SDense n i o) (x : EV n i) (rho) (j : Fin o) :
    eval (den w x j) rho = dense (interpDense w rho) (iv x rho) j :=
  congrFun (interp_den w x rho) j
@[simp] theorem interp_lin_at {i o : ℕ} (w : EM n o i) (x : EV n i) (rho) (j : Fin o) :
    eval (lin w x j) rho = linear (im w rho) (iv x rho) j :=
  congrFun (interp_lin w x rho) j
@[simp] theorem interp_sigmap_at {k : ℕ} (x : EV n k) (rho) (j : Fin k) :
    eval (smap sigE x j) rho = map sig (iv x rho) j :=
  congrFun (interp_sigmap x rho) j
@[simp] theorem interp_tanhmap_at {k : ℕ} (x : EV n k) (rho) (j : Fin k) :
    eval (smap tanhE x j) rho = map th (iv x rho) j :=
  congrFun (interp_tanhmap x rho) j
@[simp] theorem interp_norm_at {k : ℕ} (x : EV n k) (rho) (j : Fin k) :
    eval (normE x j) rho = normalize (iv x rho) j :=
  congrFun (interp_norm x rho) j
@[simp] theorem interp_soft_at {k : ℕ} (x : EV n k) (rho) (j : Fin k) :
    eval (softE x j) rho = softmax (iv x rho) j :=
  congrFun (interp_soft x rho) j
@[simp] theorem interp_known_div {k : ℕ} (x : EV n k) (c : ℝ) (rho) :
    iv (fun j => x j / .constant c) rho = (fun j => iv x rho j / c) := rfl
@[simp] theorem interp_constants {k : ℕ} (x : Vec k) (rho : Fin n → ℝ) :
    iv (fun j => .constant (x j)) rho = x := rfl
@[simp] theorem interp_sig (x : Expr n) (rho) : eval (sigE x) rho = sig (eval x rho) := by
  simp only [sigE,sig,eval_div,eval_add,eval_exp,eval_neg,eval_one]
@[simp] theorem interp_matrix_read (m : EM n 8 8) (key : EV n 8) (rho) :
    iv (fun v => sumE (fun j => m v j*key j)) rho =
      (fun v => rowRead (im m rho v) (iv key rho)) := by
  funext v
  simp only [iv,im,eval_sumE,eval_mul,rowRead]

def timeE (s : SState n) (x : SInput n) : EV n 7 :=
  let now := s.elapsedClock+x.elapsedDelta
  ![Expr.log (1+(.constant s.count)), Expr.sin (.constant ((s.count : ℝ)/16)),
    Expr.cos (.constant ((s.count : ℝ)/16)), Expr.log (1+x.elapsedDelta),
    Expr.log (1+now), Expr.log (1+Expr.max (now-s.lastWriteClock) 0),
    if s.writeSeen then 1 else 0]

def keysE (w : SParameters n) (ordinal : ℕ) : Fin 2 → EV n 8 := fun h =>
  normE (fun j => sumE (fun i => .constant (ordinalCode ordinal i)*w.rotations h i j))

def readE (w : SParameters n) (m : Fin 2 → EM n 16 8) (ordinal : ℕ) : EV n 16 :=
  let k := keysE w ordinal
  let route := softE (den w.readRouter (fun i => .constant (ordinalCode ordinal i)))
  fun v => sumE (fun h => route h*sumE (fun j => m h v j*k h j))

def prepareE (w : SParameters n) (s : SState n) (x : SInput n) : SPrepared n := Id.run do
  let e := smap tanhE (den w.encoder (sc x.observation x.observedAction) +
                       lin w.timeProjection (timeE s x))
  let ctx : EV n 88 := fun j => s.fastSum j / .constant (max (s.count : ℝ) 1)
  let basis := smap tanhE (sf (den w.fastMap (sc (sc e (0 : EV n 88)) (0 : EV n 8))))
  let fg := den w.fastMap (sc (sc e ctx) (0 : EV n 8))
  let pf := mixE ctx (smap tanhE (sf fg)) (smap sigE (ss fg))
  let sg := den w.slowMap (sc (sc pf ctx) (0 : EV n 8))
  let ps := mixE ctx (smap tanhE (ctx+smap tanhE (sf sg))) (smap (fun a => sigE a/4) (ss sg))
  let gates := smap sigE (den w.memoryGates (sc (sc e pf) ps))
  let a := smap sigE (den w.retention e)
  let b := smap sigE (den w.writeStrength e)
  let route := softE (den w.writeRouter e)
  return ⟨e,basis,pf,ps,normE (den w.address e),normE (den w.content e),
    gates 0*(1-gates 1/4),gates 1,keysE w x.ordinal,
    smap tanhE (den w.sourceContent e),a,fun h => b h*route h⟩

def decodeE (w : SParameters n) (s : SState n) (x : SInput n) (p : SPrepared n) : EV n 14 := Id.run do
  let r := fun v => sumE (fun j => s.memory v j*p.nativeKey j)
  let fg := den w.fastMap (sc (sc p.embedding p.proposalFast) r)
  let f := mixE p.proposalFast (smap tanhE (sf fg)) (smap sigE (ss fg))
  let sg := den w.slowMap (sc (sc f p.proposalSlow) r)
  let slow := mixE p.proposalSlow (smap tanhE (p.proposalSlow+smap tanhE (sf sg)))
               (smap (fun a => sigE a/4) (ss sg))
  let z := sc (sc (sc f slow) r) p.embedding
  let d := smap tanhE (den w.readoutOut (smap siluE (den w.readoutIn z)))
  let post := smap tanhE (den w.residualOut (smap siluE (den w.residualIn z)))
  let old := match x.query with
    | some q => readE w s.aligned q
    | none => if x.ordinal=0 then 0 else readE w s.aligned (x.ordinal-1)
  let extra := smap tanhE (lin w.predictiveMemory old)
  return fun j => x.observation j+w.deltaScale j*d j+
    w.deltaScale j*sigE w.residualLogit*post j+
    w.deltaScale j*sigE w.predictiveLogit*extra j

def rowE (row key : EV n 8) (value alpha beta : Expr n) (admitted : Bool) : EV n 8 :=
  if admitted then fun j => alpha*row j+beta*(value-alpha*sumE (fun i => row i*key i))*key j
  else row

def advanceE (s : SState n) (x : SInput n) (p : SPrepared n) : SState n :=
  {fastSum := s.fastSum+p.basis
   memory := fun v => rowE (s.memory v) p.nativeKey (p.nativeValue v)
                       p.nativeAlpha p.nativeBeta x.admitted
   elapsedClock := s.elapsedClock+x.elapsedDelta
   lastWriteClock := if x.admitted then s.elapsedClock+x.elapsedDelta else s.lastWriteClock
   writeSeen := s.writeSeen || x.admitted
   count := s.count+1
   aligned := fun h v => rowE (s.aligned h v) (p.sourceKey h) (p.sourceValue v)
                         (p.sourceAlpha h) (p.sourceBeta h) x.admitted
   owner := s.owner
   lastOrdinal := some x.ordinal}

def stepE (w : SParameters n) (s : SState n) (x : SInput n) : EV n 14 × SState n :=
  let p := prepareE w s x
  (decodeE w s x p,advanceE s x p)

@[simp] theorem interp_time (s : SState n) (x : SInput n) (rho) :
    iv (timeE s x) rho = temporalFeatures (interpState s rho) (interpInput x rho) := by
  funext i
  fin_cases i <;> cases hs : s.writeSeen <;>
    norm_num [iv,timeE,temporalFeatures,interpState,interpInput,eval,hs,sub_eq_add_neg]

@[simp] theorem interp_keys (w : SParameters n) (ordinal : ℕ) (rho) :
    (fun h => iv (keysE w ordinal h) rho) = keys (interpParameters w rho) ordinal := by
  funext h
  change iv (normE (fun j => sumE (fun i => .constant (ordinalCode ordinal i)*w.rotations h i j))) rho =
    PNDS.SourceAlignedModel.normalize (fun j => ∑ i, ordinalCode ordinal i*eval (w.rotations h i j) rho)
  rw [interp_norm]
  apply congrArg PNDS.SourceAlignedModel.normalize
  funext j
  simp only [iv,eval_sumE,eval_mul,eval_constant]

@[simp] theorem interp_read (w : SParameters n) (m : Fin 2 → EM n 16 8) (ordinal : ℕ) (rho) :
    iv (readE w m ordinal) rho = memoryRead (interpParameters w rho)
      (fun h => im (m h) rho) ordinal := by
  have hk (h : Fin 2) (j : Fin 8) : eval (keysE w ordinal h j) rho =
      keys (interpParameters w rho) ordinal h j :=
    congrFun (congrFun (interp_keys w ordinal rho) h) j
  have hr (h : Fin 2) :
      eval (softE (den w.readRouter (fun i => .constant (ordinalCode ordinal i))) h) rho =
        softmax (dense (interpParameters w rho).readRouter (ordinalCode ordinal)) h := by
    rw [interp_soft_at,interp_den,interp_constants]
    rfl
  funext v
  simp only [iv,readE,memoryRead,eval_sumE,eval_mul,hk,hr,rowRead,im]

@[simp] theorem interp_row (row key : EV n 8) (value alpha beta : Expr n) (admitted : Bool) (rho) :
    iv (rowE row key value alpha beta admitted) rho =
      writeRow (iv row rho) (iv key rho) (eval value rho) (eval alpha rho) (eval beta rho) admitted := by
  funext j
  cases admitted <;> simp only [rowE,writeRow,iv,rowRead,Bool.true_eq,Bool.false_eq_true,↓reduceIte,
    eval_add,eval_mul,eval_sub,eval_sumE]

set_option maxRecDepth 4096 in
set_option maxHeartbeats 600000 in
theorem interp_prepare (w : SParameters n) (s : SState n) (x : SInput n) (rho) :
    interpPrepared (prepareE w s x) rho =
      prepare (interpParameters w rho) (interpState s rho) (interpInput x rho) := by
  simp only [prepareE,prepare,Id.run,pure,interpPrepared,interp_den,interp_lin,interp_cat,
    interp_tanhmap,interp_sigmap,interp_quartermap,interp_first,interp_second,
    interp_vadd,interp_zero,interp_mix,interp_norm,interp_soft,interp_known_div,
    interp_time,interp_keys,eval_mul,eval_add,eval_sub,eval_div,eval_nat,
    eval_zero,eval_one,eval_two,eval_four,
    interp_sigmap_at,interp_soft_at,interpParameters,interpState,interpInput]
  congr 1 <;> try rfl

set_option maxRecDepth 4096 in
set_option maxHeartbeats 600000 in
theorem interp_decode (w : SParameters n) (s : SState n) (x : SInput n) (p : SPrepared n) (rho) :
    iv (decodeE w s x p) rho = decode (interpParameters w rho) (interpState s rho)
      (interpInput x rho) (interpPrepared p rho) := by
  cases hq : x.query <;> by_cases ho : x.ordinal=0 <;>
    simp only [decodeE,decode,Id.run,pure,interpPrepared,interpParameters,interpState,interpInput,
      hq,ho,↓reduceIte,interp_den,interp_lin,interp_cat,interp_tanhmap,interp_silumap,
      interp_first,interp_second,interp_sigmap,interp_quartermap,interp_mix,
      interp_vadd,interp_matrix_read,interp_read,interp_zero]
  all_goals
    funext j
    simp only [iv,eval_add,eval_mul,interp_sig,interp_tanhmap_at,interp_lin,interp_den,
      interp_cat,interp_silumap,interp_mix,interp_first,interp_second,interp_sigmap,
      interp_quartermap,interp_vadd,interp_matrix_read,interp_read,interp_zero]
    try rfl

theorem interp_advance (s : SState n) (x : SInput n) (p : SPrepared n) (rho) :
    interpState (advanceE s x p) rho =
      advance (interpState s rho) (interpInput x rho) (interpPrepared p rho) := by
  cases ha : x.admitted <;>
    simp only [advanceE,advance,interpState,interpInput,interpPrepared,iv,im,rowE,
      writeRow,rowRead,PNDS.SourceAlignedVerification.writeMatrix,ha,Bool.false_eq_true,↓reduceIte,
      eval_add,eval_mul,eval_sub,eval_sumE]
  all_goals rfl

theorem complete_graph_compiler_correct (w : SParameters n) (s : SState n)
    (x : SInput n) (rho) :
    (iv (stepE w s x).1 rho,interpState (stepE w s x).2 rho) =
      step (interpParameters w rho) (interpState s rho) (interpInput x rho) := by
  simp only [stepE,step,interp_decode,interp_advance,interp_prepare]

/-- Every output coordinate of the complete compiled neural graph has the
AD derivative, for arbitrary simultaneous parameter/input/state curves.
The regularity predicate records inverses, logs, sqrt and clamp margins. -/
theorem complete_graph_ad_correct (w : SParameters n) (s : SState n) (x : SInput n)
    (curve : Fin n → ℝ → ℝ) (dx : Fin n → ℝ) (t : ℝ) (j : Fin 14)
    (hc : ∀ i, HasDerivAt (curve i) (dx i) t)
    (hr : Regular ((stepE w s x).1 j) (fun i => curve i t)) :
    HasDerivAt (fun z => (step (interpParameters w (fun i => curve i z))
      (interpState s (fun i => curve i z)) (interpInput x (fun i => curve i z))).1 j)
      (tangent ((stepE w s x).1 j) (fun i => curve i t) dx) t := by
  have h := forward_ad_correct ((stepE w s x).1 j) curve dx t hc hr
  have heq : (fun z => eval ((stepE w s x).1 j) (fun i => curve i z)) =
      (fun z => (step (interpParameters w (fun i => curve i z))
        (interpState s (fun i => curve i z)) (interpInput x (fun i => curve i z))).1 j) := by
    funext z
    exact congrFun (congrArg Prod.fst
      (complete_graph_compiler_correct w s x (fun i => curve i z))) j
  rw [heq] at h
  exact h

def runE (w : SParameters n) (s : SState n) : List (SInput n) → SState n
  | [] => s
  | x::xs => runE w (stepE w s x).2 xs

theorem all_history_compiler_correct (w : SParameters n) (s : SState n)
    (past : List (SInput n)) (rho) :
    interpState (runE w s past) rho =
      runState (interpParameters w rho) (interpState s rho)
        (past.map (fun x => interpInput x rho)) := by
  induction past generalizing s with
  | nil => rfl
  | cons x xs ih =>
    simp only [runE,List.map_cons,runState,List.foldl_cons]
    rw [← runState,ih]
    congr 1
    exact congrArg Prod.snd (complete_graph_compiler_correct w s x rho)

/-- AD correctness includes every parameter reuse and every earlier source
event, not only a one-step memory kernel or a fixed learned embedding. -/
theorem all_history_ad_correct (w : SParameters n) (s : SState n)
    (past : List (SInput n)) (current : SInput n)
    (curve : Fin n → ℝ → ℝ) (dx : Fin n → ℝ) (t : ℝ) (j : Fin 14)
    (hc : ∀ i, HasDerivAt (curve i) (dx i) t)
    (hr : Regular ((stepE w (runE w s past) current).1 j) (fun i => curve i t)) :
    HasDerivAt (fun z => (step (interpParameters w (fun i => curve i z))
      (runState (interpParameters w (fun i => curve i z))
        (interpState s (fun i => curve i z))
        (past.map (fun x => interpInput x (fun i => curve i z))))
      (interpInput current (fun i => curve i z))).1 j)
      (tangent ((stepE w (runE w s past) current).1 j) (fun i => curve i t) dx) t := by
  have h := complete_graph_ad_correct w (runE w s past) current curve dx t j hc hr
  simpa only [all_history_compiler_correct] using h

/-- Epsilon normalization is differentiable at the zero key. The compiler
places the positive floor INSIDE sqrt, avoiding a spurious sqrt(0) AD premise. -/
theorem normalized_zero_regular (j : Fin 8) :
    Regular ((normE (fun i : Fin 8 => Expr.input i)) j) (fun _ => 0) := by
  norm_num [normE,Regular,eval,sumE]

theorem normalized_zero_tangent (j : Fin 8) (dx : Fin 8 → ℝ) :
    tangent ((normE (fun i : Fin 8 => Expr.input i)) j) (fun _ => 0) dx =
      dx j/(1/1000000) := by
  norm_num [normE,tangent,eval,sumE]

theorem normalized_epsilon_boundary_not_regular (j : Fin 8) :
    ¬ Regular ((normE (fun i : Fin 8 => Expr.input i)) j)
      (fun i => if i=0 then 1/1000000 else 0) := by
  norm_num [normE,Regular,eval,sumE]

end
end PNDS.SourceAlignedPrograms
