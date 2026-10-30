import PNDS.LangMem087Historical

/-! ARCHFUSE089 exact-real, scoped channel-retention delta reference.
The original Torch graph, finite-precision AD, memory usefulness and whole-model
equivalence remain OPEN. Parent: LangMem087 historical-state decomposition. -/

namespace PNDS.ArchFuse089
open PNDS.LangMem086

-- TAC stores matrices indexed by value channel first, then key channel.
def channelRetain (m : Memory) (alpha : Key) : Memory :=
  fun i j => m i j * alpha j

def channelDelta (m : Memory) (k v alpha : Key) (beta : ℝ) : Memory :=
  fun i j =>
    channelRetain m alpha i j +
      beta * (v i - read (channelRetain m alpha) k i) * k j

theorem zero_write_equals_retained (m : Memory) (k v alpha : Key) :
    channelDelta m k v alpha 0 = channelRetain m alpha := by
  funext i j
  simp [channelDelta]

theorem unit_channel_retention (m : Memory) :
    channelRetain m (fun _ => 1) = m := by
  funext i j
  simp [channelRetain]

end PNDS.ArchFuse089
#print axioms PNDS.ArchFuse089.zero_write_equals_retained
#print axioms PNDS.ArchFuse089.unit_channel_retention
