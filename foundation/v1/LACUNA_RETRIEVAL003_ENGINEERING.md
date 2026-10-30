# Engineering ledger

- Initial proof commit cd82ab9e73cef95cf1e27ed5ea34f8d5bd5706a0,
  CI41006980829 FAILED on ambiguous `read` in the simplifier. Scale and weighted
  value lemmas compiled; native-read lemma did not. Its failed compiler output
  included an unresolved proof placeholder, never counted as proved.
- Correct the simplifier to explicitly name `PNDS.LangMem086.read`; no equations,
  budgets, experimental gates or empirical results changed. No scientific
  source scoring had run at this correction. Preserve the original failed run.
