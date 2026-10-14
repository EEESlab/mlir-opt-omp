// A rule file used by branch-plan.mlir only, not the one the tool ships.
//
// rules.dsl cannot exercise `branch` yet: the constructs that need it (iomp
// parallel and task) still have their calls emitted by the outlining pass, so a
// branch in their plan would never be read.  barrier and taskwait are the
// opposite — no clauses of their own, but lowered end to end by
// PlanLoweringPass, which is the pass under test.  Branching them on a bound
// value is artificial, but it is the real emission path.
//
// The condition names global_tid because that is an i32 the pass resolves for
// us; any bound integer token would do.

runtime iomp {
  // Both arms, the false one holding a sequence.
  construct barrier {
    invoke {
      branch global_tid {
        true  => call "on_true"(ident, global_tid);
        false => {
          call "on_false_first"(ident);
          call "on_false_second"(global_tid);
        }
      }
    }
  }

  // Only a `true` arm: the false side must still be a well-formed block that
  // jumps to the join, just with nothing in it.
  construct taskwait {
    invoke {
      branch global_tid {
        true => call "only_on_true"();
      }
    }
  }
}
