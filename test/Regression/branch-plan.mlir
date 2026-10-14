// `branch` in the lowering DSL: unlike when/otherwise, which are decided while
// the rules are evaluated and collapse into a straight line, a branch survives
// into the plan and PlanLoweringPass emits real control flow for it.
//
// The rule file is Inputs/branch.dsl, not the shipped rules.dsl — see the
// comment there for why the real rules cannot exercise this yet.
//
// RUN: mlir-opt-omp %s --omp-lower-dsl=%S/Inputs/branch.dsl \
// RUN:   --omp-lower-runtime=iomp --omp-to-omp-lower --omp-outline \
// RUN:   --omp-lower-plan | FileCheck %s
//
// The branch is still in the plan after the first pass, rather than having been
// flattened to one arm the way a when/otherwise chain would be:
// RUN: mlir-opt-omp %s --omp-lower-dsl=%S/Inputs/branch.dsl \
// RUN:   --omp-lower-runtime=iomp --omp-to-omp-lower \
// RUN:   | FileCheck %s --check-prefix=PLAN

func.func @bar() {
  omp.barrier
  return
}

func.func @tw() {
  omp.taskwait
  return
}

// PLAN:      omp_lower.construct
// PLAN-SAME:   #omp_lower.branch<
// PLAN-SAME:   on_true
// PLAN-SAME:   on_false_first

// Both arms: the condition is normalised to i1, each arm gets a block, and both
// join at the continuation where the rest of the function continues.
// CHECK-LABEL: func.func @bar
// CHECK:         %[[GTID:.*]] = call @__kmpc_global_thread_num
// CHECK:         %[[C:.*]] = llvm.icmp "ne" %[[GTID]]
// CHECK:         llvm.cond_br %[[C]], ^[[T:bb[0-9]+]], ^[[F:bb[0-9]+]]
// CHECK:       ^[[T]]:
// CHECK:         call @on_true(
// CHECK:         llvm.br ^[[J:bb[0-9]+]]
// CHECK:       ^[[F]]:
// CHECK:         call @on_false_first(
// CHECK:         call @on_false_second(
// CHECK:         llvm.br ^[[J]]
// CHECK:       ^[[J]]:
// CHECK:         return

// A missing arm is an empty block that still jumps to the join, so the CFG stays
// well formed and the call appears on one side only.
// CHECK-LABEL: func.func @tw
// CHECK:         llvm.cond_br %{{.*}}, ^[[T2:bb[0-9]+]], ^[[F2:bb[0-9]+]]
// CHECK:       ^[[T2]]:
// CHECK:         call @only_on_true()
// CHECK:         llvm.br ^[[J2:bb[0-9]+]]
// CHECK:       ^[[F2]]:
// CHECK-NEXT:    llvm.br ^[[J2]]
