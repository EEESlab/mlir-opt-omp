// pmsis parallel through the WHOLE pipeline, down to the emitted call.
//
// This is the first construct with a region whose invoke is emitted by
// PlanLoweringPass rather than by the outlining pass (`lower_in = plan` in
// rules.dsl): the outlining pass attaches the outlined function pointer and the
// capture struct as named operands and leaves the construct standing.
//
// The other pmsis tests do NOT cover this.  parallel-pmsis.mlir names
// ext_pi_cl_team_fork but only inside the plan attribute, before outlining even
// runs; the wsloop and nowait ones stop at --omp-outline and look inside the
// outlined function, never at the call site.  Without the checks below, losing
// the fork call entirely would leave the suite green.
//
// RUN: mlir-opt-omp %s --omp-lower-dsl=%rules_dsl --omp-lower-runtime=pmsis \
// RUN:   --omp-to-omp-lower --omp-outline --omp-lower-plan | FileCheck %s

llvm.func @use(!llvm.ptr)

func.func @parallel_capture(%arg0: !llvm.ptr) {
  omp.parallel {
    llvm.call @use(%arg0) : (!llvm.ptr) -> ()
    omp.terminator
  }
  return
}

// The body still gets outlined into the single-ptr closure.
// CHECK:       func.func {{.*}}@outlined_parallel_{{[0-9]+}}(%{{[^,)]*}}: !llvm.ptr)

// The fork is a real call, with the team size from rules.dsl and the two
// artifacts the outlining pass handed over as operands.  The function pointer
// must be the cast of the outlined symbol — an undef there means a binding was
// missed and the region would never run.
// CHECK-LABEL: func.func @parallel_capture
// CHECK-DAG:     %[[EIGHT:.*]] = arith.constant 8 : i32
// CHECK-DAG:     %[[FN:.*]] = builtin.unrealized_conversion_cast %{{.*}} to !llvm.ptr
// CHECK:         call @ext_pi_cl_team_fork(%[[EIGHT]], %[[FN]], %{{.*}}) : (i32, !llvm.ptr, !llvm.ptr) -> ()

// Nothing may survive the three passes: a leftover construct means the plan
// pass never consumed it, which is exactly the failure the checks above would
// otherwise miss.
// CHECK-NOT:   omp_lower.construct
