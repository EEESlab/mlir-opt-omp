; ModuleID = 'test-s4.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-n8:16:32:64-S128-p270:32:32:32:32-p271:32:32:32:32-p272:64:64:64:64-i64:64-i128:128-f80:128-p0:64:64:64:64-i1:8-i8:8-i16:16-i32:32-f16:16-f64:64-f128:128"
target triple = "x86_64-unknown-linux-gnu"

@__omp_ident_0 = private constant { i32, i32, i32, i32, ptr } { i32 0, i32 2, i32 0, i32 0, ptr null }
@.str = private constant [14 x i8] c"THREADS = %d\0A\00", align 1

define void @outlined_parallel_0(ptr readonly captures(none) %0, ptr readnone captures(none) %1, ptr readnone captures(none) %2, ptr readnone captures(none) %3, ptr readonly captures(none) %4, ptr readonly captures(none) %5, ptr readonly captures(none) %6, ptr readonly captures(none) %7, ptr readonly captures(none) %8, ptr readnone captures(none) %9, ptr readnone captures(none) %10) {
  %12 = tail call i32 @omp_get_num_threads() #2
  %13 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @.str, i32 %12)
  %14 = load i32, ptr %0, align 4
  %15 = alloca i32, align 4
  %16 = alloca i32, align 4
  %17 = alloca i32, align 4
  %18 = alloca i32, align 4
  store i32 0, ptr %15, align 4
  store i32 127, ptr %16, align 4
  store i32 0, ptr %18, align 4
  call void @__kmpc_for_static_init_4(ptr nonnull @__omp_ident_0, i32 %14, i32 34, ptr nonnull %18, ptr nonnull %15, ptr nonnull %16, ptr nonnull %17, i32 1, i32 0)
  %19 = load i32, ptr %15, align 4
  %20 = load i32, ptr %16, align 4
  %.not17 = icmp sgt i32 %19, %20
  br i1 %.not17, label %._crit_edge, label %.preheader

.preheader:                                       ; preds = %11, %47
  %.018 = phi i32 [ %48, %47 ], [ %19, %11 ]
  %21 = sext i32 %.018 to i64
  br label %22

22:                                               ; preds = %.preheader, %46
  %indvars.iv20 = phi i64 [ 0, %.preheader ], [ %indvars.iv.next21, %46 ]
  %23 = load double, ptr %4, align 8
  %24 = load ptr, ptr %5, align 8
  %25 = getelementptr [128 x double], ptr %24, i64 %21
  %26 = getelementptr double, ptr %25, i64 %indvars.iv20
  %27 = load double, ptr %26, align 8
  %28 = fmul double %23, %27
  store double %28, ptr %26, align 8
  br label %29

29:                                               ; preds = %22, %29
  %indvars.iv = phi i64 [ 0, %22 ], [ %indvars.iv.next, %29 ]
  %30 = load double, ptr %6, align 8
  %31 = load ptr, ptr %7, align 8
  %32 = getelementptr [128 x double], ptr %31, i64 %21
  %33 = getelementptr double, ptr %32, i64 %indvars.iv
  %34 = load double, ptr %33, align 8
  %35 = fmul double %30, %34
  %36 = load ptr, ptr %8, align 8
  %37 = getelementptr [128 x double], ptr %36, i64 %indvars.iv
  %38 = getelementptr double, ptr %37, i64 %indvars.iv20
  %39 = load double, ptr %38, align 8
  %40 = fmul double %35, %39
  %41 = load ptr, ptr %5, align 8
  %42 = getelementptr [128 x double], ptr %41, i64 %21
  %43 = getelementptr double, ptr %42, i64 %indvars.iv20
  %44 = load double, ptr %43, align 8
  %45 = fadd double %40, %44
  store double %45, ptr %43, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 128
  br i1 %exitcond.not, label %46, label %29

46:                                               ; preds = %29
  %indvars.iv.next21 = add nuw nsw i64 %indvars.iv20, 1
  %exitcond23.not = icmp eq i64 %indvars.iv.next21, 128
  br i1 %exitcond23.not, label %47, label %22

47:                                               ; preds = %46
  %48 = add i32 %.018, 1
  %.not = icmp sgt i32 %48, %20
  br i1 %.not, label %._crit_edge, label %.preheader

._crit_edge:                                      ; preds = %47, %11
  call void @__kmpc_for_static_fini(ptr nonnull @__omp_ident_0, i32 %14)
  call void @__kmpc_barrier(ptr nonnull @__omp_ident_0, i32 %14)
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #0

declare i32 @omp_get_num_threads() local_unnamed_addr

; Function Attrs: noinline
define dso_local void @gemm(double %0, double %1, ptr %2, ptr %3, ptr %4) local_unnamed_addr #1 {
  %6 = alloca double, align 8
  %7 = alloca double, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca ptr, align 8
  %11 = alloca i32, align 4
  %12 = alloca i32, align 4
  store double %0, ptr %6, align 8
  store double %1, ptr %7, align 8
  store ptr %2, ptr %8, align 8
  store ptr %3, ptr %9, align 8
  store ptr %4, ptr %10, align 8
  %13 = tail call i32 @__kmpc_global_thread_num(ptr nonnull @__omp_ident_0)
  call void @__kmpc_fork_call(ptr nonnull @__omp_ident_0, i32 7, ptr nonnull @outlined_parallel_0, ptr nonnull %11, ptr nonnull %12, ptr nonnull %7, ptr nonnull %10, ptr nonnull %6, ptr nonnull %8, ptr nonnull %9)
  ret void
}

declare i32 @__kmpc_global_thread_num(ptr) local_unnamed_addr

declare void @__kmpc_fork_call(ptr, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr) local_unnamed_addr

declare void @__kmpc_for_static_init_4(ptr, i32, i32, ptr, ptr, ptr, ptr, i32, i32) local_unnamed_addr

declare void @__kmpc_for_static_fini(ptr, i32) local_unnamed_addr

declare void @__kmpc_barrier(ptr, i32) local_unnamed_addr

attributes #0 = { nofree nounwind }
attributes #1 = { noinline }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
