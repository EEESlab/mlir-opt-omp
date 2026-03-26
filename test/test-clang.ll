; ModuleID = 'test2.c'
source_filename = "test2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.ident_t = type { i32, i32, i32, i32, ptr }

@.str = private unnamed_addr constant [14 x i8] c"THREADS = %d\0A\00", align 1
@0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 514, i32 0, i32 22, ptr @0 }, align 8
@2 = private unnamed_addr constant %struct.ident_t { i32 0, i32 66, i32 0, i32 22, ptr @0 }, align 8
@3 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 22, ptr @0 }, align 8

; Function Attrs: nounwind uwtable
define dso_local void @gemm(double noundef %0, double noundef %1, ptr noundef %2, ptr noundef %3, ptr noundef %4) local_unnamed_addr #0 {
  %6 = alloca double, align 8
  %7 = alloca double, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca ptr, align 8
  store double %0, ptr %6, align 8, !tbaa !10
  store double %1, ptr %7, align 8, !tbaa !10
  store ptr %2, ptr %8, align 8, !tbaa !12
  store ptr %3, ptr %9, align 8, !tbaa !12
  store ptr %4, ptr %10, align 8, !tbaa !12
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr nonnull @3, i32 5, ptr nonnull @gemm.omp_outlined, ptr nonnull %10, ptr nonnull %7, ptr nonnull %6, ptr nonnull %8, ptr nonnull %9)
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: alwaysinline norecurse nounwind uwtable
define internal void @gemm.omp_outlined(ptr noalias noundef readonly captures(none) %0, ptr noalias readnone captures(none) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %3, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %4, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %5, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %6) #2 {
  %8 = alloca i32, align 4
  %9 = alloca i32, align 4
  %10 = alloca i32, align 4
  %11 = alloca i32, align 4
  %12 = tail call i32 @omp_get_num_threads() #5
  %13 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %12)
  call void @llvm.lifetime.start.p0(ptr nonnull %8) #5
  store i32 0, ptr %8, align 4, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %9) #5
  store i32 127, ptr %9, align 4, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %10) #5
  store i32 1, ptr %10, align 4, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %11) #5
  store i32 0, ptr %11, align 4, !tbaa !6
  %14 = load i32, ptr %0, align 4, !tbaa !6
  call void @__kmpc_for_static_init_4(ptr nonnull @1, i32 %14, i32 34, ptr nonnull %11, ptr nonnull %8, ptr nonnull %9, ptr nonnull %10, i32 1, i32 1)
  %15 = load i32, ptr %9, align 4, !tbaa !6
  %16 = call i32 @llvm.smin.i32(i32 %15, i32 127)
  store i32 %16, ptr %9, align 4, !tbaa !6
  %17 = load i32, ptr %8, align 4, !tbaa !6
  %18 = icmp sgt i32 %17, %16
  br i1 %18, label %63, label %19

19:                                               ; preds = %7
  %20 = load ptr, ptr %2, align 8, !tbaa !12
  %21 = load ptr, ptr %5, align 8, !tbaa !12
  %22 = load ptr, ptr %6, align 8, !tbaa !12
  %23 = sext i32 %17 to i64
  %24 = add nsw i32 %16, 1
  br label %25

25:                                               ; preds = %19, %59
  %26 = phi i64 [ %23, %19 ], [ %60, %59 ]
  %27 = getelementptr inbounds [128 x double], ptr %20, i64 %26
  %28 = getelementptr inbounds [128 x double], ptr %21, i64 %26
  br label %29

29:                                               ; preds = %25, %56
  %30 = phi i64 [ 0, %25 ], [ %57, %56 ]
  %31 = load double, ptr %3, align 8, !tbaa !10
  %32 = getelementptr inbounds nuw double, ptr %27, i64 %30
  %33 = load double, ptr %32, align 8, !tbaa !10
  %34 = fmul double %31, %33
  store double %34, ptr %32, align 8, !tbaa !10
  %35 = getelementptr inbounds nuw double, ptr %22, i64 %30
  br label %36

36:                                               ; preds = %36, %29
  %37 = phi double [ %34, %29 ], [ %53, %36 ]
  %38 = phi i64 [ 0, %29 ], [ %54, %36 ]
  %39 = load double, ptr %4, align 8, !tbaa !10
  %40 = getelementptr inbounds nuw double, ptr %28, i64 %38
  %41 = load double, ptr %40, align 8, !tbaa !10
  %42 = fmul double %39, %41
  %43 = getelementptr inbounds nuw [128 x double], ptr %35, i64 %38
  %44 = load double, ptr %43, align 8, !tbaa !10
  %45 = call double @llvm.fmuladd.f64(double %42, double %44, double %37)
  store double %45, ptr %32, align 8, !tbaa !10
  %46 = or disjoint i64 %38, 1
  %47 = load double, ptr %4, align 8, !tbaa !10
  %48 = getelementptr inbounds nuw double, ptr %28, i64 %46
  %49 = load double, ptr %48, align 8, !tbaa !10
  %50 = fmul double %47, %49
  %51 = getelementptr inbounds nuw [128 x double], ptr %35, i64 %46
  %52 = load double, ptr %51, align 8, !tbaa !10
  %53 = call double @llvm.fmuladd.f64(double %50, double %52, double %45)
  store double %53, ptr %32, align 8, !tbaa !10
  %54 = add nuw nsw i64 %38, 2
  %55 = icmp eq i64 %54, 128
  br i1 %55, label %56, label %36, !llvm.loop !15

56:                                               ; preds = %36
  %57 = add nuw nsw i64 %30, 1
  %58 = icmp eq i64 %57, 128
  br i1 %58, label %59, label %29, !llvm.loop !17

59:                                               ; preds = %56
  %60 = add nsw i64 %26, 1
  %61 = trunc i64 %60 to i32
  %62 = icmp eq i32 %24, %61
  br i1 %62, label %63, label %25

63:                                               ; preds = %59, %7
  call void @__kmpc_for_static_fini(ptr nonnull @1, i32 %14)
  call void @llvm.lifetime.end.p0(ptr nonnull %11) #5
  call void @llvm.lifetime.end.p0(ptr nonnull %10) #5
  call void @llvm.lifetime.end.p0(ptr nonnull %9) #5
  call void @llvm.lifetime.end.p0(ptr nonnull %8) #5
  call void @__kmpc_barrier(ptr nonnull @2, i32 %14)
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #3

; Function Attrs: nounwind
declare i32 @omp_get_num_threads() local_unnamed_addr #4

; Function Attrs: nounwind
declare void @__kmpc_for_static_init_4(ptr, i32, i32, ptr, ptr, ptr, ptr, i32, i32) local_unnamed_addr #5

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #6

; Function Attrs: nounwind
declare void @__kmpc_for_static_fini(ptr, i32) local_unnamed_addr #5

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: convergent nounwind
declare void @__kmpc_barrier(ptr, i32) local_unnamed_addr #7

; Function Attrs: nounwind
declare !callback !18 void @__kmpc_fork_call(ptr, i32, ptr, ...) local_unnamed_addr #5

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smin.i32(i32, i32) #8

attributes #0 = { nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { alwaysinline norecurse nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { nofree nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #4 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #5 = { nounwind }
attributes #6 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #7 = { convergent nounwind }
attributes #8 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}
!llvm.errno.tbaa = !{!6}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"openmp", i32 51}
!2 = !{i32 8, !"PIC Level", i32 2}
!3 = !{i32 7, !"PIE Level", i32 2}
!4 = !{i32 7, !"uwtable", i32 2}
!5 = !{!"clang version 23.0.0git (git@github.com:EEESlab/llvm-project.git 87bb8e20d8c841c0691db5c24a2a68f89480e79e)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"double", !8, i64 0}
!12 = !{!13, !13, i64 0}
!13 = !{!"p1 double", !14, i64 0}
!14 = !{!"any pointer", !8, i64 0}
!15 = distinct !{!15, !16}
!16 = !{!"llvm.loop.mustprogress"}
!17 = distinct !{!17, !16}
!18 = !{!19}
!19 = !{i64 2, i64 -1, i64 -1, i1 true}
