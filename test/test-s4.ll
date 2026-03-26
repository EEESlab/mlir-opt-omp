; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-n8:16:32:64-S128-p270:32:32:32:32-p271:32:32:32:32-p272:64:64:64:64-i64:64-i128:128-f80:128-p0:64:64:64:64-i1:8-i8:8-i16:16-i32:32-f16:16-f64:64-f128:128"
target triple = "x86_64-unknown-linux-gnu"

@__omp_ident_0 = private constant { i32, i32, i32, i32, ptr } { i32 0, i32 2, i32 0, i32 0, ptr null }
@.str = private constant [14 x i8] c"THREADS = %d\0A\00", align 1

define void @outlined_parallel_0(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, ptr %9, ptr %10) {
  %12 = alloca i32, i64 1, align 4
  %13 = load i32, ptr %2, align 4
  store i32 %13, ptr %12, align 4
  %14 = alloca i32, i64 1, align 4
  %15 = load i32, ptr %3, align 4
  store i32 %15, ptr %14, align 4
  %16 = alloca i32, i64 1, align 4
  %17 = call i32 @omp_get_num_threads() #1
  %18 = call i32 (ptr, ...) @printf(ptr @.str, i32 %17)
  store i32 0, ptr %16, align 4
  %19 = load i32, ptr %0, align 4
  %20 = alloca i32, i64 1, align 4
  %21 = alloca i32, i64 1, align 4
  %22 = alloca i32, i64 1, align 4
  %23 = alloca i32, i64 1, align 4
  store i32 0, ptr %20, align 4
  store i32 127, ptr %21, align 4
  store i32 0, ptr %23, align 4
  call void @__kmpc_for_static_init_4(ptr @__omp_ident_0, i32 %19, i32 34, ptr %23, ptr %20, ptr %21, ptr %22, i32 1, i32 0)
  %24 = load i32, ptr %20, align 4
  %25 = load i32, ptr %21, align 4
  %26 = alloca i32, i64 1, align 4
  store i32 %24, ptr %26, align 4
  br label %27

27:                                               ; preds = %93, %11
  %28 = load i32, ptr %26, align 4
  %29 = icmp sle i32 %28, %25
  br i1 %29, label %30, label %95

30:                                               ; preds = %27
  store i32 %28, ptr %16, align 4
  br label %31

31:                                               ; preds = %30
  store i32 0, ptr %12, align 4
  br label %32

32:                                               ; preds = %88, %31
  %33 = load i32, ptr %12, align 4
  %34 = icmp slt i32 %33, 128
  br i1 %34, label %35, label %91

35:                                               ; preds = %32
  br label %36

36:                                               ; preds = %35
  %37 = load double, ptr %4, align 8
  %38 = load i32, ptr %12, align 4
  %39 = load i32, ptr %16, align 4
  %40 = load ptr, ptr %5, align 8
  %41 = sext i32 %39 to i64
  %42 = getelementptr [128 x double], ptr %40, i64 %41
  %43 = sext i32 %38 to i64
  %44 = getelementptr [128 x double], ptr %42, i32 0, i64 %43
  %45 = load double, ptr %44, align 8
  %46 = fmul double %45, %37
  store double %46, ptr %44, align 8
  br label %47

47:                                               ; preds = %36
  store i32 0, ptr %14, align 4
  br label %48

48:                                               ; preds = %82, %47
  %49 = load i32, ptr %14, align 4
  %50 = icmp slt i32 %49, 128
  br i1 %50, label %51, label %85

51:                                               ; preds = %48
  br label %52

52:                                               ; preds = %51
  %53 = load double, ptr %6, align 8
  %54 = load i32, ptr %14, align 4
  %55 = load i32, ptr %16, align 4
  %56 = load ptr, ptr %7, align 8
  %57 = sext i32 %55 to i64
  %58 = getelementptr [128 x double], ptr %56, i64 %57
  %59 = sext i32 %54 to i64
  %60 = getelementptr [128 x double], ptr %58, i32 0, i64 %59
  %61 = load double, ptr %60, align 8
  %62 = fmul double %53, %61
  %63 = load i32, ptr %12, align 4
  %64 = load i32, ptr %14, align 4
  %65 = load ptr, ptr %8, align 8
  %66 = sext i32 %64 to i64
  %67 = getelementptr [128 x double], ptr %65, i64 %66
  %68 = sext i32 %63 to i64
  %69 = getelementptr [128 x double], ptr %67, i32 0, i64 %68
  %70 = load double, ptr %69, align 8
  %71 = fmul double %62, %70
  %72 = load i32, ptr %12, align 4
  %73 = load i32, ptr %16, align 4
  %74 = load ptr, ptr %5, align 8
  %75 = sext i32 %73 to i64
  %76 = getelementptr [128 x double], ptr %74, i64 %75
  %77 = sext i32 %72 to i64
  %78 = getelementptr [128 x double], ptr %76, i32 0, i64 %77
  %79 = load double, ptr %78, align 8
  %80 = fadd double %79, %71
  store double %80, ptr %78, align 8
  br label %81

81:                                               ; preds = %52
  br label %82

82:                                               ; preds = %81
  %83 = load i32, ptr %14, align 4
  %84 = add nsw i32 %83, 1
  store i32 %84, ptr %14, align 4
  br label %48

85:                                               ; preds = %48
  br label %86

86:                                               ; preds = %85
  br label %87

87:                                               ; preds = %86
  br label %88

88:                                               ; preds = %87
  %89 = load i32, ptr %12, align 4
  %90 = add nsw i32 %89, 1
  store i32 %90, ptr %12, align 4
  br label %32

91:                                               ; preds = %32
  br label %92

92:                                               ; preds = %91
  br label %93

93:                                               ; preds = %92
  %94 = add i32 %28, 1
  store i32 %94, ptr %26, align 4
  br label %27

95:                                               ; preds = %27
  call void @__kmpc_for_static_fini(ptr @__omp_ident_0, i32 %19)
  call void @__kmpc_barrier(ptr @__omp_ident_0, i32 %19)
  ret void
}

declare i32 @printf(ptr, ...)

declare i32 @omp_get_num_threads()

; Function Attrs: noinline
define dso_local void @gemm(double %0, double %1, ptr %2, ptr %3, ptr %4) #0 {
  %6 = alloca double, i64 1, align 8
  %7 = alloca double, i64 1, align 8
  %8 = alloca ptr, i64 1, align 8
  %9 = alloca ptr, i64 1, align 8
  %10 = alloca ptr, i64 1, align 8
  %11 = alloca i32, i64 1, align 4
  %12 = alloca i32, i64 1, align 4
  %13 = alloca i32, i64 1, align 4
  store double %0, ptr %6, align 8
  store double %1, ptr %7, align 8
  store ptr %2, ptr %8, align 8
  store ptr %3, ptr %9, align 8
  store ptr %4, ptr %10, align 8
  store i32 42, ptr %11, align 4
  %14 = call i32 @__kmpc_global_thread_num(ptr @__omp_ident_0)
  call void @__kmpc_fork_call(ptr @__omp_ident_0, i32 7, ptr @outlined_parallel_0, ptr %12, ptr %13, ptr %7, ptr %10, ptr %6, ptr %8, ptr %9)
  ret void
}

declare i32 @__kmpc_global_thread_num(ptr)

declare void @__kmpc_fork_call(ptr, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr)

declare void @__kmpc_for_static_init_4(ptr, i32, i32, ptr, ptr, ptr, ptr, i32, i32)

declare void @__kmpc_for_static_fini(ptr, i32)

declare void @__kmpc_barrier(ptr, i32)

attributes #0 = { noinline }
attributes #1 = { nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
