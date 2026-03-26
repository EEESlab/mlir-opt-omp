module @"/home/tagliavini/MLIR/grammar/mlir-transform/test/test2.c" attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  llvm.mlir.global private constant @__omp_ident_0() {addr_space = 0 : i32} : !llvm.struct<(i32, i32, i32, i32, ptr)> {
    %0 = llvm.mlir.undef : !llvm.struct<(i32, i32, i32, i32, ptr)>
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.insertvalue %1, %0[0] : !llvm.struct<(i32, i32, i32, i32, ptr)> 
    %3 = llvm.mlir.constant(2 : i32) : i32
    %4 = llvm.insertvalue %3, %2[1] : !llvm.struct<(i32, i32, i32, i32, ptr)> 
    %5 = llvm.mlir.constant(0 : i32) : i32
    %6 = llvm.insertvalue %5, %4[2] : !llvm.struct<(i32, i32, i32, i32, ptr)> 
    %7 = llvm.mlir.constant(0 : i32) : i32
    %8 = llvm.insertvalue %7, %6[3] : !llvm.struct<(i32, i32, i32, i32, ptr)> 
    %9 = llvm.mlir.constant(0 : i64) : i64
    %10 = llvm.inttoptr %9 : i64 to !llvm.ptr
    %11 = llvm.insertvalue %10, %8[4] : !llvm.struct<(i32, i32, i32, i32, ptr)> 
    llvm.return %11 : !llvm.struct<(i32, i32, i32, i32, ptr)>
  }
  func.func private @outlined_parallel_0(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: !llvm.ptr, %arg10: !llvm.ptr) {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x i32 : (i64) -> !llvm.ptr
    %2 = llvm.load %arg2 : !llvm.ptr -> i32
    llvm.store %2, %1 {alignment = 4 : i64} : i32, !llvm.ptr
    %3 = llvm.alloca %0 x i32 : (i64) -> !llvm.ptr
    %4 = llvm.load %arg3 : !llvm.ptr -> i32
    llvm.store %4, %3 {alignment = 4 : i64} : i32, !llvm.ptr
    %5 = llvm.mlir.constant(1 : i64) : i64
    %6 = llvm.alloca %5 x i32 {alignment = 4 : i64} : (i64) -> !llvm.ptr
    %7 = llvm.mlir.addressof @".str" : !llvm.ptr
    %8 = llvm.getelementptr %7[0] : (!llvm.ptr) -> !llvm.ptr, i8
    %9 = llvm.call @omp_get_num_threads() {no_unwind} : () -> i32
    %10 = llvm.call @printf(%8, %9) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
    %11 = llvm.mlir.constant(0 : i32) : i32
    %12 = llvm.mlir.constant(128 : i32) : i32
    %13 = llvm.mlir.constant(1 : i32) : i32
    llvm.store %11, %6 {alignment = 4 : i64} : i32, !llvm.ptr
    %14 = llvm.mlir.addressof @__omp_ident_0 : !llvm.ptr
    %15 = llvm.mlir.undef : i32
    %16 = llvm.load %arg0 : !llvm.ptr -> i32
    %17 = llvm.mlir.constant(1 : i64) : i64
    %18 = llvm.mlir.constant(0 : i32) : i32
    %19 = llvm.mlir.constant(1 : i32) : i32
    %20 = llvm.alloca %17 x i32 : (i64) -> !llvm.ptr
    %21 = llvm.alloca %17 x i32 : (i64) -> !llvm.ptr
    %22 = llvm.alloca %17 x i32 : (i64) -> !llvm.ptr
    %23 = llvm.alloca %17 x i32 : (i64) -> !llvm.ptr
    %24 = llvm.sub %12, %13 : i32
    llvm.store %11, %20 : i32, !llvm.ptr
    llvm.store %24, %21 : i32, !llvm.ptr
    llvm.store %18, %23 : i32, !llvm.ptr
    %25 = llvm.mlir.constant(34 : i32) : i32
    %26 = llvm.mlir.constant(0 : i32) : i32
    call @__kmpc_for_static_init_4(%14, %16, %25, %23, %20, %21, %22, %13, %26) : (!llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32) -> ()
    %27 = llvm.load %20 : !llvm.ptr -> i32
    %28 = llvm.load %21 : !llvm.ptr -> i32
    %29 = llvm.alloca %17 x i32 : (i64) -> !llvm.ptr
    llvm.store %27, %29 : i32, !llvm.ptr
    llvm.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb19
    %30 = llvm.load %29 : !llvm.ptr -> i32
    %31 = llvm.icmp "sle" %30, %28 : i32
    llvm.cond_br %31, ^bb2, ^bb20
  ^bb2:  // pred: ^bb1
    llvm.store %30, %6 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb3
  ^bb3:  // pred: ^bb2
    %32 = llvm.mlir.constant(0 : i32) : i32
    llvm.store %32, %1 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb4
  ^bb4:  // 2 preds: ^bb3, ^bb16
    %33 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
    %34 = llvm.mlir.constant(128 : i32) : i32
    %35 = llvm.icmp "slt" %33, %34 : i32
    llvm.cond_br %35, ^bb5, ^bb17
  ^bb5:  // pred: ^bb4
    llvm.br ^bb6
  ^bb6:  // pred: ^bb5
    %36 = llvm.load %arg4 {alignment = 8 : i64} : !llvm.ptr -> f64
    %37 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
    %38 = llvm.load %6 {alignment = 4 : i64} : !llvm.ptr -> i32
    %39 = llvm.load %arg5 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %40 = llvm.sext %38 : i32 to i64
    %41 = llvm.getelementptr %39[%40] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<128 x f64>
    %42 = llvm.sext %37 : i32 to i64
    %43 = llvm.getelementptr %41[0, %42] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<128 x f64>
    %44 = llvm.load %43 {alignment = 8 : i64} : !llvm.ptr -> f64
    %45 = llvm.fmul %44, %36 : f64
    llvm.store %45, %43 {alignment = 8 : i64} : f64, !llvm.ptr
    llvm.br ^bb7
  ^bb7:  // pred: ^bb6
    %46 = llvm.mlir.constant(0 : i32) : i32
    llvm.store %46, %3 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb8
  ^bb8:  // 2 preds: ^bb7, ^bb12
    %47 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
    %48 = llvm.mlir.constant(128 : i32) : i32
    %49 = llvm.icmp "slt" %47, %48 : i32
    llvm.cond_br %49, ^bb9, ^bb13
  ^bb9:  // pred: ^bb8
    llvm.br ^bb10
  ^bb10:  // pred: ^bb9
    %50 = llvm.load %arg6 {alignment = 8 : i64} : !llvm.ptr -> f64
    %51 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
    %52 = llvm.load %6 {alignment = 4 : i64} : !llvm.ptr -> i32
    %53 = llvm.load %arg7 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %54 = llvm.sext %52 : i32 to i64
    %55 = llvm.getelementptr %53[%54] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<128 x f64>
    %56 = llvm.sext %51 : i32 to i64
    %57 = llvm.getelementptr %55[0, %56] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<128 x f64>
    %58 = llvm.load %57 {alignment = 8 : i64} : !llvm.ptr -> f64
    %59 = llvm.fmul %50, %58 : f64
    %60 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
    %61 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
    %62 = llvm.load %arg8 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %63 = llvm.sext %61 : i32 to i64
    %64 = llvm.getelementptr %62[%63] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<128 x f64>
    %65 = llvm.sext %60 : i32 to i64
    %66 = llvm.getelementptr %64[0, %65] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<128 x f64>
    %67 = llvm.load %66 {alignment = 8 : i64} : !llvm.ptr -> f64
    %68 = llvm.fmul %59, %67 : f64
    %69 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
    %70 = llvm.load %6 {alignment = 4 : i64} : !llvm.ptr -> i32
    %71 = llvm.load %arg5 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %72 = llvm.sext %70 : i32 to i64
    %73 = llvm.getelementptr %71[%72] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<128 x f64>
    %74 = llvm.sext %69 : i32 to i64
    %75 = llvm.getelementptr %73[0, %74] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<128 x f64>
    %76 = llvm.load %75 {alignment = 8 : i64} : !llvm.ptr -> f64
    %77 = llvm.fadd %76, %68 : f64
    llvm.store %77, %75 {alignment = 8 : i64} : f64, !llvm.ptr
    llvm.br ^bb11
  ^bb11:  // pred: ^bb10
    llvm.br ^bb12
  ^bb12:  // pred: ^bb11
    %78 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
    %79 = llvm.mlir.constant(1 : i32) : i32
    %80 = llvm.add %78, %79 overflow<nsw> : i32
    llvm.store %80, %3 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb8
  ^bb13:  // pred: ^bb8
    llvm.br ^bb14
  ^bb14:  // pred: ^bb13
    llvm.br ^bb15
  ^bb15:  // pred: ^bb14
    llvm.br ^bb16
  ^bb16:  // pred: ^bb15
    %81 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
    %82 = llvm.mlir.constant(1 : i32) : i32
    %83 = llvm.add %81, %82 overflow<nsw> : i32
    llvm.store %83, %1 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb4
  ^bb17:  // pred: ^bb4
    llvm.br ^bb18
  ^bb18:  // pred: ^bb17
    llvm.br ^bb19
  ^bb19:  // pred: ^bb18
    %84 = llvm.add %30, %13 : i32
    llvm.store %84, %29 : i32, !llvm.ptr
    llvm.br ^bb1
  ^bb20:  // pred: ^bb1
    call @__kmpc_for_static_fini(%14, %16) : (!llvm.ptr, i32) -> ()
    call @__kmpc_barrier(%14, %16) : (!llvm.ptr, i32) -> ()
    return
  }
  llvm.mlir.global private constant @".str"() {addr_space = 0 : i32, alignment = 1 : i64, dso_local} : !llvm.array<14 x i8> {
    %0 = llvm.mlir.undef : !llvm.array<14 x i8>
    %1 = llvm.mlir.constant(84 : i8) : i8
    %2 = llvm.insertvalue %1, %0[0] : !llvm.array<14 x i8> 
    %3 = llvm.mlir.constant(72 : i8) : i8
    %4 = llvm.insertvalue %3, %2[1] : !llvm.array<14 x i8> 
    %5 = llvm.mlir.constant(82 : i8) : i8
    %6 = llvm.insertvalue %5, %4[2] : !llvm.array<14 x i8> 
    %7 = llvm.mlir.constant(69 : i8) : i8
    %8 = llvm.insertvalue %7, %6[3] : !llvm.array<14 x i8> 
    %9 = llvm.mlir.constant(65 : i8) : i8
    %10 = llvm.insertvalue %9, %8[4] : !llvm.array<14 x i8> 
    %11 = llvm.mlir.constant(68 : i8) : i8
    %12 = llvm.insertvalue %11, %10[5] : !llvm.array<14 x i8> 
    %13 = llvm.mlir.constant(83 : i8) : i8
    %14 = llvm.insertvalue %13, %12[6] : !llvm.array<14 x i8> 
    %15 = llvm.mlir.constant(32 : i8) : i8
    %16 = llvm.insertvalue %15, %14[7] : !llvm.array<14 x i8> 
    %17 = llvm.mlir.constant(61 : i8) : i8
    %18 = llvm.insertvalue %17, %16[8] : !llvm.array<14 x i8> 
    %19 = llvm.mlir.constant(32 : i8) : i8
    %20 = llvm.insertvalue %19, %18[9] : !llvm.array<14 x i8> 
    %21 = llvm.mlir.constant(37 : i8) : i8
    %22 = llvm.insertvalue %21, %20[10] : !llvm.array<14 x i8> 
    %23 = llvm.mlir.constant(100 : i8) : i8
    %24 = llvm.insertvalue %23, %22[11] : !llvm.array<14 x i8> 
    %25 = llvm.mlir.constant(10 : i8) : i8
    %26 = llvm.insertvalue %25, %24[12] : !llvm.array<14 x i8> 
    %27 = llvm.mlir.constant(0 : i8) : i8
    %28 = llvm.insertvalue %27, %26[13] : !llvm.array<14 x i8> 
    llvm.return %28 : !llvm.array<14 x i8>
  }
  omp.private {type = private} @k.privatizer : i32 init {
  ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
    omp.yield(%arg1 : !llvm.ptr)
  }
  omp.private {type = private} @j.privatizer : i32 init {
  ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
    omp.yield(%arg1 : !llvm.ptr)
  }
  llvm.func @printf(!llvm.ptr, ...) -> i32 attributes {sym_visibility = "private"}
  llvm.func @omp_get_num_threads() -> i32 attributes {nothrow, sym_visibility = "private"}
  llvm.func @gemm(%arg0: f64, %arg1: f64, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr) attributes {dso_local, no_inline} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x f64 {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = llvm.alloca %2 x f64 {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %4 = llvm.mlir.constant(1 : i64) : i64
    %5 = llvm.alloca %4 x !llvm.ptr {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %6 = llvm.mlir.constant(1 : i64) : i64
    %7 = llvm.alloca %6 x !llvm.ptr {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %8 = llvm.mlir.constant(1 : i64) : i64
    %9 = llvm.alloca %8 x !llvm.ptr {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %10 = llvm.mlir.constant(1 : i64) : i64
    %11 = llvm.alloca %10 x i32 {alignment = 4 : i64} : (i64) -> !llvm.ptr
    %12 = llvm.mlir.constant(1 : i64) : i64
    %13 = llvm.alloca %12 x i32 {alignment = 4 : i64} : (i64) -> !llvm.ptr
    %14 = llvm.mlir.constant(1 : i64) : i64
    %15 = llvm.alloca %14 x i32 {alignment = 4 : i64} : (i64) -> !llvm.ptr
    llvm.store %arg0, %1 {alignment = 8 : i64} : f64, !llvm.ptr
    llvm.store %arg1, %3 {alignment = 8 : i64} : f64, !llvm.ptr
    llvm.store %arg2, %5 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %arg3, %7 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %arg4, %9 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    %16 = llvm.mlir.constant(42 : i32) : i32
    llvm.store %16, %11 {alignment = 4 : i64} : i32, !llvm.ptr
    %17 = llvm.mlir.addressof @__omp_ident_0 : !llvm.ptr
    %18 = func.call @__kmpc_global_thread_num(%17) : (!llvm.ptr) -> i32
    %c7_i32 = arith.constant 7 : i32
    %f = func.constant @outlined_parallel_0 : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %19 = builtin.unrealized_conversion_cast %f : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> () to !llvm.ptr
    func.call @__kmpc_fork_call(%17, %c7_i32, %19, %13, %15, %3, %9, %1, %5, %7) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
  func.func private @__kmpc_global_thread_num(!llvm.ptr) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @__kmpc_fork_call(!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @__kmpc_for_static_init_4(!llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @__kmpc_for_static_fini(!llvm.ptr, i32) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @__kmpc_barrier(!llvm.ptr, i32) attributes {llvm.linkage = #llvm.linkage<external>}
}

