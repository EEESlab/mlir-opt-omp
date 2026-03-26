module @"/home/tagliavini/MLIR/grammar/mlir-transform/test/test2.c" attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
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
    omp.parallel private(@j.privatizer %13 -> %arg5, @k.privatizer %15 -> %arg6 : !llvm.ptr, !llvm.ptr) {
      %17 = llvm.mlir.constant(1 : i64) : i64
      %18 = llvm.alloca %17 x i32 {alignment = 4 : i64} : (i64) -> !llvm.ptr
      %19 = llvm.mlir.addressof @".str" : !llvm.ptr
      %20 = llvm.getelementptr %19[0] : (!llvm.ptr) -> !llvm.ptr, i8
      %21 = llvm.call @omp_get_num_threads() {no_unwind} : () -> i32
      %22 = llvm.call @printf(%20, %21) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
      %23 = llvm.mlir.constant(0 : i32) : i32
      %24 = llvm.mlir.constant(128 : i32) : i32
      %25 = llvm.mlir.constant(1 : i32) : i32
      llvm.store %23, %18 {alignment = 4 : i64} : i32, !llvm.ptr
      omp.wsloop {
        omp.loop_nest (%arg7) : i32 = (%23) to (%24) step (%25) {
          llvm.store %arg7, %18 {alignment = 4 : i64} : i32, !llvm.ptr
          llvm.br ^bb1
        ^bb1:  // pred: ^bb0
          %26 = llvm.mlir.constant(0 : i32) : i32
          llvm.store %26, %arg5 {alignment = 1 : i64} : i32, !llvm.ptr
          llvm.br ^bb2
        ^bb2:  // 2 preds: ^bb1, ^bb14
          %27 = llvm.load %arg5 {alignment = 1 : i64} : !llvm.ptr -> i32
          %28 = llvm.mlir.constant(128 : i32) : i32
          %29 = llvm.icmp "slt" %27, %28 : i32
          llvm.cond_br %29, ^bb3, ^bb15
        ^bb3:  // pred: ^bb2
          llvm.br ^bb4
        ^bb4:  // pred: ^bb3
          %30 = llvm.load %3 {alignment = 8 : i64} : !llvm.ptr -> f64
          %31 = llvm.load %arg5 {alignment = 1 : i64} : !llvm.ptr -> i32
          %32 = llvm.load %18 {alignment = 4 : i64} : !llvm.ptr -> i32
          %33 = llvm.load %9 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
          %34 = llvm.sext %32 : i32 to i64
          %35 = llvm.getelementptr %33[%34] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<128 x f64>
          %36 = llvm.sext %31 : i32 to i64
          %37 = llvm.getelementptr %35[0, %36] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<128 x f64>
          %38 = llvm.load %37 {alignment = 8 : i64} : !llvm.ptr -> f64
          %39 = llvm.fmul %38, %30 : f64
          llvm.store %39, %37 {alignment = 8 : i64} : f64, !llvm.ptr
          llvm.br ^bb5
        ^bb5:  // pred: ^bb4
          %40 = llvm.mlir.constant(0 : i32) : i32
          llvm.store %40, %arg6 {alignment = 1 : i64} : i32, !llvm.ptr
          llvm.br ^bb6
        ^bb6:  // 2 preds: ^bb5, ^bb10
          %41 = llvm.load %arg6 {alignment = 1 : i64} : !llvm.ptr -> i32
          %42 = llvm.mlir.constant(128 : i32) : i32
          %43 = llvm.icmp "slt" %41, %42 : i32
          llvm.cond_br %43, ^bb7, ^bb11
        ^bb7:  // pred: ^bb6
          llvm.br ^bb8
        ^bb8:  // pred: ^bb7
          %44 = llvm.load %1 {alignment = 8 : i64} : !llvm.ptr -> f64
          %45 = llvm.load %arg6 {alignment = 1 : i64} : !llvm.ptr -> i32
          %46 = llvm.load %18 {alignment = 4 : i64} : !llvm.ptr -> i32
          %47 = llvm.load %5 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
          %48 = llvm.sext %46 : i32 to i64
          %49 = llvm.getelementptr %47[%48] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<128 x f64>
          %50 = llvm.sext %45 : i32 to i64
          %51 = llvm.getelementptr %49[0, %50] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<128 x f64>
          %52 = llvm.load %51 {alignment = 8 : i64} : !llvm.ptr -> f64
          %53 = llvm.fmul %44, %52 : f64
          %54 = llvm.load %arg5 {alignment = 1 : i64} : !llvm.ptr -> i32
          %55 = llvm.load %arg6 {alignment = 1 : i64} : !llvm.ptr -> i32
          %56 = llvm.load %7 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
          %57 = llvm.sext %55 : i32 to i64
          %58 = llvm.getelementptr %56[%57] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<128 x f64>
          %59 = llvm.sext %54 : i32 to i64
          %60 = llvm.getelementptr %58[0, %59] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<128 x f64>
          %61 = llvm.load %60 {alignment = 8 : i64} : !llvm.ptr -> f64
          %62 = llvm.fmul %53, %61 : f64
          %63 = llvm.load %arg5 {alignment = 1 : i64} : !llvm.ptr -> i32
          %64 = llvm.load %18 {alignment = 4 : i64} : !llvm.ptr -> i32
          %65 = llvm.load %9 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
          %66 = llvm.sext %64 : i32 to i64
          %67 = llvm.getelementptr %65[%66] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<128 x f64>
          %68 = llvm.sext %63 : i32 to i64
          %69 = llvm.getelementptr %67[0, %68] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<128 x f64>
          %70 = llvm.load %69 {alignment = 8 : i64} : !llvm.ptr -> f64
          %71 = llvm.fadd %70, %62 : f64
          llvm.store %71, %69 {alignment = 8 : i64} : f64, !llvm.ptr
          llvm.br ^bb9
        ^bb9:  // pred: ^bb8
          llvm.br ^bb10
        ^bb10:  // pred: ^bb9
          %72 = llvm.load %arg6 {alignment = 1 : i64} : !llvm.ptr -> i32
          %73 = llvm.mlir.constant(1 : i32) : i32
          %74 = llvm.add %72, %73 overflow<nsw> : i32
          llvm.store %74, %arg6 {alignment = 1 : i64} : i32, !llvm.ptr
          llvm.br ^bb6
        ^bb11:  // pred: ^bb6
          llvm.br ^bb12
        ^bb12:  // pred: ^bb11
          llvm.br ^bb13
        ^bb13:  // pred: ^bb12
          llvm.br ^bb14
        ^bb14:  // pred: ^bb13
          %75 = llvm.load %arg5 {alignment = 1 : i64} : !llvm.ptr -> i32
          %76 = llvm.mlir.constant(1 : i32) : i32
          %77 = llvm.add %75, %76 overflow<nsw> : i32
          llvm.store %77, %arg5 {alignment = 1 : i64} : i32, !llvm.ptr
          llvm.br ^bb2
        ^bb15:  // pred: ^bb2
          llvm.br ^bb16
        ^bb16:  // pred: ^bb15
          omp.yield
        }
      }
      omp.terminator
    }
    llvm.return
  }
}

