module @"firstprivate.c" attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  omp.private {type = firstprivate} @x.privatizer : i32 init {
  ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
    omp.yield(%arg1 : !llvm.ptr)
  } copy {
  ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
    %0 = llvm.load %arg0 : !llvm.ptr -> i32
    llvm.store %0, %arg1 : i32, !llvm.ptr
    omp.yield(%arg1 : !llvm.ptr)
  }
  llvm.mlir.global external @saw() {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<4 x i32> {
    %0 = llvm.mlir.zero : !llvm.array<4 x i32>
    llvm.return %0 : !llvm.array<4 x i32>
  }
  llvm.mlir.global private constant @".str"() {addr_space = 0 : i32, alignment = 1 : i64, dso_local} : !llvm.array<4 x i8> {
    %0 = llvm.mlir.undef : !llvm.array<4 x i8>
    %1 = llvm.mlir.constant(37 : i8) : i8
    %2 = llvm.insertvalue %1, %0[0] : !llvm.array<4 x i8> 
    %3 = llvm.mlir.constant(100 : i8) : i8
    %4 = llvm.insertvalue %3, %2[1] : !llvm.array<4 x i8> 
    %5 = llvm.mlir.constant(10 : i8) : i8
    %6 = llvm.insertvalue %5, %4[2] : !llvm.array<4 x i8> 
    %7 = llvm.mlir.constant(0 : i8) : i8
    %8 = llvm.insertvalue %7, %6[3] : !llvm.array<4 x i8> 
    llvm.return %8 : !llvm.array<4 x i8>
  }
  llvm.func @omp_get_thread_num() -> i32 attributes {nothrow, sym_visibility = "private"}
  llvm.func @printf(!llvm.ptr, ...) -> i32 attributes {sym_visibility = "private"}
  llvm.func @main() -> i32 attributes {dso_local, no_inline} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = llvm.alloca %2 x i32 {alignment = 4 : i64} : (i64) -> !llvm.ptr
    %4 = llvm.mlir.constant(1 : i64) : i64
    %5 = llvm.alloca %4 x i32 {alignment = 4 : i64} : (i64) -> !llvm.ptr
    %6 = llvm.mlir.constant(1 : i64) : i64
    %7 = llvm.alloca %6 x i32 {alignment = 4 : i64} : (i64) -> !llvm.ptr
    %8 = llvm.mlir.constant(42 : i32) : i32
    llvm.store %8, %3 {alignment = 4 : i64} : i32, !llvm.ptr
    %9 = llvm.mlir.constant(4 : i32) : i32
    omp.parallel num_threads(%9 : i32) private(@x.privatizer %3 -> %arg0 : !llvm.ptr) {
      %40 = llvm.load %arg0 {alignment = 1 : i64} : !llvm.ptr -> i32
      %41 = llvm.call @omp_get_thread_num() {no_unwind} : () -> i32
      %42 = llvm.mlir.addressof @saw : !llvm.ptr
      %43 = llvm.sext %41 : i32 to i64
      %44 = llvm.getelementptr %42[0, %43] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i32>
      llvm.store %40, %44 {alignment = 4 : i64} : i32, !llvm.ptr
      %45 = llvm.mlir.constant(100 : i32) : i32
      %46 = llvm.load %arg0 {alignment = 1 : i64} : !llvm.ptr -> i32
      %47 = llvm.add %46, %45 overflow<nsw> : i32
      llvm.store %47, %arg0 {alignment = 1 : i64} : i32, !llvm.ptr
      omp.terminator
    }
    %10 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
    %11 = llvm.mlir.constant(42 : i32) : i32
    %12 = llvm.icmp "eq" %10, %11 : i32
    %13 = llvm.zext %12 : i1 to i32
    llvm.store %13, %7 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %14 = llvm.mlir.constant(0 : i32) : i32
    llvm.store %14, %5 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb1, ^bb8
    %15 = llvm.load %5 {alignment = 4 : i64} : !llvm.ptr -> i32
    %16 = llvm.mlir.constant(4 : i32) : i32
    %17 = llvm.icmp "slt" %15, %16 : i32
    llvm.cond_br %17, ^bb3, ^bb9
  ^bb3:  // pred: ^bb2
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    %18 = llvm.load %5 {alignment = 4 : i64} : !llvm.ptr -> i32
    %19 = llvm.mlir.addressof @saw : !llvm.ptr
    %20 = llvm.sext %18 : i32 to i64
    %21 = llvm.getelementptr %19[0, %20] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i32>
    %22 = llvm.load %21 {alignment = 4 : i64} : !llvm.ptr -> i32
    %23 = llvm.mlir.constant(42 : i32) : i32
    %24 = llvm.icmp "ne" %22, %23 : i32
    llvm.cond_br %24, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %25 = llvm.mlir.constant(0 : i32) : i32
    llvm.store %25, %7 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb6
  ^bb6:  // 2 preds: ^bb4, ^bb5
    llvm.br ^bb7
  ^bb7:  // pred: ^bb6
    llvm.br ^bb8
  ^bb8:  // pred: ^bb7
    %26 = llvm.load %5 {alignment = 4 : i64} : !llvm.ptr -> i32
    %27 = llvm.mlir.constant(1 : i32) : i32
    %28 = llvm.add %26, %27 overflow<nsw> : i32
    llvm.store %28, %5 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb2
  ^bb9:  // pred: ^bb2
    llvm.br ^bb10
  ^bb10:  // pred: ^bb9
    %29 = llvm.mlir.addressof @".str" : !llvm.ptr
    %30 = llvm.getelementptr %29[0] : (!llvm.ptr) -> !llvm.ptr, i8
    %31 = llvm.load %7 {alignment = 4 : i64} : !llvm.ptr -> i32
    %32 = llvm.mlir.constant(0 : i32) : i32
    %33 = llvm.icmp "ne" %31, %32 : i32
    llvm.cond_br %33, ^bb11, ^bb12
  ^bb11:  // pred: ^bb10
    %34 = llvm.mlir.constant(42 : i32) : i32
    llvm.br ^bb13(%34 : i32)
  ^bb12:  // pred: ^bb10
    %35 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
    llvm.br ^bb13(%35 : i32)
  ^bb13(%36: i32):  // 2 preds: ^bb11, ^bb12
    llvm.br ^bb14
  ^bb14:  // pred: ^bb13
    %37 = llvm.call @printf(%30, %36) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
    %38 = llvm.mlir.constant(0 : i32) : i32
    llvm.store %38, %1 {alignment = 4 : i64} : i32, !llvm.ptr
    %39 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
    llvm.return %39 : i32
  }
}

