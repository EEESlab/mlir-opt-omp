module @"task_firstprivate.c" attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
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
  omp.private {type = firstprivate} @x.privatizer : i32 init {
  ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
    omp.yield(%arg1 : !llvm.ptr)
  } copy {
  ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
    %0 = llvm.load %arg0 : !llvm.ptr -> i32
    llvm.store %0, %arg1 : i32, !llvm.ptr
    omp.yield(%arg1 : !llvm.ptr)
  }
  llvm.func @printf(!llvm.ptr, ...) -> i32 attributes {sym_visibility = "private"}
  llvm.func @main() -> i32 attributes {dso_local, no_inline} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = llvm.alloca %2 x i32 {alignment = 4 : i64} : (i64) -> !llvm.ptr
    %4 = llvm.mlir.constant(1 : i64) : i64
    %5 = llvm.alloca %4 x i32 {alignment = 4 : i64} : (i64) -> !llvm.ptr
    %6 = llvm.mlir.constant(1 : i64) : i64
    %7 = llvm.alloca %6 x !llvm.ptr {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %8 = llvm.mlir.constant(42 : i32) : i32
    llvm.store %8, %3 {alignment = 4 : i64} : i32, !llvm.ptr
    %9 = llvm.mlir.constant(0 : i32) : i32
    llvm.store %9, %5 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.store %5, %7 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    %10 = llvm.mlir.constant(1 : i32) : i32
    omp.parallel num_threads(%10 : i32) {
      omp.task private(@x.privatizer %3 -> %arg0 : !llvm.ptr) {
        %27 = llvm.load %arg0 {alignment = 1 : i64} : !llvm.ptr -> i32
        %28 = llvm.load %7 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
        llvm.store %27, %28 {alignment = 4 : i64} : i32, !llvm.ptr
        %29 = llvm.mlir.constant(999 : i32) : i32
        llvm.store %29, %arg0 {alignment = 1 : i64} : i32, !llvm.ptr
        omp.terminator
      }
      omp.taskwait
      omp.terminator
    }
    %11 = llvm.mlir.addressof @".str" : !llvm.ptr
    %12 = llvm.getelementptr %11[0] : (!llvm.ptr) -> !llvm.ptr, i8
    %13 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
    %14 = llvm.mlir.constant(42 : i32) : i32
    %15 = llvm.icmp "eq" %13, %14 : i32
    llvm.cond_br %15, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %16 = llvm.load %5 {alignment = 4 : i64} : !llvm.ptr -> i32
    %17 = llvm.mlir.constant(42 : i32) : i32
    %18 = llvm.icmp "eq" %16, %17 : i32
    llvm.br ^bb3(%18 : i1)
  ^bb2:  // pred: ^bb0
    %19 = llvm.mlir.constant(false) : i1
    llvm.br ^bb3(%19 : i1)
  ^bb3(%20: i1):  // 2 preds: ^bb1, ^bb2
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    llvm.cond_br %20, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %21 = llvm.mlir.constant(42 : i32) : i32
    llvm.br ^bb7(%21 : i32)
  ^bb6:  // pred: ^bb4
    %22 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
    llvm.br ^bb7(%22 : i32)
  ^bb7(%23: i32):  // 2 preds: ^bb5, ^bb6
    llvm.br ^bb8
  ^bb8:  // pred: ^bb7
    %24 = llvm.call @printf(%12, %23) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
    %25 = llvm.mlir.constant(0 : i32) : i32
    llvm.store %25, %1 {alignment = 4 : i64} : i32, !llvm.ptr
    %26 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
    llvm.return %26 : i32
  }
}

