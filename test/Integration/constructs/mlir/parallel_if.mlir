module @"parallel_if.c" attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  llvm.mlir.global external @seen() {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<256 x i32> {
    %0 = llvm.mlir.zero : !llvm.array<256 x i32>
    llvm.return %0 : !llvm.array<256 x i32>
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
    %6 = llvm.mlir.constant(4 : i32) : i32
    %7 = llvm.mlir.constant(0 : i32) : i32
    %8 = llvm.mlir.constant(0 : i32) : i32
    %9 = llvm.icmp "ne" %7, %8 : i32
    omp.parallel if(%9) num_threads(%6 : i32) {
      %36 = llvm.mlir.constant(1 : i32) : i32
      %37 = llvm.call @omp_get_thread_num() {no_unwind} : () -> i32
      %38 = llvm.mlir.addressof @seen : !llvm.ptr
      %39 = llvm.sext %37 : i32 to i64
      %40 = llvm.getelementptr %38[0, %39] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<256 x i32>
      llvm.store %36, %40 {alignment = 4 : i64} : i32, !llvm.ptr
      omp.terminator
    }
    %10 = llvm.mlir.constant(0 : i32) : i32
    llvm.store %10, %5 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %11 = llvm.mlir.constant(0 : i32) : i32
    llvm.store %11, %3 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb1, ^bb4
    %12 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
    %13 = llvm.mlir.constant(256 : i32) : i32
    %14 = llvm.icmp "slt" %12, %13 : i32
    llvm.cond_br %14, ^bb3, ^bb5
  ^bb3:  // pred: ^bb2
    %15 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
    %16 = llvm.mlir.addressof @seen : !llvm.ptr
    %17 = llvm.sext %15 : i32 to i64
    %18 = llvm.getelementptr %16[0, %17] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<256 x i32>
    %19 = llvm.load %18 {alignment = 4 : i64} : !llvm.ptr -> i32
    %20 = llvm.load %5 {alignment = 4 : i64} : !llvm.ptr -> i32
    %21 = llvm.add %20, %19 overflow<nsw> : i32
    llvm.store %21, %5 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    %22 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
    %23 = llvm.mlir.constant(1 : i32) : i32
    %24 = llvm.add %22, %23 overflow<nsw> : i32
    llvm.store %24, %3 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb2
  ^bb5:  // pred: ^bb2
    llvm.br ^bb6
  ^bb6:  // pred: ^bb5
    %25 = llvm.mlir.addressof @".str" : !llvm.ptr
    %26 = llvm.getelementptr %25[0] : (!llvm.ptr) -> !llvm.ptr, i8
    %27 = llvm.load %5 {alignment = 4 : i64} : !llvm.ptr -> i32
    %28 = llvm.mlir.constant(1 : i32) : i32
    %29 = llvm.icmp "eq" %27, %28 : i32
    llvm.cond_br %29, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    %30 = llvm.mlir.constant(42 : i32) : i32
    llvm.br ^bb9(%30 : i32)
  ^bb8:  // pred: ^bb6
    %31 = llvm.load %5 {alignment = 4 : i64} : !llvm.ptr -> i32
    llvm.br ^bb9(%31 : i32)
  ^bb9(%32: i32):  // 2 preds: ^bb7, ^bb8
    llvm.br ^bb10
  ^bb10:  // pred: ^bb9
    %33 = llvm.call @printf(%26, %32) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
    %34 = llvm.mlir.constant(0 : i32) : i32
    llvm.store %34, %1 {alignment = 4 : i64} : i32, !llvm.ptr
    %35 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
    llvm.return %35 : i32
  }
}

