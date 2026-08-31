module @"taskwait.c" attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
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
    %8 = llvm.mlir.constant(1 : i64) : i64
    %9 = llvm.alloca %8 x !llvm.ptr {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %10 = llvm.mlir.constant(1 : i64) : i64
    %11 = llvm.alloca %10 x !llvm.ptr {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %12 = llvm.mlir.constant(1 : i64) : i64
    %13 = llvm.alloca %12 x !llvm.ptr {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %14 = llvm.mlir.constant(0 : i32) : i32
    llvm.store %14, %3 {alignment = 4 : i64} : i32, !llvm.ptr
    %15 = llvm.mlir.constant(0 : i32) : i32
    llvm.store %15, %5 {alignment = 4 : i64} : i32, !llvm.ptr
    %16 = llvm.mlir.constant(0 : i32) : i32
    llvm.store %16, %7 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.store %3, %9 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %5, %11 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %7, %13 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    %17 = llvm.mlir.constant(4 : i32) : i32
    omp.parallel num_threads(%17 : i32) {
      omp.task {
        %27 = llvm.mlir.constant(1 : i64) : i64
        %28 = llvm.alloca %27 x i32 {alignment = 4 : i64} : (i64) -> !llvm.ptr
        %29 = llvm.mlir.constant(1 : i64) : i64
        %30 = llvm.alloca %29 x i32 {alignment = 4 : i64} : (i64) -> !llvm.ptr
        %31 = llvm.mlir.constant(0 : i32) : i32
        llvm.store %31, %30 {alignment = 4 : i64} : i32, !llvm.ptr
        llvm.br ^bb1
      ^bb1:  // pred: ^bb0
        %32 = llvm.mlir.constant(0 : i32) : i32
        llvm.store %32, %28 {alignment = 4 : i64} : i32, !llvm.ptr
        llvm.br ^bb2
      ^bb2:  // 2 preds: ^bb1, ^bb4
        %33 = llvm.load %28 {alignment = 4 : i64} : !llvm.ptr -> i32
        %34 = llvm.mlir.constant(200000 : i32) : i32
        %35 = llvm.icmp "slt" %33, %34 : i32
        llvm.cond_br %35, ^bb3, ^bb5
      ^bb3:  // pred: ^bb2
        %36 = llvm.load %28 {alignment = 4 : i64} : !llvm.ptr -> i32
        %37 = llvm.mlir.constant(7 : i32) : i32
        %38 = llvm.srem %36, %37 : i32
        %39 = llvm.load %30 {alignment = 4 : i64} : !llvm.ptr -> i32
        %40 = llvm.add %39, %38 overflow<nsw> : i32
        llvm.store %40, %30 {alignment = 4 : i64} : i32, !llvm.ptr
        llvm.br ^bb4
      ^bb4:  // pred: ^bb3
        %41 = llvm.load %28 {alignment = 4 : i64} : !llvm.ptr -> i32
        %42 = llvm.mlir.constant(1 : i32) : i32
        %43 = llvm.add %41, %42 overflow<nsw> : i32
        llvm.store %43, %28 {alignment = 4 : i64} : i32, !llvm.ptr
        llvm.br ^bb2
      ^bb5:  // pred: ^bb2
        llvm.br ^bb6
      ^bb6:  // pred: ^bb5
        %44 = llvm.load %30 {alignment = 4 : i64} : !llvm.ptr -> i32
        %45 = llvm.load %13 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
        llvm.store %44, %45 {alignment = 4 : i64} : i32, !llvm.ptr
        %46 = llvm.mlir.constant(42 : i32) : i32
        %47 = llvm.load %9 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
        llvm.store %46, %47 {alignment = 4 : i64} : i32, !llvm.ptr
        omp.terminator
      }
      omp.taskwait
      %24 = llvm.load %9 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
      %25 = llvm.load %24 {alignment = 4 : i64} : !llvm.ptr -> i32
      %26 = llvm.load %11 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
      llvm.store %25, %26 {alignment = 4 : i64} : i32, !llvm.ptr
      omp.terminator
    }
    %18 = llvm.mlir.addressof @".str" : !llvm.ptr
    %19 = llvm.getelementptr %18[0] : (!llvm.ptr) -> !llvm.ptr, i8
    %20 = llvm.load %5 {alignment = 4 : i64} : !llvm.ptr -> i32
    %21 = llvm.call @printf(%19, %20) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
    %22 = llvm.mlir.constant(0 : i32) : i32
    llvm.store %22, %1 {alignment = 4 : i64} : i32, !llvm.ptr
    %23 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
    llvm.return %23 : i32
  }
}

