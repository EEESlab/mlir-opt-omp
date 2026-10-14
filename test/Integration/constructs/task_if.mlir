module @"task_if.c" attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
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
    %7 = llvm.alloca %6 x !llvm.ptr {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %8 = llvm.mlir.constant(1 : i64) : i64
    %9 = llvm.alloca %8 x !llvm.ptr {alignment = 8 : i64} : (i64) -> !llvm.ptr
    %10 = llvm.mlir.constant(0 : i32) : i32
    llvm.store %10, %3 {alignment = 4 : i64} : i32, !llvm.ptr
    %11 = llvm.mlir.constant(0 : i32) : i32
    llvm.store %11, %5 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.store %3, %7 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %5, %9 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    %12 = llvm.mlir.constant(1 : i32) : i32
    omp.parallel num_threads(%12 : i32) {
      %19 = llvm.mlir.constant(0 : i32) : i32
      %20 = llvm.mlir.constant(0 : i32) : i32
      %21 = llvm.icmp "ne" %19, %20 : i32
      omp.task if(%21) {
        %25 = llvm.mlir.constant(42 : i32) : i32
        %26 = llvm.load %7 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
        llvm.store %25, %26 {alignment = 4 : i64} : i32, !llvm.ptr
        omp.terminator
      }
      %22 = llvm.load %7 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
      %23 = llvm.load %22 {alignment = 4 : i64} : !llvm.ptr -> i32
      %24 = llvm.load %9 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
      llvm.store %23, %24 {alignment = 4 : i64} : i32, !llvm.ptr
      omp.terminator
    }
    %13 = llvm.mlir.addressof @".str" : !llvm.ptr
    %14 = llvm.getelementptr %13[0] : (!llvm.ptr) -> !llvm.ptr, i8
    %15 = llvm.load %5 {alignment = 4 : i64} : !llvm.ptr -> i32
    %16 = llvm.call @printf(%14, %15) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
    %17 = llvm.mlir.constant(0 : i32) : i32
    llvm.store %17, %1 {alignment = 4 : i64} : i32, !llvm.ptr
    %18 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
    llvm.return %18 : i32
  }
}

