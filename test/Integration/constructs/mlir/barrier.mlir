module @"barrier.c" attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  llvm.mlir.global external @data() {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<4 x i32> {
    %0 = llvm.mlir.zero : !llvm.array<4 x i32>
    llvm.return %0 : !llvm.array<4 x i32>
  }
  llvm.mlir.global external @bad() {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<256 x i32> {
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
    omp.parallel num_threads(%6 : i32) {
      %33 = llvm.mlir.constant(1 : i64) : i64
      %34 = llvm.alloca %33 x i32 {alignment = 4 : i64} : (i64) -> !llvm.ptr
      %35 = llvm.mlir.constant(1 : i64) : i64
      %36 = llvm.alloca %35 x i32 {alignment = 4 : i64} : (i64) -> !llvm.ptr
      %37 = llvm.mlir.constant(1 : i64) : i64
      %38 = llvm.alloca %37 x i32 {alignment = 4 : i64} : (i64) -> !llvm.ptr
      %39 = llvm.call @omp_get_thread_num() {no_unwind} : () -> i32
      llvm.store %39, %34 {alignment = 4 : i64} : i32, !llvm.ptr
      %40 = llvm.load %34 {alignment = 4 : i64} : !llvm.ptr -> i32
      %41 = llvm.mlir.constant(1 : i32) : i32
      %42 = llvm.add %40, %41 overflow<nsw> : i32
      %43 = llvm.load %34 {alignment = 4 : i64} : !llvm.ptr -> i32
      %44 = llvm.mlir.addressof @data : !llvm.ptr
      %45 = llvm.sext %43 : i32 to i64
      %46 = llvm.getelementptr %44[0, %45] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i32>
      llvm.store %42, %46 {alignment = 4 : i64} : i32, !llvm.ptr
      omp.barrier
      %47 = llvm.mlir.constant(0 : i32) : i32
      llvm.store %47, %38 {alignment = 4 : i64} : i32, !llvm.ptr
      llvm.br ^bb1
    ^bb1:  // pred: ^bb0
      %48 = llvm.mlir.constant(0 : i32) : i32
      llvm.store %48, %36 {alignment = 4 : i64} : i32, !llvm.ptr
      llvm.br ^bb2
    ^bb2:  // 2 preds: ^bb1, ^bb4
      %49 = llvm.load %36 {alignment = 4 : i64} : !llvm.ptr -> i32
      %50 = llvm.mlir.constant(4 : i32) : i32
      %51 = llvm.icmp "slt" %49, %50 : i32
      llvm.cond_br %51, ^bb3, ^bb5
    ^bb3:  // pred: ^bb2
      %52 = llvm.load %36 {alignment = 4 : i64} : !llvm.ptr -> i32
      %53 = llvm.mlir.addressof @data : !llvm.ptr
      %54 = llvm.sext %52 : i32 to i64
      %55 = llvm.getelementptr %53[0, %54] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i32>
      %56 = llvm.load %55 {alignment = 4 : i64} : !llvm.ptr -> i32
      %57 = llvm.load %38 {alignment = 4 : i64} : !llvm.ptr -> i32
      %58 = llvm.add %57, %56 overflow<nsw> : i32
      llvm.store %58, %38 {alignment = 4 : i64} : i32, !llvm.ptr
      llvm.br ^bb4
    ^bb4:  // pred: ^bb3
      %59 = llvm.load %36 {alignment = 4 : i64} : !llvm.ptr -> i32
      %60 = llvm.mlir.constant(1 : i32) : i32
      %61 = llvm.add %59, %60 overflow<nsw> : i32
      llvm.store %61, %36 {alignment = 4 : i64} : i32, !llvm.ptr
      llvm.br ^bb2
    ^bb5:  // pred: ^bb2
      llvm.br ^bb6
    ^bb6:  // pred: ^bb5
      llvm.br ^bb7
    ^bb7:  // pred: ^bb6
      %62 = llvm.load %38 {alignment = 4 : i64} : !llvm.ptr -> i32
      %63 = llvm.mlir.constant(4 : i32) : i32
      %64 = llvm.mlir.constant(4 : i32) : i32
      %65 = llvm.mlir.constant(1 : i32) : i32
      %66 = llvm.add %64, %65 overflow<nsw> : i32
      %67 = llvm.mul %63, %66 overflow<nsw> : i32
      %68 = llvm.mlir.constant(2 : i32) : i32
      %69 = llvm.sdiv %67, %68 : i32
      %70 = llvm.icmp "ne" %62, %69 : i32
      llvm.cond_br %70, ^bb8, ^bb9
    ^bb8:  // pred: ^bb7
      %71 = llvm.mlir.constant(1 : i32) : i32
      %72 = llvm.load %34 {alignment = 4 : i64} : !llvm.ptr -> i32
      %73 = llvm.mlir.addressof @bad : !llvm.ptr
      %74 = llvm.sext %72 : i32 to i64
      %75 = llvm.getelementptr %73[0, %74] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<256 x i32>
      llvm.store %71, %75 {alignment = 4 : i64} : i32, !llvm.ptr
      llvm.br ^bb9
    ^bb9:  // 2 preds: ^bb7, ^bb8
      llvm.br ^bb10
    ^bb10:  // pred: ^bb9
      omp.terminator
    }
    %7 = llvm.mlir.constant(0 : i32) : i32
    llvm.store %7, %5 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %8 = llvm.mlir.constant(0 : i32) : i32
    llvm.store %8, %3 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb1, ^bb4
    %9 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
    %10 = llvm.mlir.constant(256 : i32) : i32
    %11 = llvm.icmp "slt" %9, %10 : i32
    llvm.cond_br %11, ^bb3, ^bb5
  ^bb3:  // pred: ^bb2
    %12 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
    %13 = llvm.mlir.addressof @bad : !llvm.ptr
    %14 = llvm.sext %12 : i32 to i64
    %15 = llvm.getelementptr %13[0, %14] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<256 x i32>
    %16 = llvm.load %15 {alignment = 4 : i64} : !llvm.ptr -> i32
    %17 = llvm.load %5 {alignment = 4 : i64} : !llvm.ptr -> i32
    %18 = llvm.add %17, %16 overflow<nsw> : i32
    llvm.store %18, %5 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    %19 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
    %20 = llvm.mlir.constant(1 : i32) : i32
    %21 = llvm.add %19, %20 overflow<nsw> : i32
    llvm.store %21, %3 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb2
  ^bb5:  // pred: ^bb2
    llvm.br ^bb6
  ^bb6:  // pred: ^bb5
    %22 = llvm.mlir.addressof @".str" : !llvm.ptr
    %23 = llvm.getelementptr %22[0] : (!llvm.ptr) -> !llvm.ptr, i8
    %24 = llvm.load %5 {alignment = 4 : i64} : !llvm.ptr -> i32
    %25 = llvm.mlir.constant(0 : i32) : i32
    %26 = llvm.icmp "eq" %24, %25 : i32
    llvm.cond_br %26, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    %27 = llvm.mlir.constant(42 : i32) : i32
    llvm.br ^bb9(%27 : i32)
  ^bb8:  // pred: ^bb6
    %28 = llvm.load %5 {alignment = 4 : i64} : !llvm.ptr -> i32
    llvm.br ^bb9(%28 : i32)
  ^bb9(%29: i32):  // 2 preds: ^bb7, ^bb8
    llvm.br ^bb10
  ^bb10:  // pred: ^bb9
    %30 = llvm.call @printf(%23, %29) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
    %31 = llvm.mlir.constant(0 : i32) : i32
    llvm.store %31, %1 {alignment = 4 : i64} : i32, !llvm.ptr
    %32 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
    llvm.return %32 : i32
  }
}

