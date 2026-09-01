module @"nowait.c" attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  omp.private {type = private} @i.privatizer : i32 init {
  ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
    omp.yield(%arg1 : !llvm.ptr)
  }
  llvm.mlir.global external @a() {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<512 x i32> {
    %0 = llvm.mlir.zero : !llvm.array<512 x i32>
    llvm.return %0 : !llvm.array<512 x i32>
  }
  llvm.mlir.global external @b() {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<512 x i32> {
    %0 = llvm.mlir.zero : !llvm.array<512 x i32>
    llvm.return %0 : !llvm.array<512 x i32>
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
  llvm.func @printf(!llvm.ptr, ...) -> i32 attributes {sym_visibility = "private"}
  llvm.func @main() -> i32 attributes {dso_local, no_inline} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = llvm.alloca %2 x i32 {alignment = 4 : i64} : (i64) -> !llvm.ptr
    %4 = llvm.mlir.constant(1 : i64) : i64
    %5 = llvm.alloca %4 x i32 {alignment = 4 : i64} : (i64) -> !llvm.ptr
    %6 = llvm.mlir.constant(4 : i32) : i32
    omp.parallel num_threads(%6 : i32) private(@i.privatizer %3 -> %arg0 : !llvm.ptr) {
      %51 = llvm.mlir.constant(1 : i64) : i64
      %52 = llvm.alloca %51 x i32 {alignment = 4 : i64} : (i64) -> !llvm.ptr
      %53 = llvm.mlir.constant(1 : i64) : i64
      %54 = llvm.alloca %53 x i32 {alignment = 4 : i64} : (i64) -> !llvm.ptr
      %55 = llvm.mlir.constant(0 : i32) : i32
      %56 = llvm.mlir.constant(512 : i32) : i32
      %57 = llvm.mlir.constant(1 : i32) : i32
      llvm.store %55, %52 {alignment = 4 : i64} : i32, !llvm.ptr
      omp.wsloop nowait {
        omp.loop_nest (%arg1) : i32 = (%55) to (%56) step (%57) {
          llvm.store %arg1, %52 {alignment = 4 : i64} : i32, !llvm.ptr
          %61 = llvm.load %52 {alignment = 4 : i64} : !llvm.ptr -> i32
          %62 = llvm.mlir.constant(1 : i32) : i32
          %63 = llvm.add %61, %62 overflow<nsw> : i32
          %64 = llvm.load %52 {alignment = 4 : i64} : !llvm.ptr -> i32
          %65 = llvm.mlir.addressof @a : !llvm.ptr
          %66 = llvm.sext %64 : i32 to i64
          %67 = llvm.getelementptr %65[0, %66] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<512 x i32>
          llvm.store %63, %67 {alignment = 4 : i64} : i32, !llvm.ptr
          omp.yield
        }
      }
      omp.barrier
      %58 = llvm.mlir.constant(0 : i32) : i32
      %59 = llvm.mlir.constant(512 : i32) : i32
      %60 = llvm.mlir.constant(1 : i32) : i32
      llvm.store %58, %54 {alignment = 4 : i64} : i32, !llvm.ptr
      omp.wsloop {
        omp.loop_nest (%arg1) : i32 = (%58) to (%59) step (%60) {
          llvm.store %arg1, %54 {alignment = 4 : i64} : i32, !llvm.ptr
          %61 = llvm.load %54 {alignment = 4 : i64} : !llvm.ptr -> i32
          %62 = llvm.mlir.addressof @a : !llvm.ptr
          %63 = llvm.sext %61 : i32 to i64
          %64 = llvm.getelementptr %62[0, %63] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<512 x i32>
          %65 = llvm.load %64 {alignment = 4 : i64} : !llvm.ptr -> i32
          %66 = llvm.mlir.constant(2 : i32) : i32
          %67 = llvm.mul %65, %66 overflow<nsw> : i32
          %68 = llvm.load %54 {alignment = 4 : i64} : !llvm.ptr -> i32
          %69 = llvm.mlir.addressof @b : !llvm.ptr
          %70 = llvm.sext %68 : i32 to i64
          %71 = llvm.getelementptr %69[0, %70] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<512 x i32>
          llvm.store %67, %71 {alignment = 4 : i64} : i32, !llvm.ptr
          omp.yield
        }
      }
      omp.terminator
    }
    %7 = llvm.mlir.constant(0 : i32) : i32
    llvm.store %7, %5 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %8 = llvm.mlir.constant(0 : i32) : i32
    llvm.store %8, %3 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb1, ^bb12
    %9 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
    %10 = llvm.mlir.constant(512 : i32) : i32
    %11 = llvm.icmp "slt" %9, %10 : i32
    llvm.cond_br %11, ^bb3, ^bb13
  ^bb3:  // pred: ^bb2
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    %12 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
    %13 = llvm.mlir.addressof @a : !llvm.ptr
    %14 = llvm.sext %12 : i32 to i64
    %15 = llvm.getelementptr %13[0, %14] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<512 x i32>
    %16 = llvm.load %15 {alignment = 4 : i64} : !llvm.ptr -> i32
    %17 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
    %18 = llvm.mlir.constant(1 : i32) : i32
    %19 = llvm.add %17, %18 overflow<nsw> : i32
    %20 = llvm.icmp "ne" %16, %19 : i32
    llvm.cond_br %20, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %21 = llvm.mlir.constant(true) : i1
    llvm.br ^bb7(%21 : i1)
  ^bb6:  // pred: ^bb4
    %22 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
    %23 = llvm.mlir.addressof @b : !llvm.ptr
    %24 = llvm.sext %22 : i32 to i64
    %25 = llvm.getelementptr %23[0, %24] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<512 x i32>
    %26 = llvm.load %25 {alignment = 4 : i64} : !llvm.ptr -> i32
    %27 = llvm.mlir.constant(2 : i32) : i32
    %28 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
    %29 = llvm.mlir.constant(1 : i32) : i32
    %30 = llvm.add %28, %29 overflow<nsw> : i32
    %31 = llvm.mul %27, %30 overflow<nsw> : i32
    %32 = llvm.icmp "ne" %26, %31 : i32
    llvm.br ^bb7(%32 : i1)
  ^bb7(%33: i1):  // 2 preds: ^bb5, ^bb6
    llvm.br ^bb8
  ^bb8:  // pred: ^bb7
    llvm.cond_br %33, ^bb9, ^bb10
  ^bb9:  // pred: ^bb8
    %34 = llvm.load %5 {alignment = 4 : i64} : !llvm.ptr -> i32
    %35 = llvm.mlir.constant(1 : i32) : i32
    %36 = llvm.add %34, %35 overflow<nsw> : i32
    llvm.store %36, %5 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb10
  ^bb10:  // 2 preds: ^bb8, ^bb9
    llvm.br ^bb11
  ^bb11:  // pred: ^bb10
    llvm.br ^bb12
  ^bb12:  // pred: ^bb11
    %37 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
    %38 = llvm.mlir.constant(1 : i32) : i32
    %39 = llvm.add %37, %38 overflow<nsw> : i32
    llvm.store %39, %3 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb2
  ^bb13:  // pred: ^bb2
    llvm.br ^bb14
  ^bb14:  // pred: ^bb13
    %40 = llvm.mlir.addressof @".str" : !llvm.ptr
    %41 = llvm.getelementptr %40[0] : (!llvm.ptr) -> !llvm.ptr, i8
    %42 = llvm.load %5 {alignment = 4 : i64} : !llvm.ptr -> i32
    %43 = llvm.mlir.constant(0 : i32) : i32
    %44 = llvm.icmp "eq" %42, %43 : i32
    llvm.cond_br %44, ^bb15, ^bb16
  ^bb15:  // pred: ^bb14
    %45 = llvm.mlir.constant(42 : i32) : i32
    llvm.br ^bb17(%45 : i32)
  ^bb16:  // pred: ^bb14
    %46 = llvm.load %5 {alignment = 4 : i64} : !llvm.ptr -> i32
    llvm.br ^bb17(%46 : i32)
  ^bb17(%47: i32):  // 2 preds: ^bb15, ^bb16
    llvm.br ^bb18
  ^bb18:  // pred: ^bb17
    %48 = llvm.call @printf(%41, %47) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
    %49 = llvm.mlir.constant(0 : i32) : i32
    llvm.store %49, %1 {alignment = 4 : i64} : i32, !llvm.ptr
    %50 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
    llvm.return %50 : i32
  }
}

