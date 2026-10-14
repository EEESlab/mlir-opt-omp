module @"schedule_dynamic.c" attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  omp.private {type = private} @i.privatizer : i32 init {
  ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
    omp.yield(%arg1 : !llvm.ptr)
  }
  llvm.mlir.global external @hits() {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<1001 x i32> {
    %0 = llvm.mlir.zero : !llvm.array<1001 x i32>
    llvm.return %0 : !llvm.array<1001 x i32>
  }
  llvm.mlir.global private constant @".str"() {addr_space = 0 : i32, alignment = 1 : i64, dso_local} : !llvm.array<4 x i8> {
    %0 = llvm.mlir.undef : !llvm.array<4 x i8>
    %1 = llvm.mlir.constant(52 : i8) : i8
    %2 = llvm.insertvalue %1, %0[0] : !llvm.array<4 x i8> 
    %3 = llvm.mlir.constant(50 : i8) : i8
    %4 = llvm.insertvalue %3, %2[1] : !llvm.array<4 x i8> 
    %5 = llvm.mlir.constant(10 : i8) : i8
    %6 = llvm.insertvalue %5, %4[2] : !llvm.array<4 x i8> 
    %7 = llvm.mlir.constant(0 : i8) : i8
    %8 = llvm.insertvalue %7, %6[3] : !llvm.array<4 x i8> 
    llvm.return %8 : !llvm.array<4 x i8>
  }
  llvm.mlir.global private constant @".str.1"() {addr_space = 0 : i32, alignment = 1 : i64, dso_local} : !llvm.array<20 x i8> {
    %0 = llvm.mlir.undef : !llvm.array<20 x i8>
    %1 = llvm.mlir.constant(109 : i8) : i8
    %2 = llvm.insertvalue %1, %0[0] : !llvm.array<20 x i8> 
    %3 = llvm.mlir.constant(105 : i8) : i8
    %4 = llvm.insertvalue %3, %2[1] : !llvm.array<20 x i8> 
    %5 = llvm.mlir.constant(115 : i8) : i8
    %6 = llvm.insertvalue %5, %4[2] : !llvm.array<20 x i8> 
    %7 = llvm.mlir.constant(115 : i8) : i8
    %8 = llvm.insertvalue %7, %6[3] : !llvm.array<20 x i8> 
    %9 = llvm.mlir.constant(101 : i8) : i8
    %10 = llvm.insertvalue %9, %8[4] : !llvm.array<20 x i8> 
    %11 = llvm.mlir.constant(100 : i8) : i8
    %12 = llvm.insertvalue %11, %10[5] : !llvm.array<20 x i8> 
    %13 = llvm.mlir.constant(61 : i8) : i8
    %14 = llvm.insertvalue %13, %12[6] : !llvm.array<20 x i8> 
    %15 = llvm.mlir.constant(37 : i8) : i8
    %16 = llvm.insertvalue %15, %14[7] : !llvm.array<20 x i8> 
    %17 = llvm.mlir.constant(100 : i8) : i8
    %18 = llvm.insertvalue %17, %16[8] : !llvm.array<20 x i8> 
    %19 = llvm.mlir.constant(32 : i8) : i8
    %20 = llvm.insertvalue %19, %18[9] : !llvm.array<20 x i8> 
    %21 = llvm.mlir.constant(116 : i8) : i8
    %22 = llvm.insertvalue %21, %20[10] : !llvm.array<20 x i8> 
    %23 = llvm.mlir.constant(119 : i8) : i8
    %24 = llvm.insertvalue %23, %22[11] : !llvm.array<20 x i8> 
    %25 = llvm.mlir.constant(105 : i8) : i8
    %26 = llvm.insertvalue %25, %24[12] : !llvm.array<20 x i8> 
    %27 = llvm.mlir.constant(99 : i8) : i8
    %28 = llvm.insertvalue %27, %26[13] : !llvm.array<20 x i8> 
    %29 = llvm.mlir.constant(101 : i8) : i8
    %30 = llvm.insertvalue %29, %28[14] : !llvm.array<20 x i8> 
    %31 = llvm.mlir.constant(61 : i8) : i8
    %32 = llvm.insertvalue %31, %30[15] : !llvm.array<20 x i8> 
    %33 = llvm.mlir.constant(37 : i8) : i8
    %34 = llvm.insertvalue %33, %32[16] : !llvm.array<20 x i8> 
    %35 = llvm.mlir.constant(100 : i8) : i8
    %36 = llvm.insertvalue %35, %34[17] : !llvm.array<20 x i8> 
    %37 = llvm.mlir.constant(10 : i8) : i8
    %38 = llvm.insertvalue %37, %36[18] : !llvm.array<20 x i8> 
    %39 = llvm.mlir.constant(0 : i8) : i8
    %40 = llvm.insertvalue %39, %38[19] : !llvm.array<20 x i8> 
    llvm.return %40 : !llvm.array<20 x i8>
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
    %8 = llvm.mlir.constant(4 : i32) : i32
    omp.parallel num_threads(%8 : i32) private(@i.privatizer %3 -> %arg0 : !llvm.ptr) {
      %56 = llvm.mlir.constant(1 : i64) : i64
      %57 = llvm.alloca %56 x i32 {alignment = 4 : i64} : (i64) -> !llvm.ptr
      %58 = llvm.mlir.constant(0 : i32) : i32
      %59 = llvm.mlir.constant(1001 : i32) : i32
      %60 = llvm.mlir.constant(1 : i32) : i32
      llvm.store %58, %57 {alignment = 4 : i64} : i32, !llvm.ptr
      omp.wsloop schedule(dynamic) {
        omp.loop_nest (%arg1) : i32 = (%58) to (%59) step (%60) {
          llvm.store %arg1, %57 {alignment = 4 : i64} : i32, !llvm.ptr
          %61 = llvm.load %57 {alignment = 4 : i64} : !llvm.ptr -> i32
          %62 = llvm.mlir.addressof @hits : !llvm.ptr
          %63 = llvm.sext %61 : i32 to i64
          %64 = llvm.getelementptr %62[0, %63] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<1001 x i32>
          %65 = llvm.load %64 {alignment = 4 : i64} : !llvm.ptr -> i32
          %66 = llvm.mlir.constant(1 : i32) : i32
          %67 = llvm.add %65, %66 overflow<nsw> : i32
          %68 = llvm.load %57 {alignment = 4 : i64} : !llvm.ptr -> i32
          %69 = llvm.mlir.addressof @hits : !llvm.ptr
          %70 = llvm.sext %68 : i32 to i64
          %71 = llvm.getelementptr %69[0, %70] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<1001 x i32>
          llvm.store %67, %71 {alignment = 4 : i64} : i32, !llvm.ptr
          omp.yield
        }
      }
      omp.terminator
    }
    %9 = llvm.mlir.constant(0 : i32) : i32
    llvm.store %9, %5 {alignment = 4 : i64} : i32, !llvm.ptr
    %10 = llvm.mlir.constant(0 : i32) : i32
    llvm.store %10, %7 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %11 = llvm.mlir.constant(0 : i32) : i32
    llvm.store %11, %3 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb1, ^bb15
    %12 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
    %13 = llvm.mlir.constant(1001 : i32) : i32
    %14 = llvm.icmp "slt" %12, %13 : i32
    llvm.cond_br %14, ^bb3, ^bb16
  ^bb3:  // pred: ^bb2
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5
  ^bb5:  // pred: ^bb4
    %15 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
    %16 = llvm.mlir.addressof @hits : !llvm.ptr
    %17 = llvm.sext %15 : i32 to i64
    %18 = llvm.getelementptr %16[0, %17] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<1001 x i32>
    %19 = llvm.load %18 {alignment = 4 : i64} : !llvm.ptr -> i32
    %20 = llvm.mlir.constant(0 : i32) : i32
    %21 = llvm.icmp "eq" %19, %20 : i32
    llvm.cond_br %21, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %22 = llvm.load %5 {alignment = 4 : i64} : !llvm.ptr -> i32
    %23 = llvm.mlir.constant(1 : i32) : i32
    %24 = llvm.add %22, %23 overflow<nsw> : i32
    llvm.store %24, %5 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb12
  ^bb7:  // pred: ^bb5
    llvm.br ^bb8
  ^bb8:  // pred: ^bb7
    %25 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
    %26 = llvm.mlir.addressof @hits : !llvm.ptr
    %27 = llvm.sext %25 : i32 to i64
    %28 = llvm.getelementptr %26[0, %27] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<1001 x i32>
    %29 = llvm.load %28 {alignment = 4 : i64} : !llvm.ptr -> i32
    %30 = llvm.mlir.constant(1 : i32) : i32
    %31 = llvm.icmp "sgt" %29, %30 : i32
    llvm.cond_br %31, ^bb9, ^bb10
  ^bb9:  // pred: ^bb8
    %32 = llvm.load %7 {alignment = 4 : i64} : !llvm.ptr -> i32
    %33 = llvm.mlir.constant(1 : i32) : i32
    %34 = llvm.add %32, %33 overflow<nsw> : i32
    llvm.store %34, %7 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb10
  ^bb10:  // 2 preds: ^bb8, ^bb9
    llvm.br ^bb11
  ^bb11:  // pred: ^bb10
    llvm.br ^bb12
  ^bb12:  // 2 preds: ^bb6, ^bb11
    llvm.br ^bb13
  ^bb13:  // pred: ^bb12
    llvm.br ^bb14
  ^bb14:  // pred: ^bb13
    llvm.br ^bb15
  ^bb15:  // pred: ^bb14
    %35 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
    %36 = llvm.mlir.constant(1 : i32) : i32
    %37 = llvm.add %35, %36 overflow<nsw> : i32
    llvm.store %37, %3 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb2
  ^bb16:  // pred: ^bb2
    llvm.br ^bb17
  ^bb17:  // pred: ^bb16
    llvm.br ^bb18
  ^bb18:  // pred: ^bb17
    %38 = llvm.load %5 {alignment = 4 : i64} : !llvm.ptr -> i32
    %39 = llvm.mlir.constant(0 : i32) : i32
    %40 = llvm.icmp "eq" %38, %39 : i32
    llvm.cond_br %40, ^bb19, ^bb20
  ^bb19:  // pred: ^bb18
    %41 = llvm.load %7 {alignment = 4 : i64} : !llvm.ptr -> i32
    %42 = llvm.mlir.constant(0 : i32) : i32
    %43 = llvm.icmp "eq" %41, %42 : i32
    llvm.br ^bb21(%43 : i1)
  ^bb20:  // pred: ^bb18
    %44 = llvm.mlir.constant(false) : i1
    llvm.br ^bb21(%44 : i1)
  ^bb21(%45: i1):  // 2 preds: ^bb19, ^bb20
    llvm.br ^bb22
  ^bb22:  // pred: ^bb21
    llvm.cond_br %45, ^bb23, ^bb24
  ^bb23:  // pred: ^bb22
    %46 = llvm.mlir.addressof @".str" : !llvm.ptr
    %47 = llvm.getelementptr %46[0] : (!llvm.ptr) -> !llvm.ptr, i8
    %48 = llvm.call @printf(%47) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    llvm.br ^bb25
  ^bb24:  // pred: ^bb22
    %49 = llvm.mlir.addressof @".str.1" : !llvm.ptr
    %50 = llvm.getelementptr %49[0] : (!llvm.ptr) -> !llvm.ptr, i8
    %51 = llvm.load %5 {alignment = 4 : i64} : !llvm.ptr -> i32
    %52 = llvm.load %7 {alignment = 4 : i64} : !llvm.ptr -> i32
    %53 = llvm.call @printf(%50, %51, %52) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32, i32) -> i32
    llvm.br ^bb25
  ^bb25:  // 2 preds: ^bb23, ^bb24
    llvm.br ^bb26
  ^bb26:  // pred: ^bb25
    %54 = llvm.mlir.constant(0 : i32) : i32
    llvm.store %54, %1 {alignment = 4 : i64} : i32, !llvm.ptr
    %55 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
    llvm.return %55 : i32
  }
}

