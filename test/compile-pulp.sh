rm -f *.o  *.ll *.cir *.mlir

clang -S -Xclang -fclangir -Xclang -emit-cir -fopenmp  -I/usr/lib/gcc/x86_64-linux-gnu/12/include test.c -o test.cir
cir-opt test.cir --cir-to-llvm --reconcile-unrealized-casts -o test-s1.mlir
sed -i -E 's/cir\.[^,}]+,? ?//g' test-s1.mlir
../BUILD/mlir-opt-omp   --allow-unregistered-dialect   --omp-lower-dsl=../rules.dsl   --omp-lower-runtime=pmsis   --omp-to-omp-lower --omp-outline --omp-lower-plan  test-s1.mlir > test-s2.mlir
mlir-opt test-s2.mlir --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts -o test-s3.mlir
mlir-translate test-s3.mlir --mlir-to-llvmir > test-s4.ll
/home/tagliavini/toolchains/llvm-project/builds/INSTALL-REBASE/bin/llc -mtriple=riscv32-unknown-elf -mattr=+m,+c,+xpulpv -relocation-model=pic -filetype=obj test-s4.ll -o test.o

