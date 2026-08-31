#!/bin/bash
# native.sh — the host compile pipelines. Sourced by common.sh.
#
#   compile_ref  the stock OpenMP compiler, one step
#   compile_opt  C -> ClangIR -> mlir-opt-omp -> LLVM IR -> object -> link
#
# Both take (src, outdir, binname, omp on|off) and leave the binary in outdir.

compile_opt() {
    local src="$1" outdir="$2" binname="$3" omp="${4:-on}"
    local name; name="$(basename "${src%.c}")"
    local tmpdir; tmpdir=$(mktemp -d)
    mkdir -p "$outdir"

    # OpenMP toggle: front-end flag, link flag and runtime libs follow it.
    local omp_cir="" link_omp="" link_libs="-lm"
    if [ "$omp" = "on" ]; then
        omp_cir="-fopenmp"; link_omp="$OPT_FOPENMP"; link_libs="$OPT_EXTRA_LIBS"
    fi

    echo "  compiling $name (opt, runtime=$RUNTIME, omp=$omp) ..."

    # 1) clang -> CIR
    "$CLANG" -S $CLANG_STRICT_FP $WARN_SUPPRESS \
        -Xclang -fclangir -Xclang -emit-cir $omp_cir \
        -I"$INC" -I"$(dirname "$src")" -I"$INC_OMP" \
        -D"$DATASET" $POLYBENCH_CFLAGS \
        "$src" -o "$tmpdir/$name.cir" || { rm -rf "$tmpdir"; return 1; }

    # 2) CIR -> LLVM dialect MLIR (strip residual cir.* attrs)
    "$CIR_OPT" "$tmpdir/$name.cir" --cir-to-llvm --reconcile-unrealized-casts \
        -o "$tmpdir/$name-s1.mlir" || { rm -rf "$tmpdir"; return 1; }
    sed -i -E 's/cir\.[^,}]+,? ?//g' "$tmpdir/$name-s1.mlir"

    # 3) custom OMP lowering
    "$MLIR_OPT_OMP" \
        --allow-unregistered-dialect \
        --omp-lower-dsl="$RULES" \
        --omp-lower-runtime="$RUNTIME" \
        ${BARRIER_ELIM_FLAG:+$BARRIER_ELIM_FLAG} \
        --omp-to-omp-lower --omp-outline --omp-lower-plan \
        "$tmpdir/$name-s1.mlir" > "$tmpdir/$name-s2.mlir" \
        || { rm -rf "$tmpdir"; return 1; }

    # 4) MLIR opt + lowering to LLVM dialect
    "$MLIR_OPT" "$tmpdir/$name-s2.mlir" \
        --canonicalize --cse --sccp --symbol-dce \
        --loop-invariant-code-motion --canonicalize --cse \
        --convert-arith-to-llvm --convert-func-to-llvm \
        --reconcile-unrealized-casts \
        -o "$tmpdir/$name-s3.mlir" || { rm -rf "$tmpdir"; return 1; }

    # 5) MLIR -> LLVM IR
    "$MLIR_TRANSLATE" "$tmpdir/$name-s3.mlir" --mlir-to-llvmir \
        > "$tmpdir/$name.ll" || { rm -rf "$tmpdir"; return 1; }

    # 6) opt -O3, 7) llc -> object
    "$OPT" -S -O3 "$tmpdir/$name.ll" > "$tmpdir/$name.opt.ll" \
        || { rm -rf "$tmpdir"; return 1; }
    "$LLC" -O3 -relocation-model=pic -filetype=obj "$tmpdir/$name.opt.ll" \
        -o "$tmpdir/$name.o" || { rm -rf "$tmpdir"; return 1; }

    # 8) polybench support object
    "$CLANG" -O3 $CLANG_STRICT_FP $WARN_SUPPRESS \
        -c "$INC/polybench.c" -I"$INC" \
        -D"$DATASET" $POLYBENCH_CFLAGS \
        -o "$tmpdir/polybench.o" || { rm -rf "$tmpdir"; return 1; }

    # 9) link — seq builds also pull in the serial OpenMP stubs so kernels that
    #    call omp_get_thread_num()/omp_get_num_threads() unconditionally resolve.
    local stub=""
    [ "$omp" = "off" ] && stub="$OMP_STUBS_SRC"
    "$CLANG" -O3 $CLANG_STRICT_FP \
        $link_omp -no-pie \
        "$tmpdir/$name.o" "$tmpdir/polybench.o" $stub \
        $link_libs $POLYBENCH_LFLAGS -o "$outdir/$binname" \
        || { rm -rf "$tmpdir"; return 1; }

    cp "$tmpdir/$name.opt.ll" "$outdir/$binname.ll"   # keep final IR for debugging
    rm -rf "$tmpdir"
    [ -f "$outdir/$binname" ]
}

# --- ref: stock OpenMP compiler --------------------------------------------
# compile_ref <src> <outdir> <binname> <omp:on|off>
compile_ref() {
    local src="$1" outdir="$2" binname="$3" omp="${4:-on}"
    mkdir -p "$outdir"

    local omp_flag="" stub=""
    if [ "$omp" = "on" ]; then
        omp_flag="-fopenmp"
    else
        stub="$OMP_STUBS_SRC"   # serial OpenMP stubs for the seq baseline
    fi

    echo "  compiling $(basename "${src%.c}") (ref, $REF_CC, omp=$omp) ..."

    "$REF_CC" -O3 $REF_FP $WARN_SUPPRESS $omp_flag \
        -I"$INC" -I"$(dirname "$src")" $REF_OMP_INC \
        -D"$DATASET" $POLYBENCH_CFLAGS \
        "$src" "$INC/polybench.c" $stub \
        -lm $POLYBENCH_LFLAGS -o "$outdir/$binname" || return 1
    [ -f "$outdir/$binname" ]
}
