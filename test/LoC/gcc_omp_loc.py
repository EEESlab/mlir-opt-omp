#!/usr/bin/env python3
"""Lines of GCC on the lowering path of omp parallel / for with
schedule(static|dynamic) / task / barrier.

Reproduces Table 1 of the paper's supplementary material: 21,761 attributed
lines out of 82,400, over 15 files. Which files and which entities count, and
why, is in README.md.

Usage:  gcc_omp_loc.py [--root PATH]      # GCC checkout, or $GCC_SRC
"""
import argparse
import os
import re
import subprocess
import sys

# GCC 17.0.0, the commit the line ranges refer to.
PIN = '113f406e521057894e4cd3af2355f814ad203e9a'

# Table 1, to flag drift.
PUBLISHED = {
    'gcc/omp-low.cc': 8014,
    'gcc/omp-expand.cc': 6033,
    'gcc/gimplify.cc': 4953,
    'gcc/gimple.h': 926,
    'gcc/omp-general.cc': 912,
    'gcc/cp/cp-gimplify.cc': 442,
    'gcc/gimple.cc': 98,
    'gcc/gimple.def': 86,
    'gcc/tree.def': 73,
    'gcc/omp-builtins.def': 52,
    'gcc/omp-general.h': 44,
    'gcc/tree-core.h': 41,
    'gcc/omp-expand.h': 32,
    'gcc/omp-low.h': 31,
    'gcc/tree.h': 24,
}

problems = []


def note(msg):
    problems.append(msg)
    print('  !! ' + msg)


def clean_lines(lines):
    """Blank comments and literals, so braces inside them do not count."""
    cleaned, in_block = [], False
    for raw in lines:
        res, i, n = [], 0, len(raw)
        while i < n:
            if in_block:
                j = raw.find('*/', i)
                if j == -1: res.append(' ' * (n - i)); i = n
                else: res.append(' ' * (j + 2 - i)); i = j + 2; in_block = False
            elif raw.startswith('//', i):
                res.append(' ' * (n - i)); i = n
            elif raw.startswith('/*', i):
                in_block = True; res.append('  '); i += 2
            elif raw[i] in ('"', "'"):
                q = raw[i]; res.append(q); i += 1
                while i < n:
                    if raw[i] == '\\' and i + 1 < n:
                        res.append('..'); i += 2; continue
                    if raw[i] == q:
                        res.append(q); i += 1; break
                    res.append(' '); i += 1
            else:
                res.append(raw[i]); i += 1
        cleaned.append(''.join(res))
    return cleaned


def extend_up(lines, start):
    """Walk back over the return type and the leading comment."""
    i = start
    while i > 0:
        prev = lines[i - 1].strip()
        if (prev and not prev.endswith((';', '}', '{', '*/'))
                and not prev.startswith('#')
                and not prev.startswith('/*')):
            i -= 1
        else:
            break
    j = i
    if j > 0 and not lines[j - 1].strip():
        j -= 1
    if j > 0 and lines[j - 1].strip().endswith('*/'):
        k = j - 1
        while k >= 0:
            if lines[k].lstrip().startswith('/*'):
                return k
            k -= 1
    return i


def find_end_body(cleaned, start):
    depth, opened = 0, False
    for i in range(start, len(cleaned)):
        for ch in cleaned[i]:
            if ch == '{': depth += 1; opened = True
            elif ch == '}': depth -= 1
        if opened and depth == 0:
            return i
    return len(cleaned) - 1


def process(root, path, entities):
    with open(os.path.join(root, path), encoding='utf-8', errors='replace') as f:
        lines = f.read().splitlines()
    cleaned = clean_lines(lines)
    rows, counted = [], set()
    for ent in entities:
        kind = ent[0]
        matches = []   # (label, start_line_idx)
        if kind == 'fn':
            name = ent[1]
            rx = re.compile(r'^' + re.escape(name) + r' ?\(')
            for idx, l in enumerate(lines):
                if rx.match(l):
                    matches.append((name, idx))
        elif kind == 'fnrx':
            rx = re.compile(r'^(' + ent[1] + r') ?\(')
            for idx, l in enumerate(lines):
                m = rx.match(l)
                if m:
                    matches.append((m.group(1), idx))
        elif kind == 'rxline':
            rx = re.compile(ent[2])
            for idx, l in enumerate(lines):
                if rx.search(l):
                    matches.append((ent[1], idx))
        elif kind == 'struct':
            name = ent[1]
            rx = re.compile(r'^(struct|class|enum)\s+(GTY\s*\(\([^)]*\)\)\s*)?'
                            + re.escape(name) + r'\b')
            for idx, l in enumerate(lines):
                if rx.match(l):
                    matches.append(('%s (struct)' % name, idx))
        elif kind == 'range':
            label, a, b = ent[1], ent[2], ent[3]
            for i in range(a - 1, b):
                counted.add(i)
            rows.append((label, a, b, b - a + 1))
            continue
        elif kind == 'file':
            for i in range(len(lines)):
                counted.add(i)
            rows.append((ent[1], 1, len(lines), len(lines)))
            continue
        if not matches:
            note('NOT FOUND: %r in %s' % (ent[1], path))
            continue
        for label, s0 in matches:
            start = extend_up(lines, s0)
            end = find_end_body(cleaned, s0)
            span = set(range(start, end + 1))
            if span & counted:
                note('OVERLAP: %s (L%d-L%d) in %s'
                     % (label, start + 1, end + 1, path))
            counted |= span
            rows.append((label, start + 1, end + 1, end - start + 1))
    return rows, len(counted), len(lines)


GOMP_FOR_STRUCT_RX = (r'^\s+(gimple_statement_omp|gomp_for|gomp_parallel|'
                      r'gomp_task|gomp_continue|gimple_statement_omp_parallel_layout|'
                      r'gimple_statement_omp_taskreg|gimple_statement_omp_return)\s*:')
IS_A_HELPER_RX = (r'^is_a_helper <(const )?(gomp_for|gomp_parallel|gomp_task|'
                  r'gomp_continue|gimple_statement_omp_taskreg|'
                  r'gimple_statement_omp_return) \*>::test')

FILES = {
 # gimplification
 'gcc/gimplify.cc': [
  ('struct', 'omp_region_type'),
  ('struct', 'gimplify_omp_ctx'),
  ('fn', 'new_omp_context'),
  ('fn', 'delete_omp_context'),
  ('fn', 'omp_firstprivatize_variable'),
  ('fn', 'omp_firstprivatize_type_sizes'),
  ('fn', 'omp_add_variable'),
  ('fn', 'omp_default_clause'),
  ('fn', 'omp_notice_variable'),
  ('fn', 'omp_is_private'),
  ('fn', 'omp_check_private'),
  ('fn', 'find_decl_expr'),
  ('fn', 'gimplify_scan_omp_clauses'),
  ('fn', 'omp_shared_to_firstprivate_optimizable_decl_p'),
  ('fn', 'omp_mark_stores'),
  ('fn', 'gimplify_adjust_omp_clauses_1'),
  ('fn', 'gimplify_adjust_omp_clauses'),
  ('fn', 'gimplify_omp_parallel'),
  ('fn', 'gimplify_omp_task'),
  ('fn', 'gimplify_omp_for'),
 ],
 'gcc/cp/cp-gimplify.cc': [
  ('fn', 'cp_gimplify_omp_for'),
  ('struct', 'cp_genericize_omp_taskreg'),
  ('fn', 'omp_var_to_track'),
  ('fn', 'omp_cxx_notice_variable'),
  ('fn', 'cxx_omp_clause_apply_fn'),
  ('fn', 'cxx_omp_clause_default_ctor'),
  ('fn', 'cxx_omp_clause_copy_ctor'),
  ('fn', 'cxx_omp_clause_assign_op'),
  ('fn', 'cxx_omp_clause_dtor'),
  ('fn', 'cxx_omp_privatize_by_reference'),
  ('fn', 'cxx_omp_const_qual_no_mutable'),
  ('fn', 'cxx_omp_predetermined_sharing_1'),
  ('fn', 'cxx_omp_predetermined_sharing'),
  ('fn', 'cxx_omp_finish_clause'),
  ('fn', 'cxx_omp_disregard_value_expr'),
 ],
 # lowering
 'gcc/omp-low.cc': [
  ('struct', 'omp_context'),
  ('fn', 'omp_member_access_dummy_var'),
  ('fn', 'unshare_and_remap_1'),
  ('fn', 'unshare_and_remap'),
  ('fn', 'scan_omp_op'),
  ('fn', 'is_parallel_ctx'),
  ('fn', 'is_task_ctx'),
  ('fn', 'is_taskreg_ctx'),
  ('fn', 'is_variable_sized'),
  ('fn', 'lookup_decl'),
  ('fn', 'maybe_lookup_decl'),
  ('fn', 'lookup_field'),
  ('fn', 'lookup_sfield'),
  ('fn', 'maybe_lookup_field'),
  ('fn', 'use_pointer_for_field'),
  ('fn', 'omp_copy_decl_2'),
  ('fn', 'omp_copy_decl_1'),
  ('fn', 'build_receiver_ref'),
  ('fn', 'build_outer_var_ref'),
  ('fn', 'build_sender_ref'),
  ('fn', 'install_var_field'),
  ('fn', 'install_var_local'),
  ('fn', 'fixup_remapped_decl'),
  ('fn', 'omp_copy_decl'),
  ('fn', 'new_omp_context'),
  ('fn', 'finalize_task_copyfn'),
  ('fn', 'delete_omp_context'),
  ('fn', 'fixup_child_record_type'),
  ('fn', 'scan_sharing_clauses'),
  ('fn', 'create_omp_child_function_name'),
  ('fn', 'omp_maybe_offloaded_ctx'),
  ('fn', 'create_omp_child_function'),
  ('fn', 'omp_find_combined_for'),
  ('fn', 'add_taskreg_looptemp_clauses'),
  ('fn', 'scan_omp_parallel'),
  ('fn', 'scan_omp_task'),
  ('fn', 'finish_taskreg_remap'),
  ('fn', 'finish_taskreg_scan'),
  ('fn', 'scan_omp_for'),
  ('fn', 'check_omp_nesting_restrictions'),
  ('fn', 'scan_omp_1_op'),
  ('fn', 'setjmp_or_longjmp_p'),
  ('fn', 'scan_omp_1_stmt'),
  ('fn', 'scan_omp'),
  ('fn', 'maybe_remove_omp_member_access_dummy_vars'),
  ('fn', 'remove_member_access_dummy_vars'),
  ('fn', 'maybe_lookup_ctx'),
  ('fn', 'lookup_decl_in_outer_ctx'),
  ('fn', 'maybe_lookup_decl_in_outer_ctx'),
  ('fn', 'lower_rec_input_clauses'),
  ('fn', 'lower_send_clauses'),
  ('fn', 'lower_send_shared_vars'),
  ('fn', 'maybe_add_implicit_barrier_cancel'),
  ('fn', 'lower_omp_for'),
  ('fn', 'check_combined_parallel'),
  ('struct', 'omp_taskcopy_context'),
  ('fn', 'task_copyfn_copy_decl'),
  ('fn', 'task_copyfn_remap_type'),
  ('fn', 'create_task_copyfn'),
  ('fn', 'lower_omp_taskreg'),
  ('fn', 'lower_omp_regimplify_p'),
  ('fn', 'lower_omp_regimplify_operands_p'),
  ('fn', 'lower_omp_regimplify_operands'),
  ('fn', 'lower_omp_1'),
  ('fn', 'lower_omp'),
  ('fn', 'execute_lower_omp'),
  ('rxline', 'pass_data_lower_omp', r'^const pass_data pass_data_lower_omp ='),
  ('struct', 'pass_lower_omp'),
  ('fn', 'make_pass_lower_omp'),
  ('fn', 'diagnose_sb_0'),
  ('fn', 'diagnose_sb_1'),
  ('fn', 'diagnose_sb_2'),
  ('fn', 'diagnose_omp_structured_block_errors'),
  ('rxline', 'pass_data_diagnose_omp_blocks',
   r'^const pass_data pass_data_diagnose_omp_blocks ='),
  ('struct', 'pass_diagnose_omp_blocks'),
  ('fn', 'make_pass_diagnose_omp_blocks'),
 ],
 # expansion
 'gcc/omp-expand.cc': [
  ('struct', 'omp_region'),
  ('fn', 'is_combined_parallel'),
  ('fn', 'is_in_offload_region'),
  ('fn', 'workshare_safe_to_combine_p'),
  ('fn', 'omp_adjust_chunk_size'),
  ('fn', 'get_ws_args_for'),
  ('fn', 'determine_parallel_type'),
  ('fn', 'new_omp_region'),
  ('fn', 'free_omp_region_1'),
  ('fn', 'omp_free_regions'),
  ('fn', 'gimple_build_cond_empty'),
  ('fn', 'adjust_context_and_scope'),
  ('fn', 'expand_parallel_call'),
  ('fn', 'expand_task_call'),
  ('fn', 'vec2chain'),
  ('fn', 'remove_exit_barrier'),
  ('fn', 'remove_exit_barriers'),
  ('fn', 'optimize_omp_library_calls'),
  ('fn', 'expand_omp_regimplify_p'),
  ('fn', 'expand_omp_build_assign'),
  ('fn', 'expand_omp_build_cond'),
  ('fn', 'expand_omp_taskreg'),
  ('fn', 'expand_omp_for_init_counts'),
  ('fn', 'expand_omp_for_init_vars'),
  ('fn', 'extract_omp_for_update_vars'),
  ('fn', 'expand_omp_for_generic'),
  ('fn', 'expand_omp_for_static_nochunk'),
  ('fn', 'find_phi_with_arg_on_edge'),
  ('fn', 'expand_omp_for_static_chunk'),
  ('fn', 'expand_omp_for'),
  ('fn', 'expand_omp'),
  ('fn', 'build_omp_regions_1'),
  ('fn', 'build_omp_regions'),
  ('fn', 'execute_expand_omp'),
  ('rxline', 'pass_data_expand_omp', r'^const pass_data pass_data_expand_omp ='),
  ('struct', 'pass_expand_omp'),
  ('fn', 'make_pass_expand_omp'),
  ('rxline', 'pass_data_expand_omp_ssa',
   r'^const pass_data pass_data_expand_omp_ssa ='),
  ('struct', 'pass_expand_omp_ssa'),
  ('fn', 'make_pass_expand_omp_ssa'),
  ('fn', 'omp_make_gimple_edges'),
 ],
 'gcc/omp-general.cc': [
  ('fn', 'omp_find_clause'),
  ('fn', 'omp_privatize_by_reference'),
  ('fn', 'omp_adjust_for_condition'),
  ('fn', 'omp_get_for_step_from_incr'),
  ('fn', 'omp_extract_for_data'),
  ('fn', 'omp_build_barrier'),
  ('fn', 'find_combined_omp_for'),
  ('fn', 'omp_build_component_ref'),
 ],
 'gcc/omp-general.h': [
  ('struct', 'omp_for_data_loop'),
  ('struct', 'omp_for_data'),
 ],
 'gcc/omp-low.h': [('file', 'whole header (declarations)')],
 'gcc/omp-expand.h': [('file', 'whole header (declarations)')],
 # IR: gimple and tree
 'gcc/gimple.h': [
  ('rxline', 'struct gimple_statement_omp/gomp_*', GOMP_FOR_STRUCT_RX),
  ('rxline', 'is_a_helper<gomp_*>::test', IS_A_HELPER_RX),
  ('fnrx', r'gimple_omp_(subcode|set_subcode|body|body_ptr|set_body|'
           r'return_nowait_p|return_set_nowait|return_lhs|return_lhs_ptr|'
           r'return_set_lhs|continue_\w+|for_\w+|parallel_\w+|taskreg_\w+|'
           r'task_\w+)'),
 ],
 'gcc/gimple.cc': [
  ('fn', 'gimple_build_omp_for'),
  ('fn', 'gimple_build_omp_parallel'),
  ('fn', 'gimple_build_omp_task'),
  ('fn', 'gimple_build_omp_continue'),
  ('fn', 'gimple_build_omp_return'),
 ],
 'gcc/gimple.def': [
  ('range', 'GIMPLE_OMP_CONTINUE', 226, 228),
  ('range', 'GIMPLE_OMP_FOR', 238, 276),
  ('range', 'GIMPLE_OMP_PARALLEL', 296, 312),
  ('range', 'GIMPLE_OMP_TASK', 314, 338),
  ('range', 'GIMPLE_OMP_RETURN', 340, 341),
 ],
 'gcc/tree.def': [
  ('range', 'OMP_PARALLEL', 1165, 1169),
  ('range', 'OMP_TASK', 1171, 1175),
  ('range', 'OMP_FOR', 1177, 1237),
  ('range', 'OMP_CLAUSE', 1422, 1423),
 ],
 'gcc/tree.h': [
  ('range', 'OMP_PARALLEL/TASK/TASKREG/FOR_* macros', 1578, 1595),
  ('range', 'OMP_CLAUSE_SCHEDULE_KIND/SIMD', 2130, 2135),
 ],
 'gcc/tree-core.h': [
  ('range', 'OMP_CLAUSE_SCHEDULE (enum omp_clause_code)', 446, 446),
  ('range', 'enum omp_clause_schedule_kind', 614, 624),
  ('rxline', 'struct tree_omp_clause', r'^struct GTY\(\(\)\) tree_omp_clause'),
 ],
 'gcc/omp-builtins.def': [
  ('range', 'omp_get_thread_num/num_threads', 80, 83),
  ('range', 'GOMP_barrier', 101, 102),
  ('range', 'GOMP_loop_{static,dynamic}_start (+ note)', 132, 142),
  ('range', 'GOMP_loop_nonmonotonic_dynamic_start', 151, 154),
  ('range', 'GOMP_loop_{static,dynamic}_next', 211, 214),
  ('range', 'GOMP_loop_nonmonotonic_dynamic_next', 219, 221),
  ('range', 'GOMP_parallel_loop_{static,dynamic} (+ note)', 355, 365),
  ('range', 'GOMP_parallel_loop_nonmonotonic_dynamic', 374, 377),
  ('range', 'GOMP_loop_end', 390, 391),
  ('range', 'GOMP_loop_end_nowait', 394, 395),
  ('range', 'GOMP_parallel', 411, 412),
  ('range', 'GOMP_task', 416, 418),
 ],
}


def check_pin(root):
    try:
        head = subprocess.run(['git', '-C', root, 'rev-parse', 'HEAD'],
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return
    if head and head != PIN:
        print('  !! checkout is at %s, the line ranges refer to %s\n'
              '     (git -C %s checkout %s)' % (head[:12], PIN[:12], root, PIN[:12]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--root', default=os.environ.get('GCC_SRC', '.'),
                    help='GCC checkout (default: $GCC_SRC, or .)')
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    if not os.path.isfile(os.path.join(root, 'gcc', 'omp-low.cc')):
        sys.exit('%s is not a GCC checkout (no gcc/omp-low.cc)' % root)
    print('gcc: %s' % root)
    check_pin(root)

    grand = grand_tot = 0
    summary = []
    for path, ents in FILES.items():
        rows, total, nlines = process(root, path, ents)
        grand += total; grand_tot += nlines
        summary.append((path, total, nlines))
        print('\n=== %s  (attributed: %d / total: %d) ===' % (path, total, nlines))
        for name, s, e, c in sorted(rows, key=lambda r: r[1]):
            print('  %6d  L%d-L%d  %s' % (c, s, e, name))
        if PUBLISHED.get(path) not in (None, total):
            note('%s: %d attributed, Table 1 reports %d' % (path, total, PUBLISHED[path]))

    print('\n%-38s %10s %10s' % ('FILE', 'attributed', 'total'))
    for path, total, nlines in sorted(summary, key=lambda x: -x[1]):
        print('%-38s %10d %10d' % (path, total, nlines))
    print('%-38s %10d %10d' % ('TOTAL', grand, grand_tot))
    if grand != sum(PUBLISHED.values()):
        print('\nTable 1 reports %d attributed lines.' % sum(PUBLISHED.values()))
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
