#!/usr/bin/env python3
"""Lines of Clang on the lowering path of omp parallel / for with
schedule(static|dynamic) / task / barrier.

Reproduces Table 2 of the paper's supplementary material: 7,617 attributed
lines out of 50,754, over 4 files. Which files and which entities count, and
why, is in README.md.

Usage:  clang_omp_loc.py [--root PATH]    # llvm-project checkout, or $LLVM_SRC
"""
import argparse
import os
import re
import subprocess
import sys

# LLVM 23.0.0git, the commit the anchors refer to.
PIN = '4f92cf9599c4077c08b7fac0a21624e55da572f9'

# Table 2, to flag drift.
PUBLISHED = {
    'clang/lib/Sema/SemaOpenMP.cpp': 3542,
    'clang/lib/CodeGen/CGOpenMPRuntime.cpp': 1936,
    'clang/lib/CodeGen/CGStmtOpenMP.cpp': 1832,
    'clang/lib/CodeGen/CGOpenMPRuntime.h': 307,
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


def anchor_ident(name):
    """The identifier to expect at the anchor line."""
    base = name.split(' (')[0].strip()
    if '::' in base:
        base = base.split('::')[-1]
    return base


SIG_RE = re.compile(r'^(static|virtual|inline|template|constexpr|const|unsigned|void|bool|class|struct|enum|llvm::|std::|clang::|Expr|Stmt|OMP|LValue|Address|StmtResult|ExprResult|DeclarationName|StringRef|QualType|int)\b')


def extend_up(lines, start):
    """Walk back over the return type and the leading comment."""
    i = start
    while i > 0:
        prev = lines[i - 1].strip()
        if prev and not prev.endswith((';', '}', '{', ':')) and SIG_RE.match(prev) \
           and not prev.startswith(('//', '/*', '*')):
            i -= 1
        else:
            break
    while i > 0:
        prev = lines[i - 1].strip()
        if prev.startswith(('//', '///')):
            i -= 1
        else:
            break
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


def find_end_decl(cleaned, start):
    for i in range(start, len(cleaned)):
        s = cleaned[i].rstrip()
        if s.endswith(';'):
            return i
    return len(cleaned) - 1


def process(root, path, entities):
    with open(os.path.join(root, path), encoding='utf-8', errors='replace') as f:
        lines = f.read().splitlines()
    cleaned = clean_lines(lines)
    rows, counted = [], set()
    for ent in entities:
        name, line1 = ent[0], ent[1]
        mode = ent[2] if len(ent) > 2 else 'body'
        s0 = line1 - 1
        # anchors are fixed line numbers: a miss means another checkout
        ident = anchor_ident(name)
        window = '\n'.join(lines[max(0, s0 - 1):s0 + 2])
        if ident not in window:
            hints = [str(i + 1) for i, l in enumerate(lines) if ident in l][:4]
            note('WRONG ANCHOR: %r not at L%d of %s (candidates: %s)'
                 % (ident, line1, path, ', '.join(hints) or 'none'))
        if mode == 'range':
            start, end = s0, ent[3] - 1
        else:
            start = extend_up(lines, s0)
            end = find_end_body(cleaned, s0) if mode == 'body' else find_end_decl(cleaned, s0)
        span = set(range(start, end + 1))
        if span & counted:
            note('OVERLAP: %s (L%d-L%d) in %s' % (name, start + 1, end + 1, path))
        counted |= span
        rows.append((name, start + 1, end + 1, end - start + 1))
    return rows, len(counted), len(lines)


FILES = {
 'clang/lib/Sema/SemaOpenMP.cpp': [
  ('ActOnOpenMPRegionStart', 4582), ('ActOnOpenMPRegionEnd', 4782),
  ('ActOnOpenMPExecutableDirective', 6202),
  ('ActOnOpenMPParallelDirective', 7891),
  ('OpenMPIterationSpaceChecker (class + methods)', 8002, 'range', 9331),
  ('ActOnOpenMPLoopInitialization', 9332),
  ('checkOpenMPIterationSpace', 9436),
  ('buildCounterUpdate', 9679), ('widenIterationCount', 9756),
  ('fitsInto', 9772), ('buildPreInits', 9782), ('buildPreInits (overload)', 9825),
  ('buildPostUpdate', 9836),
  ('checkOpenMPLoop', 9892),
  ('ActOnOpenMPForDirective', 10694),
  ('ActOnOpenMPTaskDirective', 11282),
  ('ActOnOpenMPBarrierDirective', 11307),
  ('ActOnOpenMPScheduleClause', 18046),
 ],
 'clang/lib/CodeGen/CGStmtOpenMP.cpp': [
  ('OMPLexicalScope', 56), ('OMPParallelScope', 117), ('OMPLoopScope', 148),
  ('GenerateOpenMPCapturedVars', 405), ('emitOutlinedFunctionPrologue', 507),
  ('GenerateOpenMPCapturedStmtFunction', 697),
  ('emitCommonOMPParallelDirective', 1639), ('emitEmptyBoundParameters', 1713),
  ('EmitOMPParallelDirective', 1849),
  ('EmitOMPLoopBody', 2016), ('EmitOMPInnerLoop', 2218), ('EmitOMPHelperVar', 2678),
  ('EmitOMPOuterLoop', 3056), ('EmitOMPForOuterLoop', 3173),
  ('EmitOMPWorksharingLoop', 3521),
  ('emitForLoopBounds', 3765), ('emitDispatchForLoopBounds', 3779),
  ('emitWorksharingDirective', 4055), ('convertClauseKindToSchedKind', 4132),
  ('emitOMPForDirective', 4153), ('EmitOMPForDirective', 4209),
  ('EmitOMPTaskBasedDirective', 4922), ('EmitOMPTaskDirective', 5576),
  ('EmitOMPBarrierDirective', 5624),
 ],
 'clang/lib/CodeGen/CGOpenMPRuntime.cpp': [
  ('CGOpenMPRegionInfo', 54), ('CGOpenMPOutlinedRegionInfo', 116),
  ('CGOpenMPTaskOutlinedRegionInfo', 149), ('InlinedOpenMPRegionRAII', 412),
  ('CGOpenMPRegionInfo::getThreadIDVariableLValue', 995),
  ('CGOpenMPTaskOutlinedRegionInfo::getThreadIDVariableLValue', 1016),
  ('emitParallelOrTeamsOutlinedFunction', 1216),
  ('emitParallelOutlinedFunction', 1267), ('emitTaskOutlinedFunction', 1287),
  ('emitUpdateLocation', 1376), ('getThreadID', 1405),
  ('emitParallelCall', 1973), ('emitThreadIDAddress', 2050),
  ('getDefaultFlagsForBarriers', 2422), ('getDefaultScheduleAndChunk', 2437),
  ('emitBarrierCall', 2455),
  ('getRuntimeSchedule', 2523), ('getRuntimeSchedule (dist overload)', 2546),
  ('addMonoNonMonoModifier', 2584),
  ('emitForDispatchInit', 2635), ('emitForDispatchDeinit', 2669),
  ('emitForStaticInitCall', 2678), ('emitForStaticInit', 2727),
  ('emitForStaticFinish', 2769), ('emitForNext', 2812),
  ('KmpTaskTFields (enum)', 2906),
  ('createKmpTaskTRecordDecl', 3054), ('createKmpTaskTWithPrivatesRecordDecl', 3100),
  ('emitProxyTaskFunction', 3128), ('emitDestructorsFunction', 3233),
  ('emitTaskPrivateMappingFunction', 3294), ('emitPrivatesInit', 3396),
  ('checkInitIsRequired', 3515), ('emitTaskDupFunction', 3543),
  ('checkDestructorsRequired', 3616),
  ('emitTaskInit', 3756), ('emitTaskCall', 4654),
 ],
 'clang/lib/CodeGen/CGOpenMPRuntime.h': [
  ('OMPTaskDataTy (struct)', 93),
  ('getThreadID (decl)', 341, 'decl'), ('emitThreadIDAddress (decl)', 357, 'decl'),
  ('getDefaultFlagsForBarriers (decl)', 373, 'decl'),
  ('TaskResultTy (struct)', 552), ('emitTaskInit (decl)', 582, 'decl'),
  ('emitUpdateLocation (decl)', 620, 'decl'),
  ('emitParallelOutlinedFunction (decl)', 728, 'decl'),
  ('emitTaskOutlinedFunction (decl)', 762, 'decl'),
  ('emitParallelCall (decl)', 791, 'decl'),
  ('emitBarrierCall (decl)', 866, 'decl'),
  ('DispatchRTInput (struct)', 908),
  ('emitForDispatchInit (decl)', 940, 'decl'),
  ('emitForDispatchDeinit (decl)', 951, 'decl'),
  ('StaticRTInput (struct)', 954),
  ('emitForStaticInit (decl)', 997, 'decl'),
  ('emitForStaticFinish (decl)', 1032, 'decl'),
  ('emitForNext (decl)', 1049, 'decl'),
  ('emitTaskCall (decl)', 1174, 'decl'),
  ('getDefaultScheduleAndChunk (decl)', 1577, 'decl'),
 ],
}


def check_pin(root):
    try:
        head = subprocess.run(['git', '-C', root, 'rev-parse', 'HEAD'],
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return
    if head and head != PIN:
        print('  !! checkout is at %s, the anchors refer to %s\n'
              '     (git -C %s checkout %s)' % (head[:12], PIN[:12], root, PIN[:12]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--root', default=os.environ.get('LLVM_SRC', '.'),
                    help='llvm-project checkout (default: $LLVM_SRC, or .)')
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    if not os.path.isdir(os.path.join(root, 'clang')):
        sys.exit('%s is not an llvm-project checkout (no clang/)' % root)
    print('llvm-project: %s' % root)
    check_pin(root)

    grand = grand_tot = 0
    summary = []
    for path, ents in FILES.items():
        rows, total, nlines = process(root, path, ents)
        grand += total; grand_tot += nlines
        summary.append((path, total, nlines))
        print('\n=== %s  (attributed: %d / total: %d) ===' % (path, total, nlines))
        for name, s, e, c in rows:
            print('  %6d  L%d-L%d  %s' % (c, s, e, name))
        if PUBLISHED.get(path) not in (None, total):
            note('%s: %d attributed, Table 2 reports %d' % (path, total, PUBLISHED[path]))

    print('\n%-38s %10s %10s' % ('FILE', 'attributed', 'total'))
    for path, total, nlines in sorted(summary, key=lambda x: -x[1]):
        print('%-38s %10d %10d' % (path, total, nlines))
    print('%-38s %10d %10d' % ('TOTAL', grand, grand_tot))
    if grand != sum(PUBLISHED.values()):
        print('\nTable 2 reports %d attributed lines.' % sum(PUBLISHED.values()))
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
