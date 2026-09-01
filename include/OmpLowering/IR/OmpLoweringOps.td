// TableGen definition of the omp_lower dialect, its plan attributes and
// omp_lower.construct.

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// --- Dialect ---

def OmpLowering_Dialect : Dialect {
  let name         = "omp_lower";
  let summary      = "Dialect for OpenMP runtime lowering plans";
  let cppNamespace = "::mlir::omp_lower";
  let useDefaultAttributePrinterParser = 1;
}

// --- Attribute base class ---

class OmpLowering_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<OmpLowering_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

// --- PlanCallAttr  #omp_lower.call<"callee", ["arg0", "arg1"]> ---

def PlanCallAttr : OmpLowering_Attr<"PlanCall", "call"> {
  let summary = "A single planned runtime call";
  let parameters = (ins
    "::mlir::StringAttr":$callee,
    "::mlir::ArrayAttr":$args,
    // Non-null when the call's SSA result is bound (`let <name> = call ...`).
    OptionalParameter<"::mlir::StringAttr">:$result
  );
  let assemblyFormat = "`<` $callee `,` `[` $args `]` (`,` `result` `=` $result^)? `>`";
}

// --- PlanBranchAttr  #omp_lower.branch<"%cond", [...], [...]> ---

def PlanBranchAttr : OmpLowering_Attr<"PlanBranch", "branch"> {
  let summary = "A planned branch on a run-time value";
  let description = [{
    The `when`/`otherwise` chains in the rule file are decided while the rules
    are evaluated and collapse into a flat sequence.  This one does not: its
    condition names a value known only at run time, so both arms reach the
    emitting pass and become real control flow.  Either arm may be empty.
  }];
  let parameters = (ins
    // Names the SSA value to branch on, resolved like any other plan argument.
    "::mlir::Attribute":$cond,
    "::mlir::ArrayAttr":$ifTrue,
    "::mlir::ArrayAttr":$ifFalse
  );
  let assemblyFormat = "`<` $cond `,` `[` $ifTrue `]` `,` `[` $ifFalse `]` `>`";
}

// --- PlanEmitAttr  #omp_lower.emit<"name", value> ---

def PlanEmitAttr : OmpLowering_Attr<"PlanEmit", "emit"> {
  let summary = "A named value to materialise before the construct body";
  let parameters = (ins
    "::mlir::StringAttr":$sym_name,
    "::mlir::Attribute":$value
  );
  let assemblyFormat = "`<` $sym_name `,` $value `>`";
}

// --- omp_lower.construct ---

def OmpLowering_ConstructOp : Op<OmpLowering_Dialect, "construct",
    [NoMemoryEffect]> {

  let summary = "Encodes a fully-evaluated LoweringPlan as MLIR attributes";

  let arguments = (ins
    // Clause operands, carried as real SSA operands so the values from the
    // original omp.* op survive into OmpOutliningPass.  One variadic list
    // rather than several Optional operands avoids AttrSizedOperandSegments;
    // the 1:1 clause_names array below says which clause each one carries, so a
    // construct can hold several at once.
    Variadic<AnyType>:$clause_operands,

    StrAttr:$runtime,
    // 'construct' is a reserved TableGen keyword, hence construct_name; the
    // assembly format still prints 'construct ='.
    StrAttr:$construct_name,
    DictionaryAttr:$prop_dict,
    ArrayAttr:$pre,
    ArrayAttr:$invoke,
    ArrayAttr:$post,
    // Clause name of each clause_operands entry, same order.  Optional so
    // barrier and taskwait stay unchanged.
    OptionalAttr<StrArrayAttr>:$clause_names,
    // Names in clause_names standing for a *list*, so every operand under that
    // name belongs to it (captures).  Recorded separately because a list can be
    // empty, and only this attribute tells "empty list" from "not a list".
    OptionalAttr<StrArrayAttr>:$list_names,
    // proc_bind is a compile-time enum, so unlike num_threads and if_clause it
    // has no SSA value to ride in clause_operands.  The kind's spelling travels
    // here and the outlining pass turns it into the runtime's own numbering,
    // which differs from the MLIR enum's ordinals.
    OptionalAttr<StrAttr>:$proc_bind
  );

  let results = (outs);

  // Optional region holding the body before outlining; empty afterwards.
  let regions = (region AnyRegion:$body);

  let assemblyFormat = [{
    `runtime` `=` $runtime `,`
    `construct` `=` $construct_name `,`
    `prop_dict` `=` $prop_dict `,`
    `pre` `=` $pre `,`
    `invoke` `=` $invoke `,`
    `post` `=` $post
    (`,` `clause_operands` `=` `(` $clause_operands^ `:` type($clause_operands) `)`)?
    $body
    attr-dict
  }];

  let hasVerifier = 1;
}
