// OmpLoweringOps.td
//
// TableGen dialect definition for the omp_lower dialect.
// mlir-tblgen generates:
//   OmpLoweringOps.h.inc           – op declarations
//   OmpLoweringOps.cpp.inc         – op definitions
//   OmpLoweringOps.dialect.cpp.inc – dialect registration
//   OmpLoweringAttrs.h.inc         – attribute declarations (via -gen-attrdef-decls)
//   OmpLoweringAttrs.cpp.inc       – attribute definitions  (via -gen-attrdef-defs)

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// ---------------------------------------------------------------------------
// Dialect
// ---------------------------------------------------------------------------

def OmpLowering_Dialect : Dialect {
  let name         = "omp_lower";
  let summary      = "Dialect for OpenMP runtime lowering plans";
  let cppNamespace = "::mlir::omp_lower";
  let useDefaultAttributePrinterParser = 1;
  // Disable the LLVM-17+ "Properties" storage model.
  // With usePropertiesForAttributes = 1 (the new default) TableGen generates
  // code that depends on BytecodeOpInterface, InferredProperties, and a new
  // getProperties() ABI that requires additional boilerplate we don't need.
  // Setting this to 0 keeps all op attributes in the classic attribute-dict,
  // which is simpler and fully sufficient for a plan-carrier operation.
  let usePropertiesForAttributes = 0;
}

// ---------------------------------------------------------------------------
// Attribute base class
// ---------------------------------------------------------------------------

class OmpLowering_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<OmpLowering_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

// ---------------------------------------------------------------------------
// PlanCallAttr  –  #omp_lower.call<"callee", ["arg0", "arg1"]>
// ---------------------------------------------------------------------------

def PlanCallAttr : OmpLowering_Attr<"PlanCall", "call"> {
  let summary = "A single planned runtime call";
  let parameters = (ins
    "::mlir::StringAttr":$callee,
    "::mlir::ArrayAttr":$args,
    // Non-null when the call's SSA result is bound (`let <name> = call ...`);
    // holds <name> so the outlining pass can thread the result to later calls.
    OptionalParameter<"::mlir::StringAttr">:$result
  );
  let assemblyFormat = "`<` $callee `,` `[` $args `]` (`,` `result` `=` $result^)? `>`";
}

// ---------------------------------------------------------------------------
// PlanBranchAttr  –  #omp_lower.branch<"%cond", [...], [...]>
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// PlanEmitAttr  –  #omp_lower.emit<"name", value>
// ---------------------------------------------------------------------------

def PlanEmitAttr : OmpLowering_Attr<"PlanEmit", "emit"> {
  let summary = "A named value to materialise before the construct body";
  let parameters = (ins
    "::mlir::StringAttr":$sym_name,
    "::mlir::Attribute":$value
  );
  let assemblyFormat = "`<` $sym_name `,` $value `>`";
}

// ---------------------------------------------------------------------------
// omp_lower.construct
// ---------------------------------------------------------------------------

def OmpLowering_ConstructOp : Op<OmpLowering_Dialect, "construct",
    [NoMemoryEffect]> {

  let summary = "Encodes a fully-evaluated LoweringPlan as MLIR attributes";

  let arguments = (ins
    // Clause operands. Carried as real SSA operands so the values from the
    // original omp.* op survive into OmpOutliningPass — looking them up by
    // symbolic name in the plan would otherwise lose the SSA reference.
    //
    // A single variadic list (rather than several Optional operands) avoids
    // AttrSizedOperandSegments, whose generated verifier is broken when the
    // dialect disables usePropertiesForAttributes.  The 1:1 clause_names
    // array below says which clause each operand carries ("num_threads",
    // "if_clause", ...), so a construct can carry several at once (e.g.
    // `parallel num_threads(n) if(c)`).
    Variadic<AnyType>:$clause_operands,

    StrAttr:$runtime,
    // NOTE: 'construct' is a reserved TableGen keyword so the accessor is
    // named construct_name here; the assembly format still prints 'construct ='.
    StrAttr:$construct_name,
    DictionaryAttr:$prop_dict,
    ArrayAttr:$pre,
    ArrayAttr:$invoke,
    ArrayAttr:$post,
    // Clause name of each clause_operands entry, same order.  Optional so
    // constructs without clause operands (barrier, taskwait) stay unchanged.
    OptionalAttr<StrArrayAttr>:$clause_names,
    // Names in clause_names that stand for a *list* rather than one value, so
    // every operand carrying that name belongs to it: `captures` is the case,
    // spliced into as many call arguments as there are entries.  Recorded
    // separately because a list can legitimately be empty — a parallel with no
    // captures has no operand at all, and only this attribute distinguishes
    // "empty list" from "not a list".
    OptionalAttr<StrArrayAttr>:$list_names
  );

  let results = (outs);

  // Optional region holding the parallel body before outlining.
  // After OmpOutliningPass runs this region is empty.
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
