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
    "::mlir::ArrayAttr":$args
  );
  let assemblyFormat = "`<` $callee `,` `[` $args `]` `>`";
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
    [NoMemoryEffect, AttrSizedOperandSegments]> {

  let summary = "Encodes a fully-evaluated LoweringPlan as MLIR attributes";

  let arguments = (ins
    // Clause operands. Carried as real SSA operands so the values from the
    // original omp.* op survive into OmpOutliningPass — looking them up by
    // symbolic name in the plan would otherwise lose the SSA reference.
    // (Two optional operands require AttrSizedOperandSegments above.)
    Optional<AnyType>:$num_threads,
    Optional<AnyType>:$if_clause,

    StrAttr:$runtime,
    // NOTE: 'construct' is a reserved TableGen keyword so the accessor is
    // named construct_name here; the assembly format still prints 'construct ='.
    StrAttr:$construct_name,
    DictionaryAttr:$prop_dict,
    ArrayAttr:$pre,
    ArrayAttr:$invoke,
    ArrayAttr:$post
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
    (`,` `num_threads` `=` $num_threads^ `:` type($num_threads))?
    (`,` `if_clause` `=` $if_clause^ `:` type($if_clause))?
    $body
    attr-dict
  }];

  let hasVerifier = 1;
}
