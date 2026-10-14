// Includes the mlir-tblgen output for OmpLoweringOps.td, in dependency order.

#pragma once

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
// Referenced by the generated .inc: attributes are stored as properties.
#include "mlir/Bytecode/BytecodeOpInterface.h"

#include "OmpLowering/IR/OmpLoweringOps.dialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "OmpLowering/IR/OmpLoweringAttrs.h.inc"

#define GET_OP_CLASSES
#include "OmpLowering/IR/OmpLoweringOps.h.inc"
