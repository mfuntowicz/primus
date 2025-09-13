#ifndef PRIMUS_DIALECT_PRIMUS_ATTRS
#define PRIMUS_DIALECT_PRIMUS_ATTRS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

def Primus_AttentionMemberDescriptor : AttrDef<Primus_Dialect, "AttentionMemberDescriptor"> {
    let mnemonic = "attention_member";
    let summary = "Attribute that models a single Scaled Dot Product Attention (SDPA) operand information";
      let parameters = (ins
          "uint32_t":$numHeads,
          "uint32_t":$headSize
      );
      let assemblyFormat = "`<` `num_heads` `=` $numHeads `,` `head_size` `=` $headSize `>`";
}

#endif // PRIMUS_DIALECT_PRIMUS_ATTRS