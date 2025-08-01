module {
    func.func @apply_rotary_embedding_f32(
        %x1: tensor<?x?xf32>,
        %x2: tensor<?x?xf32>,
        %cos: tensor<?x?xf32>,
        %sin: tensor<?x?xf32>,
        %out1: tensor<?x?xf32>,
        %out2: tensor<?x?xf32>
    ) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
        // Constants indexes (0 and 1) for tensor dimensions
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index

        // Get the dynamic dimensions of the input tensors
        %batch = shape.dim %x1, %c0 : tensor<?x?xf32>, index -> index
        %seqlen = shape.dim %x1, %c1 : tensor<?x?xf32>, index -> index

        // Create destination tensors for the output
        %out_cos_x1 = tensor.empty(%batch, %seqlen) : tensor<?x?xf32>
        %out_sin_x1 = tensor.empty(%batch, %seqlen) : tensor<?x?xf32>
        %out_cos_x2 = tensor.empty(%batch, %seqlen) : tensor<?x?xf32>
        %out_sin_x2 = tensor.empty(%batch, %seqlen) : tensor<?x?xf32>

        // Perform the rotary embedding operation
        %cos_x1 = linalg.mul ins(%x1, %cos: tensor<?x?xf32>, tensor<?x?xf32>) outs(%out_cos_x1 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %sin_x1 = linalg.mul ins(%x2, %sin: tensor<?x?xf32>, tensor<?x?xf32>) outs(%out_sin_x1 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %out1_res = linalg.sub ins(%cos_x1, %sin_x1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%out1 : tensor<?x?xf32>) -> tensor<?x?xf32>

        %cos_x2 = linalg.mul ins(%x2, %cos : tensor<?x?xf32>, tensor<?x?xf32>) outs(%out_cos_x2: tensor<?x?xf32>) -> tensor<?x?xf32>
        %sin_x2 = linalg.mul ins(%x1, %sin : tensor<?x?xf32>, tensor<?x?xf32>) outs(%out_sin_x2: tensor<?x?xf32>) -> tensor<?x?xf32>
        %out2_res = linalg.add ins(%cos_x2, %sin_x2 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%out2 : tensor<?x?xf32>) -> tensor<?x?xf32>

        // return %out1_res, %out2_res : tensor<?x?xf32>, tensor<?x?xf32>
        return %out1_res, %out2_res : tensor<?x?xf32>, tensor<?x?xf32>
    }
}