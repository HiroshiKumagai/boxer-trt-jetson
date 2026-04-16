#!/usr/bin/env python3
"""
Fix dinov3.onnx by inlining the /rope_embed/If and /rope_embed/If_1 nodes.

TensorRT requires both branches of an ONNX If node to have the same output shape,
but rope_embed generates If nodes where branches have different shapes.

Analysis shows:
  /rope_embed/If  : condition = (sub_result > 0) — always True for 2D angle tensor
                    → inline then_branch (adds Constant/Reshape/Expand/Concat nodes)
  /rope_embed/If_1: condition = (sub_result < 0) — always False
                    → inline else_branch (Identity of /rope_embed/Flatten_1_output_0)
"""
import onnx
import onnx.helper as helper
import sys
import os


def fix_rope_embed_if(input_path, output_path):
    model = onnx.load(input_path)
    graph = model.graph

    new_nodes = []
    extra_nodes = []

    for node in graph.node:
        if node.name == '/rope_embed/If':
            # Condition (sub_result > 0) is always True for a 2D angles tensor.
            # Inline then_branch nodes into the main graph.
            for attr in node.attribute:
                if attr.name == 'then_branch':
                    extra_nodes.extend(list(attr.g.node))
            # Map If output -> then_branch final output via Identity
            extra_nodes.append(helper.make_node(
                'Identity',
                inputs=['/rope_embed/Concat_4_output_0'],
                outputs=['/rope_embed/If_output_0'],
                name='/rope_embed/If_inline',
            ))
            # Do not append the original If node

        elif node.name == '/rope_embed/If_1':
            # Condition (sub_result < 0) is always False.
            # Inline else_branch: Identity('/rope_embed/Flatten_1_output_0')
            extra_nodes.append(helper.make_node(
                'Identity',
                inputs=['/rope_embed/Flatten_1_output_0'],
                outputs=['/rope_embed/If_1_output_0'],
                name='/rope_embed/If_1_inline',
            ))
            # Do not append the original If_1 node

        else:
            new_nodes.append(node)

    new_nodes.extend(extra_nodes)

    del graph.node[:]
    graph.node.extend(new_nodes)

    # Run shape inference to rebuild value_info
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"  Shape inference warning (non-fatal): {e}")

    onnx.save(model, output_path)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Saved: {output_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    onnx_dir = os.environ.get("ONNX_DIR", "/workspace/onnx_weights")
    input_path = os.path.join(onnx_dir, "dinov3.onnx")
    output_path = input_path  # overwrite in-place
    print(f"Fixing rope_embed If nodes in {input_path} ...")
    fix_rope_embed_if(input_path, output_path)
    print("Done.")
