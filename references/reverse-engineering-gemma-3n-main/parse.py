import os
import sys

# This dictionary is from the zhenhuaw-me/tflite/tflite-master/tflite/utils.py file you provided.
BUILTIN_OPCODE2NAME = {
    0: 'ADD', 1: 'AVERAGE_POOL_2D', 2: 'CONCATENATION', 3: 'CONV_2D', 4: 'DEPTHWISE_CONV_2D',
    5: 'DEPTH_TO_SPACE', 6: 'DEQUANTIZE', 7: 'EMBEDDING_LOOKUP', 8: 'FLOOR', 9: 'FULLY_CONNECTED',
    10: 'HASHTABLE_LOOKUP', 11: 'L2_NORMALIZATION', 12: 'L2_POOL_2D', 13: 'LOCAL_RESPONSE_NORMALIZATION',
    14: 'LOGISTIC', 15: 'LSH_PROJECTION', 16: 'LSTM', 17: 'MAX_POOL_2D', 18: 'MUL', 19: 'RELU',
    20: 'RELU_N1_TO_1', 21: 'RELU6', 22: 'RESHAPE', 23: 'RESIZE_BILINEAR', 24: 'RNN',
    25: 'SOFTMAX', 26: 'SPACE_TO_DEPTH', 27: 'SVDF', 28: 'TANH', 29: 'CONCAT_EMBEDDINGS',
    30: 'SKIP_GRAM', 31: 'CALL', 32: 'CUSTOM', 33: 'EMBEDDING_LOOKUP_SPARSE', 34: 'PAD',
    35: 'UNIDIRECTIONAL_SEQUENCE_RNN', 36: 'GATHER', 37: 'BATCH_TO_SPACE_ND', 38: 'SPACE_TO_BATCH_ND',
    39: 'TRANSPOSE', 40: 'MEAN', 41: 'SUB', 42: 'DIV', 43: 'SQUEEZE', 44: 'UNIDIRECTIONAL_SEQUENCE_LSTM',
    45: 'STRIDED_SLICE', 46: 'BIDIRECTIONAL_SEQUENCE_RNN', 47: 'EXP', 48: 'TOPK_V2', 49: 'SPLIT',
    50: 'LOG_SOFTMAX', 51: 'DELEGATE', 52: 'BIDIRECTIONAL_SEQUENCE_LSTM', 53: 'CAST', 54: 'PRELU',
    55: 'MAXIMUM', 56: 'ARG_MAX', 57: 'MINIMUM', 58: 'LESS', 59: 'NEG', 60: 'PADV2',
    61: 'GREATER', 62: 'GREATER_EQUAL', 63: 'LESS_EQUAL', 64: 'SELECT', 65: 'SLICE',
    66: 'SIN', 67: 'TRANSPOSE_CONV', 68: 'SPARSE_TO_DENSE', 69: 'TILE', 70: 'EXPAND_DIMS',
    71: 'EQUAL', 72: 'NOT_EQUAL', 73: 'LOG', 74: 'SUM', 75: 'SQRT', 76: 'RSQRT',
    77: 'SHAPE', 78: 'POW', 79: 'ARG_MIN', 80: 'FAKE_QUANT', 81: 'REDUCE_PROD',
    82: 'REDUCE_MAX', 83: 'PACK', 84: 'LOGICAL_OR', 85: 'ONE_HOT', 86: 'LOGICAL_AND',
    87: 'LOGICAL_NOT', 88: 'UNPACK', 89: 'REDUCE_MIN', 90: 'FLOOR_DIV', 91: 'REDUCE_ANY',
    92: 'SQUARE', 93: 'ZEROS_LIKE', 94: 'FILL', 95: 'FLOOR_MOD', 96: 'RANGE',
    97: 'RESIZE_NEAREST_NEIGHBOR', 98: 'LEAKY_RELU', 99: 'SQUARED_DIFFERENCE', 100: 'MIRROR_PAD',
    101: 'ABS', 102: 'SPLIT_V', 103: 'UNIQUE', 104: 'CEIL', 105: 'REVERSE_V2', 106: 'ADD_N',
    107: 'GATHER_ND', 108: 'COS', 109: 'WHERE', 110: 'RANK', 111: 'ELU', 112: 'REVERSE_SEQUENCE',
    113: 'MATRIX_DIAG', 114: 'QUANTIZE', 115: 'MATRIX_SET_DIAG', 116: 'ROUND', 117: 'HARD_SWISH',
    118: 'IF', 119: 'WHILE', 120: 'NON_MAX_SUPPRESSION_V4', 121: 'NON_MAX_SUPPRESSION_V5',
    122: 'SCATTER_ND', 123: 'SELECT_V2', 124: 'DENSIFY', 125: 'SEGMENT_SUM', 126: 'BATCH_MATMUL',
    127: 'PLACEHOLDER_FOR_GREATER_OP_CODES', 128: 'CUMSUM', 129: 'CALL_ONCE', 130: 'BROADCAST_TO',
    131: 'RFFT2D', 132: 'CONV_3D', 133: 'IMAG', 134: 'REAL', 135: 'COMPLEX_ABS', 136: 'HASHTABLE',
    137: 'HASHTABLE_FIND', 138: 'HASHTABLE_IMPORT', 139: 'HASHTABLE_SIZE', 140: 'REDUCE_ALL',
    141: 'CONV_3D_TRANSPOSE', 142: 'VAR_HANDLE', 143: 'READ_VARIABLE', 144: 'ASSIGN_VARIABLE',
    145: 'BROADCAST_ARGS', 146: 'RANDOM_STANDARD_NORMAL', 147: 'BUCKETIZE', 148: 'RANDOM_UNIFORM',
    149: 'MULTINomial', 150: 'GELU', 151: 'DYNAMIC_UPDATE_SLICE', 152: 'RELU_0_TO_1',
    153: 'UNSORTED_SEGMENT_PROD', 154: 'UNSORTED_SEGMENT_MAX', 155: 'UNSORTED_SEGMENT_SUM',
    156: 'ATAN2', 157: 'UNSORTED_SEGMENT_MIN', 158: 'SIGN', 159: 'BITCAST', 160: 'BITWISE_XOR',
    161: 'RIGHT_SHIFT',162: 'STABLEHLO_LOGISTIC', 163: 'STABLEHLO_ADD', 164: 'STABLEHLO_DIVIDE',
    165: 'STABLEHLO_MULTIPLY', 166: 'STABLEHLO_MAXIMUM', 167: 'STABLEHLO_RESHAPE', 168: 'STABLEHLO_CLAMP',
    169: 'STABLEHLO_CONCATENATE', 170: 'STABLEHLO_BROADCAST_IN_DIM', 171: 'STABLEHLO_CONVOLUTION',
    172: 'STABLEHLO_SLICE', 173: 'STABLEHLO_CUSTOM_CALL', 174: 'STABLEHLO_REDUCE', 175: 'STABLEHLO_ABS',
    176: 'STABLEHLO_AND', 177: 'STABLEHLO_COSINE', 178: 'STABLEHLO_EXPONENTIAL', 179: 'STABLEHLO_FLOOR',
    180: 'STABLEHLO_LOG', 181: 'STABLEHLO_MINIMUM', 182: 'STABLEHLO_NEGATE', 183: 'STABLEHLO_OR',
    184: 'STABLEHLO_POWER', 185: 'STABLEHLO_REMAINDER', 186: 'STABLEHLO_RSQRT', 187: 'STABLEHLO_SELECT',
    188: 'STABLEHLO_SUBTRACT', 189: 'STABLEHLO_TANH', 190: 'STABLEHLO_SCATTER', 191: 'STABLEHLO_COMPARE',
    192: 'STABLEHLO_CONVERT', 193: 'STABLEHLO_DYNAMIC_SLICE', 194: 'STABLEHLO_DYNAMIC_UPDATE_SLICE',
    195: 'STABLEHLO_PAD', 196: 'STABLEHLO_IOTA', 197: 'STABLEHLO_DOT_GENERAL', 198: 'STABLEHLO_REDUCE_WINDOW',
    199: 'STABLEHLO_SORT', 200: 'STABLEHLO_WHILE', 201: 'STABLEHLO_GATHER', 202: 'STABLEHLO_TRANSPOSE',
    203: 'DILATE', 204: 'STABLEHLO_RNG_BIT_GENERATOR', 205: 'REDUCE_WINDOW', 206: 'STABLEHLO_COMPOSITE',
    207: 'STABLEHLO_SHIFT_LEFT', 208: 'STABLEHLO_CBRT',
}

def opcode_to_name_func(opc):
    if opc in BUILTIN_OPCODE2NAME:
        return BUILTIN_OPCODE2NAME[opc]
    else:
        return f"UNKNOWN_OPCODE_{opc}"

try:
    # This imports the Model *class* directly, due to tflite/__init__.py
    from tflite import Model
except ImportError:
    print("Critical Error: The 'tflite' Python package (from the 'zhenhuaw-me/tflite/tflite-master' directory) was not found.")
    print("Please ensure that the 'tflite' directory is in your Python path (PYTHONPATH).")
    print("Alternatively, place this script inside the 'tflite-master' directory and run it from there.")
    print("Example: If your tflite-master is at '/path/to/tflite-master', add '/path/to/tflite-master' to PYTHONPATH.")
    sys.exit(1)

TENSOR_TYPE_NAMES = {
    0: "FLOAT32", 1: "FLOAT16", 2: "INT32", 3: "UINT8", 4: "INT64",
    5: "STRING", 6: "BOOL", 7: "INT16", 8: "COMPLEX64", 9: "INT8",
    10: "FLOAT64", 11: "COMPLEX128", 12: "UINT64", 13: "RESOURCE",
    14: "VARIANT", 15: "UINT32", 16: "UINT16", 17: "INT4"
}

def get_tensor_type_name(tensor_type_code):
    return TENSOR_TYPE_NAMES.get(tensor_type_code, f"UNKNOWN_TYPE_CODE_{tensor_type_code}")

def dump_tflite_graph_info(model_path):
    print(f"🔍 Processing model: {model_path}\n")

    try:
        with open(model_path, 'rb') as f:
            buf = f.read()
        # CORRECTED LINE:
        # Model is the class itself, so call GetRootAs directly on it.
        model_obj = Model.GetRootAs(buf, 0)
    except Exception as e:
        print(f"❌ Error loading or parsing model {model_path}: {e}")
        return

    if model_obj.SubgraphsLength() == 0:
        print("ℹ️ Model has no subgraphs.")
        return

    subgraph = model_obj.Subgraphs(0)
    if not subgraph:
        print("❌ Could not get the first subgraph.")
        return

    subgraph_name_bytes = subgraph.Name()
    subgraph_name = subgraph_name_bytes.decode('utf-8') if subgraph_name_bytes else 'N/A'
    print(f"📊 Subgraph Name: {subgraph_name}")
    print(f"   Number of Tensors: {subgraph.TensorsLength()}")
    print(f"   Number of Operators: {subgraph.OperatorsLength()}\n")

    operator_codes_map = {}
    for i in range(model_obj.OperatorCodesLength()):
        operator_codes_map[i] = model_obj.OperatorCodes(i)

    print("--- ⚙️ Operators ---")
    for i in range(subgraph.OperatorsLength()):
        op = subgraph.Operators(i)
        if not op:
            print(f"⚠️ Warning: Operator at index {i} is None.")
            continue

        op_code_idx = op.OpcodeIndex()
        op_code_entry = operator_codes_map.get(op_code_idx)

        op_name = f"UNKNOWN_OPCODE_IDX_{op_code_idx}"
        builtin_code_val = "N/A"

        if op_code_entry:
            builtin_code_val = op_code_entry.BuiltinCode()
            if builtin_code_val == 32:
                custom_code_bytes = op_code_entry.CustomCode()
                op_name = f"CUSTOM:{custom_code_bytes.decode('utf-8')}" if custom_code_bytes else "CUSTOM:(No custom code string)"
            else:
                op_name = opcode_to_name_func(builtin_code_val)

        print(f"\n➡️ Operator {i}: {op_name} (Opcode Index: {op_code_idx}, Builtin Code Value: {builtin_code_val})")

        print(f"  Inputs (Tensor Indices & Details):")
        if op.InputsLength() > 0:
            for k in range(op.InputsLength()):
                tensor_idx = op.Inputs(k)
                if tensor_idx < 0:
                    print(f"    {k}: Index {tensor_idx} (Optional/Unused)")
                    continue
                if tensor_idx < subgraph.TensorsLength():
                    tensor = subgraph.Tensors(tensor_idx)
                    t_name_bytes = tensor.Name()
                    t_name = t_name_bytes.decode('utf-8') if t_name_bytes else f"Tensor_{tensor_idx}"

                    shape_str = "Shape: Not Specified"
                    if tensor.ShapeSignatureLength() > 0:
                        shape_list = [tensor.ShapeSignature(s) for s in range(tensor.ShapeSignatureLength())]
                        shape_str = f"Shape Signature: {shape_list} (Dynamic)"
                    elif tensor.ShapeLength() > 0:
                        shape_list = [tensor.Shape(s) for s in range(tensor.ShapeLength())]
                        shape_str = f"Shape: {shape_list}"

                    print(f"    {k}: Index {tensor_idx} (Name: '{t_name}', {shape_str})")
                else:
                    print(f"    {k}: Index {tensor_idx} (Error: Tensor index out of bounds!)")
        else:
            print("    None")

        print(f"  Outputs (Tensor Indices & Details):")
        if op.OutputsLength() > 0:
            for k in range(op.OutputsLength()):
                tensor_idx = op.Outputs(k)
                if tensor_idx < subgraph.TensorsLength():
                    tensor = subgraph.Tensors(tensor_idx)
                    t_name_bytes = tensor.Name()
                    t_name = t_name_bytes.decode('utf-8') if t_name_bytes else f"Tensor_{tensor_idx}"

                    shape_str = "Shape: Not Specified"
                    if tensor.ShapeSignatureLength() > 0:
                        shape_list = [tensor.ShapeSignature(s) for s in range(tensor.ShapeSignatureLength())]
                        shape_str = f"Shape Signature: {shape_list} (Dynamic)"
                    elif tensor.ShapeLength() > 0:
                        shape_list = [tensor.Shape(s) for s in range(tensor.ShapeLength())]
                        shape_str = f"Shape: {shape_list}"

                    print(f"    {k}: Index {tensor_idx} (Name: '{t_name}', {shape_str})")
                else:
                     print(f"    {k}: Index {tensor_idx} (Error: Tensor index out of bounds!)")
        else:
            print("    None")
        print("-" * 30)

    print("\n--- 🧱 Tensors (including Weights) ---")
    for i in range(subgraph.TensorsLength()):
        tensor = subgraph.Tensors(i)
        # continue
        if not tensor:
            print(f"⚠️ Warning: Tensor at index {i} is None.")
            continue

        tensor_name_bytes = tensor.Name()
        tensor_name = tensor_name_bytes.decode('utf-8') if tensor_name_bytes else f"Tensor_{i}"

        shape_description = "Shape: Not Specified or Empty"
        if tensor.ShapeSignatureLength() > 0:
            shape_list = [tensor.ShapeSignature(s_idx) for s_idx in range(tensor.ShapeSignatureLength())]
            shape_description = f"Shape Signature (Dynamic): {shape_list}"
        elif tensor.ShapeLength() > 0:
            shape_list = [tensor.Shape(s_idx) for s_idx in range(tensor.ShapeLength())]
            shape_description = f"Shape: {shape_list}"

        buffer_idx = tensor.Buffer()
        is_variable = tensor.IsVariable()

        print(f"\nTensor {i}: '{tensor_name}'")
        print(f"  {shape_description}")
        print(f"  Type: {get_tensor_type_name(tensor.Type())} (Raw Code: {tensor.Type()})")
        print(f"  Buffer Index: {buffer_idx}")
        print(f"  Is Variable: {is_variable}")

        is_weight_info = "Data: Not a typical weight tensor."
        if is_variable:
            is_weight_info = "Data: Marked as a variable (stateful, could be weights)."
        elif buffer_idx > 0 and buffer_idx < model_obj.BuffersLength():
            buffer_obj = model_obj.Buffers(buffer_idx)
            if buffer_obj and buffer_obj.DataLength() > 0:
                is_weight_info = f"Data: Buffer {buffer_idx} has {buffer_obj.DataLength()} bytes. Likely a weight/constant."
            elif buffer_obj:
                is_weight_info = f"Data: Buffer {buffer_idx} exists but is empty."
            else:
                 is_weight_info = f"Data: Buffer {buffer_idx} object is None (schema issue?)."
        elif buffer_idx == 0:
            is_weight_info = "Data: Uses Buffer 0 (typically for activations/inputs/outputs, not weights)."
        else:
            is_weight_info = f"Data: Buffer index {buffer_idx} invalid (Model Buffers: {model_obj.BuffersLength()})."
        print(f"  {is_weight_info}")

        quant_params = tensor.Quantization()
        if quant_params:
            details = []
            if quant_params.ScaleLength() > 0:
                scales = [quant_params.Scale(s) for s in range(quant_params.ScaleLength())][:10]
                details.append(f"Scale: {scales} ({quant_params.ScaleLength()})")
            if quant_params.ZeroPointLength() > 0:
                zero_points = [quant_params.ZeroPoint(z) for z in range(quant_params.ZeroPointLength())][:10]
                details.append(f"ZeroPoint: {zero_points} ({quant_params.ZeroPointLength()})")
            if hasattr(quant_params, 'MinLength') and quant_params.MinLength() > 0:
                mins = [quant_params.Min(m) for m in range(quant_params.MinLength())][:10]
                details.append(f"Min: {mins} ({quant_params.MinLength()})")
            if hasattr(quant_params, 'MaxLength') and quant_params.MaxLength() > 0:
                maxs = [quant_params.Max(m) for m in range(quant_params.MaxLength())][:10]
                details.append(f"Max: {maxs} ({quant_params.MaxLength()})")
            if hasattr(quant_params, 'QuantizedDimension') and quant_params.QuantizedDimension() is not None:
                 details.append(f"QuantizedDim: {quant_params.QuantizedDimension()}")

            if details:
                print(f"  Quantization: {'; '.join(details)}")
        print("-" * 30)

    print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    tflite_model_files = [
        "TF_LITE_EMBEDDER",
        "TF_LITE_PER_LAYER_EMBEDDER",
        "TF_LITE_PREFILL_DECODE",
        "TF_LITE_VISION_ADAPTER",
        "TF_LITE_VISION_ENCODER"
    ]

    model_directory = "."

    print("🚀 TFLite Model Graph Information Dumper 🚀")
    print("="*50)
    print("Ensure the 'tflite' package from 'zhenhuaw-me/tflite/tflite-master' is accessible.")
    print(f"Looking for model files in: {os.path.abspath(model_directory)}\n")

    all_files_found = True
    for file_name in tflite_model_files:
        full_model_path = os.path.join(model_directory, file_name)
        if not os.path.exists(full_model_path):
            print(f"⚠️ File not found: {full_model_path}")
            all_files_found = False

    if not all_files_found:
        print("\n❌ Some model files were not found. Please check paths and try again.")
        sys.exit(1)

    for file_name in tflite_model_files:
        full_model_path = os.path.join(model_directory, file_name)
        dump_tflite_graph_info(full_model_path)

    print("✅ Script finished.")
