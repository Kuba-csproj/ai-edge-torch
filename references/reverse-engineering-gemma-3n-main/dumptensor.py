import numpy as np
from tflite import Model
model_path = 'TF_LITE_EMBEDDER'
#
# model_path = 'TF_LITE_PREFILL_DECODE'
with open(model_path, 'rb') as f:
    buf = f.read()
# CORRECTED LINE:
# Model is the class itself, so call GetRootAs directly on it.
model_obj = Model.GetRootAs(buf, 0)

subgraph = model_obj.Subgraphs(0)


TENSOR_TYPE_NAMES = {
    0: "FLOAT32", 1: "FLOAT16", 2: "INT32", 3: "UINT8", 4: "INT64",
    5: "STRING", 6: "BOOL", 7: "INT16", 8: "COMPLEX64", 9: "INT8",
    10: "FLOAT64", 11: "COMPLEX128", 12: "UINT64", 13: "RESOURCE",
    14: "VARIANT", 15: "UINT32", 16: "UINT16", 17: "INT4"
}

def get_tensor_type_name(tensor_type_code):
    return TENSOR_TYPE_NAMES.get(tensor_type_code, f"UNKNOWN_TYPE_CODE_{tensor_type_code}")


for i in range(subgraph.TensorsLength()):
    tensor = subgraph.Tensors(i)
    tensor_name_bytes = tensor.Name()
    tensor_name = tensor_name_bytes.decode('utf-8') if tensor_name_bytes else f"Tensor_{i}"

    shape_list = [tensor.Shape(s_idx) for s_idx in range(tensor.ShapeLength())]
    # print(tensor_name, shape_list)
    #
    # continue
    # [240 254  16 ...  44  46  15] 30402893411
    # [240 254  16 ...  44  46  15] 30402893411

    if tensor_name == 'jit(wrapper)/jit(main)/jit(jax_fn)/embedder.lookup_embedding_table/composite' or tensor_name == 'jit(wrapper)/jit(main)/jit(jax_fn)/GeminiModel.decode_graph/GeminiModel.decode_softmax/transformer.decode_softmax/embedder.decode/composite':
        print(f"  Type: {get_tensor_type_name(tensor.Type())} (Raw Code: {tensor.Type()})")
        print(tensor_name, shape_list)
        # Get tensor data as numpy array
        buffer_index = tensor.Buffer()
        buffer = model_obj.Buffers(buffer_index)
        data = buffer.DataAsNumpy()
        if data is not None:
            print(data.shape)
            print(data)
            print(np.sum(data))
            # Convert raw data to numpy array with proper type and reshape
            # tensor_type = tensor.Type()
            # dtype_map = {
            #     0: np.float32, 1: np.float16, 2: np.int32,
            #     3: np.uint8, 4: np.int64
            # }
            # numpy_dtype = dtype_map.get(tensor_type, np.float32)
            # tensor_array = np.frombuffer(data, dtype=numpy_dtype)
            # if shape_list:
            #     tensor_array = tensor_array.reshape(shape_list)
            # print(f"Tensor as numpy array:\n{tensor_array}")
