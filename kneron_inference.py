"""
Generic inference function for ONNX, BIE, or NEF model.
"""
import pathlib
import sys

ROOT_FOLDER = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_FOLDER))

import python_flow.common.exceptions as exceptions
import python_flow.nef.nef as nef
import python_flow.dynasty.dynasty as dynasty

def kneron_inference(pre_results, nef_file="", onnx_file="", bie_file="", model_id=None,
                     input_names=[], radix=8, data_type="float", reordering=[],
                     ioinfo_file="", dump=False, platform=520, model='0', threads = 16):
    """Performs inference on the input model given the specified parameters.

    Input pre_results should be in format (1, h, w, c).

    Arguments:
        pre_results: List of NumPy arrays in channel last format from preprocessing
        nef_file: Path to NEF model for inference
        onnx_file: Path to ONNX model for inference, unused if nef_file is specified
        bie_file: Path to BIE model for inference, unused if nef_file/onnx_file is specified
        model_id: Integer of model to run inference, only necessary for NEF with multiple models
        input_names: List of input node names of BIE/ONNX model, unused if nef_file is specified
        radix: Integer radix to convert from float to fixed input
        data_type: String format of the resulting output, "fixed" or "float"
        dump: Boolean flag to dump intermediate nodes
        reordering: List of node names/integers specifying the output order
        ioinfo_file: String path to file mapping output node number to name, only used with NEF
        platform: Integer indicating platform of Dynasty fixed model
    """
    dump = 2 if dump else 0
    if nef_file:
        nef_path = pathlib.Path(nef_file).resolve()
        #print("nef_path :", nef_path)
        platform, model_name, input_folder, output_folder = nef.setup_nef(nef_path, model_id, model)
        #print("platform, model_name, input_folder, output_folder:", platform, model_name, input_folder, output_folder)
        input_files = nef.prep_csim_inputs(pre_results, input_folder, radix, str(platform))
        #print("input_files: ", input_files)
        output = nef.nef_inference(
            model_name, str(platform), data_type, nef_path.parent, input_files, input_folder,
            output_folder, reordering, ioinfo_file, str(dump), threads = 16)
    elif onnx_file:
        onnx = pathlib.Path(onnx_file).resolve()
        input_folder = onnx.parent / "out/onnx/inputs" / onnx.name
        output_folder = onnx.parent / "out/onnx/outputs" / onnx.name

        input_files = dynasty.prep_dynasty(
            pre_results, input_folder, output_folder, input_names, radix, platform, False)
        output = dynasty.dynasty_inference(
            onnx_file, "Float", str(platform), data_type, input_files, input_names,
            str(output_folder), reordering, dump)
    elif bie_file:
        bie = pathlib.Path(bie_file).resolve()
        input_folder = bie.parent / "out/bie/inputs" / bie.name
        output_folder = bie.parent / "out/bie/outputs" / bie.name

        input_files = dynasty.prep_dynasty(
            pre_results, input_folder, output_folder, input_names, radix, platform, True)
        output = dynasty.dynasty_inference(
            bie_file, "bie", str(platform), data_type, input_files, input_names,
            str(output_folder), reordering, dump)
    else:
        raise exceptions.RequiredConfigError("No input model selected for inference.")

    return output
