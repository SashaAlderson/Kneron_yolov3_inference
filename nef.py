"""
Perform inference using an input NEF file.
"""
import ctypes
import math
import pathlib
import subprocess
import sys

import python_flow.common.directory_manager as dm
import python_flow.common.exceptions as exceptions
import python_flow.utils.utils as utils

FILE_DIR = pathlib.Path(__file__).resolve().parents[0]
E2E_DIR = pathlib.Path(__file__).resolve().parents[2]
NEF_LIB = ctypes.CDLL(FILE_DIR / "libnef.so")
CSIM520 = str(E2E_DIR / "python_flow/npu_sim")
CSIM720 = str(E2E_DIR / "python_flow/npu_csim")
BASE_INI = str(E2E_DIR / "python_flow/720.ini")

def advance(f, offset=0):
    f.read(offset)
    return int.from_bytes(f.read(4), byteorder='little')

def parse_fw_info(path, key):
    with open(path, 'rb') as f:
        model_cnt = advance(f)

        if "model_cnt" == key:
            yield model_cnt
            return

        model_info_list = []
        for _ in range(model_cnt):
            model_info = {}
            model_info['model_id'] = advance(f)
            model_info['model_ver'] = hex(advance(f))

            model_info['addr_in'] = hex(advance(f))
            model_info['size_in'] = hex(advance(f))

            model_info['addr_out'] = hex(advance(f))
            model_info['size_out'] = hex(advance(f))

            model_info['addr_wbuf'] = hex(advance(f))
            model_info['size_wbuf'] = hex(advance(f))

            model_info['addr_cmd'] = hex(advance(f))
            model_info['size_cmd'] = hex(advance(f))

            model_info['addr_wt'] = hex(advance(f))
            model_info['size_wt'] = hex(advance(f))

            model_info['addr_fw'] = hex(advance(f))
            model_info['size_fw'] = hex(advance(f))

            model_info_list.append(model_info)

            if "all" != key:
                yield model_info[key]

        if "all" == key:
            yield model_info_list
            #print(model_info_list)

def decompose_all_models(nef_file):
    """Separate the command/setup/weight binaries from the combined binary."""
    nef_folder = nef_file.parent
    models = nef_folder / "all_models.bin"
    fw_info = nef_folder / "fw_info.bin"
    model_cnt = next(parse_fw_info(fw_info, "model_cnt"))
    model_id_list = list(parse_fw_info(fw_info, "model_id"))
    size_cmd_list = list(parse_fw_info(fw_info, "size_cmd"))
    size_wt_list = list(parse_fw_info(fw_info, "size_wt"))
    size_fw_list = list(parse_fw_info(fw_info, "size_fw"))

    all_models = []
    with open(models, "r+b") as af:
        for idx in range(model_cnt):
            cur_model = "model_" + str(model_id_list[idx])
            all_models.append(cur_model)
            with open(nef_folder / (cur_model + "_command.bin"), "w+b") as cbf:
                size_cmd = int(size_cmd_list[idx], 16)
                cbf.write(af.read(size_cmd))
                cbf.close()
                pad = math.ceil(size_cmd / 16.0) * 16 - size_cmd
                af.read(pad) #pad to 16

            with open(nef_folder / (cur_model + "_weight.bin"), "w+b") as wbf:
                size_wt = int(size_wt_list[idx], 16)
                assert 0 == (size_wt % 16), "weight size should be aligned 16"
                wbf.write(af.read(size_wt))
                wbf.close()

            with open(nef_folder / (cur_model + "_setup.bin"), "w+b") as fbf:
                size_fw = int(size_fw_list[idx], 16)
                fbf.write(af.read(size_fw))
                fbf.close()
                pad = math.ceil(size_fw / 16.0) * 16 - size_fw
                af.read(pad) #pad to 16

    return all_models

def parse_nef(nef_file):
    """Get the combined binary, firmware info, and platform from the NEF file in C."""
    c_function = NEF_LIB.parse_nef
    c_function.argtypes = [ctypes.c_char_p]
    c_function.restype = ctypes.c_int
    platform = c_function(str(nef_file).encode())

    if platform == -1:
        raise exceptions.LibError(f"Could not parse NEF file correctly: {nef_file}")

    all_models = decompose_all_models(nef_file)
    return platform, all_models

def setup_nef(nef, model_id, model = '0'):
    """Parses NEF model and does extra checks and setup."""
    if not nef.exists():
        raise exceptions.InvalidInputError(f"Input NEF file does not exist: {nef}")

    platform, all_models = parse_nef(nef)
    platform = 520 if platform == 0 else 720

    model_num = "model_" + str(model_id) # potential model to run
    if len(all_models) == 1:
        model_num = all_models[0]
    elif model_num not in all_models:
        raise exceptions.InvalidInputError(f"Specified model ID not found in NEF. ID = {model_id}")
    input_folder = nef.parent / "out/nef/inputs" / model_num / model
    #print(input_folder)
    output_folder = nef.parent / "out/nef/outputs" / model_num / model
    input_folder.mkdir(parents=True, exist_ok=True)
    output_folder.mkdir(parents=True, exist_ok=True)

    return platform, model_num, input_folder, output_folder

def prep_csim_inputs(pre_results, input_folder, radix, platform):
    """Prepare the input files for NEF model."""
    input_files = []
    for index, result in enumerate(pre_results):
        file_name = "_".join([str(platform), "csim_rgba_in", str(index)]) + ".bin"
        input_file = str(input_folder / file_name)
        input_files.append(input_file)
        utils.convert_pre_numpy_to_rgba(result, input_file, radix, platform)

    return input_files

def update_ini(base_ini, new_ini, command, weight, setup, inputs):
    """Updates input INI file for the CSIM 720 model. Assume all files already exist."""
    file_inputs = ",".join(inputs)
    updated_lines = {
        "file_command": "".join(["file_command = ", command, "\n"]),
        "file_weight": "".join(["file_weight = ", weight, "\n"]),
        "file_setup": "".join(["file_setup = ", setup, "\n"]),
        "file_input": "".join(["file_input = ", file_inputs, "\n"])
    }

    new_file = []
    with open(base_ini, "r") as in_file:
        for line in in_file.readlines():
            need_update = None
            for key in updated_lines:
                if line.startswith(key):
                    need_update = key
                    break

            if need_update is not None:
                new_file.append(updated_lines[need_update])
                del updated_lines[need_update]
            else:
                new_file.append(line)

    with open(new_ini, "w") as out_file:
        out_file.write("".join(new_file))

def nef_inference(model_name, platform, data_type, nef_folder, input_files,
                  input_folder, output_folder, reordering, ioinfo_file, dump, threads = 16):
    """Performs inference on the specified model parsed from the NEF file."""
    #print(model_name, platform, data_type,"nef_folder: ", nef_folder,"input_files: ", input_files,
                  #"input_folder: ", input_folder,"output_folder", output_folder,"reordering: ", reordering,"ioinfo_file: ", ioinfo_file, "dump: ", dump)
    command = str(nef_folder / "_".join([model_name, "command.bin"]))
    setup = str(nef_folder / "_".join([model_name, "setup.bin"]))
    weight = str(nef_folder / "_".join([model_name, "weight.bin"]))
    with dm.DirectoryManager(output_folder):
        try:
            if platform == "520":
                subprocess.run([CSIM520, "-d", dump, command, weight, *input_files, setup,
                                "--thread", str(threads)], check=True)
                output = utils.csim_520_to_np(
                    str(output_folder), data_type, reordering, ioinfo_file, False)
            else:
                ini_file = input_folder / "720.ini"
                update_ini(BASE_INI, ini_file, command, weight, setup, input_files)
                subprocess.run([CSIM720, ini_file], check=True)
                output = utils.csim_720_to_np(
                    str(output_folder), data_type, reordering, ioinfo_file, False)
        except subprocess.CalledProcessError as error:
            raise exceptions.LibError(f"Hardware CSIM {platform} failed:\n{error}")
        except exceptions.ConfigError as error:
            sys.exit(error)
    return output
