
import argparse
from os import listdir
from os.path import isfile, join
from time import time
from time import sleep

from keras_yolo3.yolo3.model import yolo_eval
from keras_yolo3.yolo3.utils import letterbox_image

import ktc
import tensorflow as tf
import onnx
from PIL import Image
import numpy as np


def postprocess(inf_results, ori_image_shape, conf_t  , iou_t ):
    tensor_data = [tf.convert_to_tensor(data, dtype=tf.float32) for data in inf_results]

    # Get anchor info
    anchors_path = "/data1/keras_yolo3/model_data/yolo_anchors.txt"
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)

    # Postprocess of Yolov3
    num_classes = 80
    boxes, scores, classes = yolo_eval(tensor_data, anchors, num_classes, ori_image_shape, score_threshold=conf_t, iou_threshold=iou_t)
    with tf.Session() as sess:
        boxes = boxes.eval()
        scores = scores.eval()
        classes = classes.eval()

    return boxes, scores, classes

# Preprocess of Yolov3
def preprocess(pil_img, img_size):    
    model_input_size = (img_size, img_size)  # to match our model input size when converting
    boxed_image = letterbox_image(pil_img, model_input_size)
    np_data = np.array(boxed_image, dtype='float32')

    np_data /= 255.
    return np_data


def setup_parser(test_args):
    """Setup the command line parser."""
    parser = argparse.ArgumentParser(description="Runs an inference on multiple images.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--demo",action="store_true", help="run demo on your image")
    parser.add_argument("--image", help="path to your image for demo", default="/data1/000000350003.jpg", type=str)
    parser.add_argument("--path", help="directory of your images", default="/workspace/COCO/valid2017", type=str)
    parser.add_argument("--nef", help="path to your nef model", default="/data1/batch_compile/models_520.nef", type=str)    
    parser.add_argument("--step", help="number of images for one model in every step", default=20, type=int)
    parser.add_argument("--init", help="initialization time between models", default=10, type=int)
    parser.add_argument("--model", help="model's number", default=0, type=int)
    parser.add_argument("--threads", help="choose number of workers for inference(only for 520 model)", default=16, type=int)
    parser.add_argument("--img-size", help="image size", default=416, type=int)
    parser.add_argument("--conf-t", help="confidence threshold", default=0.6, type=float) 
    parser.add_argument("--iou-t", help="iou threshold for NMS", default=0.5, type=float)
    #TODO add anchors
    return parser.parse_args(test_args)
        

def main(test_args = None):
    args = setup_parser(test_args)
    sleep(int(args.model)*args.init)
    step = args.step
    
    # Run demo on one image
    if args.demo:
        input_image = Image.open(args.image)
        print("Image is in: ", args.image)
        image = preprocess(input_image, args.img_size)
        
        tic = time()
        out_data = ktc.kneron_inference([image], nef_file=args.nef, radix=7 , model=str(args.model), threads=args.threads)
        print(f'inference time {(time()-tic):.2f}')

        # Postprocess
        tic = time()
        det_res = postprocess(out_data, [input_image.size[1], input_image.size[0]], args.conf_t  , args.iou_t)
        print(f'postprocess time {(time()-tic):.2f}')
        print(det_res)
        
    # Make prediction on multiple images with multiple models    
    else:
        images_folder = args.path + "/images"
        images = [f for f in listdir(images_folder) if isfile(join(images_folder, f))]
        predictions_folder = args.path + "/" + "predictions"
        images_folder = args.path + "/" + "images"
        in_data = []
        
        # Check intersection between predictions and images      
        predictions_name =[f[:12] for f in listdir(predictions_folder) if isfile(join(predictions_folder, f))]
        images_name =[f[:12] for f in listdir(images_folder) if isfile(join(images_folder, f))]
        predicted = list(set(predictions_name) & set(images_name))
        not_predicted = images_name

        # Remove already made predictions
        for element in predicted:                
            not_predicted.remove(element)

        # Create list with images without predictions    
        not_predicted_images = []
        for elem in not_predicted:
            not_predicted_images.append(elem + ".jpg")
            
        image_shape = [] # make list to fill with image sizes
        
        # Preprocess *step* images 
        for i in range(step):
            try:
                input_image = Image.open(args.path + "/images/" + not_predicted_images[i+args.model*step])
                pre = preprocess(input_image, args.img_size)
                image_shape.append(input_image.size)
                in_data.append(pre)
            except:
                print("No more images for model ", args.model)
                break               

        images = not_predicted_images
                   
        print("-"*10)
        print("Preprocessing is done!")
        print("-"*10)
        
        count = 0 # to assign image to prediction
        
        # Make prediction 
        for image in in_data:
            # Inference
            tic = time()
            out_data = ktc.kneron_inference([image], nef_file=args.nef, radix=7 , model=str(args.model), threads=args.threads)
            print(f'inference time {(time()-tic):.2f}')
            
            # Postprocess
            tic = time()
            det_res = postprocess(out_data, [image_shape[count][1], image_shape[count][0]], args.conf_t, args.iou_t)
            print(f'postprocess time {(time()-tic):.2f}') 
            
            # Write predictions in YOLO style
            predictions = [] # create tuple to fill with predictions
            for pred in range(len(det_res[2])):
                # Convert y1x1y2x2 to XcYcwh with relative values
                prediction = (str(det_res[2][pred]) + " " + str(det_res[1][pred]) + " " 
                    + str((det_res[0][pred][1] + det_res[0][pred][3])/2/input_image.size[0])+ " " 
                    + str((det_res[0][pred][0] + det_res[0][pred][2])/2/input_image.size[1])+ " " 
                    + str((det_res[0][pred][3] - det_res[0][pred][1])/input_image.size[0])+ " " 
                    + str((det_res[0][pred][2] - det_res[0][pred][0])/input_image.size[1]))

                predictions.append(prediction)
                
            predictions = '\n'.join(predictions)

            # Save predictions to directory args.path/predictions/*.txt
            with open(predictions_folder + "/" + images[count+args.model*step][:12] + ".txt", 'w') as f:
                f.write(str(predictions))

            count += 1
          
    
if __name__ == "__main__":
    main()
