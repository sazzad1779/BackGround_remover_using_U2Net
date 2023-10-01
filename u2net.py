import sys
import cv2
import numpy as np
from PIL import Image
# sys.path.append('./utils')
import os
import tensorflow as tf
from utils.u2net_utils import (
    imread,
    load_image,
    norm,
    process_result,
    transform,
    format_input_tensor,
    get_capture
)
import argparse

def recognize_from_video(interpreter, args):
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    capture = get_capture(args.video)

    # create video writer if savepath is specified as video format
    f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    if args.savepath:
        # Create a VideoWriter object to save the processed frames as a new video
        # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        # out = cv2.VideoWriter(os.path.join(args.savepath,"demo_output.mp4"), fourcc, 30.0, (f_h, f_w))
        fourcc = cv2.VideoWriter_fourcc(*'XVID') 
        out = cv2.VideoWriter(os.path.join(args.savepath,"demo_output.avi"), fourcc, 30.0, (f_w, f_h))

    print((f_h, f_w))
    frame_shown = False
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord("q")) or not ret:
            print("breaking loop")
            break
        # if frame_shown and cv2.getWindowProperty("frame", cv2.WND_PROP_VISIBLE) == 0:
        #     print("breaking loop")
        #     break

        if args.rgb and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        input_data = transform(frame, (args.width, args.height))

        inputs = format_input_tensor(input_data, input_details, 0)
        interpreter.set_tensor(input_details[0]["index"], inputs)
        interpreter.invoke()

        details = output_details[0]
        dtype = details["dtype"]
        if dtype == np.uint8 or dtype == np.int8:
            quant_params = details["quantization_parameters"]
            int_tensor = interpreter.get_tensor(details["index"])
            real_tensor = int_tensor - quant_params["zero_points"]
            real_tensor = real_tensor.astype(np.float32) * quant_params["scales"]
        else:
            real_tensor = interpreter.get_tensor(details["index"])

        pred = norm(real_tensor[0])
        if args.rgb and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            

        pred = cv2.resize(pred, (f_w, f_h))

        # Convert to BW and make it the size of the original frame
        if isinstance(pred, np.ndarray):
            pred_uint8 = (pred * 255).astype(np.uint8)
            pred = Image.fromarray(pred_uint8)
        
        # Load the input image using PIL
        # Convert to BW and make it the size of the original frame
        if isinstance(frame, np.ndarray):
            frame = (frame ).astype(np.uint8)
            frame = Image.fromarray(frame)
        
        bg = Image.open(args.background_image)  # Create an RGB image
        bg = bg.resize((f_w, f_h))

        if pred.size == frame.size and bg.size == frame.size:
            result = Image.composite(frame, bg, pred)
            # Convert the PIL image to a NumPy array
            opencv_image = np.array(result)
            # Convert the NumPy array to an OpenCV image (BGR format)
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
            print("shape: ",opencv_image.shape)
            if args.show:
                cv2.getWindowProperty("frame", cv2.WND_PROP_VISIBLE)
                cv2.imshow("frame",opencv_image)
            if args.savepath:
                out.write(opencv_image)
        else:
            print("Dimensions of  mask, frame, or background do not match.")
    capture.release()
    if args.savepath:
        out.release()
    cv2.destroyAllWindows()
    print("Script finished successfully.")

def recognize_from_image(interpreter, args):
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    image_files = [
        f for f in os.listdir(args.input) if f.endswith((".jpg", ".jpeg", ".png"))
    ]
    
    # Load the background image
    background_path = args.background_image  # Replace with the actual path to your background image
    background = imread(background_path)
    background = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)

    # input image loop
    for image_name in image_files:
        # Construct the full path of the input image
        image_path = os.path.join(args.input, image_name)
        # prepare input data
        print("print(args.input): ", image_path)

        # prepare input data
        input_data, h, w = load_image(
            image_path, scaled_size=(args.width, args.height), rgb_mode=args.rgb
        )

        # inference
        print("Inference is starting...")

        inputs = format_input_tensor(input_data, input_details, 0)
        interpreter.set_tensor(input_details[0]["index"], inputs)
        interpreter.invoke()

        details = output_details[0]
        dtype = details["dtype"]
        if dtype == np.uint8 or dtype == np.int8:
            quant_params = details["quantization_parameters"]
            int_tensor = interpreter.get_tensor(details["index"])
            real_tensor = int_tensor - quant_params["zero_points"]
            real_tensor = real_tensor.astype(np.float32) * quant_params["scales"]
        else:
            real_tensor = interpreter.get_tensor(details["index"])

        pred = process_result(real_tensor, [h, w])
        if args.savepath:
            # # Check if the folder exists
            # if not os.path.exists(args.savepath):
            #     # If it doesn't exist, create it
            #     os.mkdir(args.savepath)
            #     print(f"Folder '{args.savepath}' created.")
            # else:
            #     print(f"Folder '{args.savepath}' already exists.")

            save_path =  os.path.join(args.savepath, f"res_{image_name}")
            cv2.imwrite(save_path, pred * 255)

        if args.composite or args.show:
            pred = cv2.resize(pred, ( w,h))
            # Convert to BW and make it the size of the original frame
            if isinstance(pred, np.ndarray):
                pred_uint8 = (pred * 255).astype(np.uint8)
                pred = Image.fromarray(pred_uint8)
            # Load the input image using PIL
            img = Image.open(image_path)
            bg = Image.open(args.background_image)  # Create an RGB image
            bg = bg.resize((w, h))
            # print(pred.size,img.size,bg.size)
            if pred.size == img.size and bg.size == img.size:
                result = Image.composite(img, bg, pred)
                if args.show:
                    result.show()
                if args.savepath:
                    result.save(save_path)
                    print(f"saved at : {save_path}")
            else:
                print("Dimensions of pred, mask, img, or empty do not match.")
        if not args.savepath and not args.show and not args.composite :   # it will show black and white when composite is false 
            cv2.imshow("output",pred)
            cv2.waitKey(0)
    print("Script finished successfully.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_path",
        default="./models/u2netp_opset11_float32.tflite",
        help=("give the model path"),
    )
    parser.add_argument(
        "-i", "--input", default="./samples", help=("give the input image")
    )
    parser.add_argument(
        "-v", "--video",
          default=None,
           help=("give the video path"))
    parser.add_argument(    
        "-s",
        "--savepath",
        default=None,
        help="path for the output (image / video).",
    )
    parser.add_argument(
        "-b",
        "--background_image",
        default="./background/bg.jpg",
        help="path for background image",
    )
    parser.add_argument(
        "-c",
        "--composite",
        action="store_true",
        help="Composite input image and predicted alpha value",
    )

    parser.add_argument(
        "-w",
        "--width",
        default=320,
        type=int,
        help="The segmentation width and height for u2net. (default: 320)",
    )
    parser.add_argument(
        "--rgb", action="store_true", help="Use rgb color space (default: bgr)"
    )
    parser.add_argument(
        "--height",
        default=320,
        type=int,
        help="The segmentation height and height for u2net. (default: 320)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show output image.",
    )
    args = parser.parse_args()

    if args.savepath:
        # Check if the folder exists
        if not os.path.exists(args.savepath):
            # If it doesn't exist, create it
            os.mkdir(args.savepath)
            print(f"Folder '{args.savepath}' created.")
        else:
            print(f"Folder '{args.savepath}' already exists.")

    interpreter = tf.lite.Interpreter(model_path=args.model_path)
    if args.video is not None:
        # video mode
        recognize_from_video(interpreter, args)
    else:
        # image mode
        recognize_from_image(interpreter, args)


if __name__ == "__main__":
    main()
