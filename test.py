import os
import sys
import argparse
import time

# your ffmpeg location and this line should be inf
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"

import moviepy
from moviepy.editor import VideoFileClip
import moviepy.video.io.ffmpeg_writer

import torch
from torchsummary import summary
import numpy
import PIL
import cv2

from model import SuckModel



def load_model(path):
    suckModel = SuckModel()
    suckModel.load_state_dict(torch.load(path))
    suckModel.cuda().eval()

    return suckModel

def img_preprocess(image):
    # if image.getmode() == "RGBA":
    #     image = image.convert("RGB")
    
    tensor = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(image)[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
    width = tensor.shape[2]
    height = tensor.shape[1]
    # print("origin shape:", tensor.shape)
    
    intPadr = (2 - (width % 2)) % 2
    intPadb = (2 - (height % 2)) % 2
    
    tensor_pre = tensor.view(1, 3, height, width)
    tenPreprocessed = torch.nn.functional.pad(input=tensor_pre, pad=[0, intPadr, 0, intPadb], mode='replicate')
    
    return tenPreprocessed

def img_postprocess(tensor, is_img=False):
    img = tensor.data.squeeze().float().clamp_(0, 1).numpy()
    # print("after shape:", img.shape)
    if is_img:
        img = numpy.transpose(img[[0, 1, 2], :, :], (1, 2, 0))
    else:
        img = numpy.transpose(img[[2, 1, 0], :, :], (1, 2, 0))
    img = (img * 255).round().astype(numpy.uint8)

    return img

def img_postprocess_for_img(tensor):
    img = tensor.data.squeeze().float().clamp_(0, 1).numpy()
    # print("after shape:", img.shape)
    
    img = (img * 255).round().astype(numpy.uint8)

    return img

def test_with_img(model, image_path1, image_path2):    
    tensor1 = img_preprocess( PIL.Image.open(image_path1) )
    tensor2 = img_preprocess( PIL.Image.open(image_path2) )

    with torch.no_grad():
        seq = model(tensor1, tensor2)
        
    return seq

def write_video(sequence, output):
    pass

def read_video(path):
    video_clip = VideoFileClip(path)
    return video_clip

def test_with_video(model, video_path, out_path):
    video_reader = read_video(video_path)
    audio_clip = video_reader.audio
    audio_clip.write_audiofile(os.path.join(out_path, "tmp.wav"))
    frames = [None, None, None]
    
    # ditn is a 4x sr model and softmax interpolates frame at 0.5 of two frames 
    # if u want to interpolate more frames u should modify the param flt in suckModel.py
    # self.softmax(tensor1, tensor2, flt) flt is a tensor of interpolation position
    with moviepy.video.io.ffmpeg_writer.FFMPEG_VideoWriter(filename=os.path.join(out_path, "out.mp4"), size=(4 * video_reader.w, 4 * video_reader.h), fps=2 * video_reader.fps) as video_writer:
        # adding audio will be implement later 
        
        for frame in video_reader.iter_frames():
            frames[2] = img_preprocess(frame)

            if frames[0] is not None:
                sequence = model(frames[0], frames[2])
                for res in sequence:
                    video_writer.write_frame(img_postprocess(res))
                    
            frames[0] = frames[2]
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True, help="model path")
    parser.add_argument("--output_path", type=str, required=True, help="path of output dir")
    parser.add_argument("--image_path1", type=str, help="first image path")
    parser.add_argument("--image_path2", type=str, help="second image path")
    parser.add_argument("--video_path", type=str, help="input path of video")

    args = parser.parse_args()

    try:
        os.mkdir(args.output_path)
    except Exception:
        pass

    model = load_model(args.model_path)

    start_time = time.time()

    if args.video_path:
        try:
            if args.video_path.split('.')[-1] not in ['avi', 'mp4', 'webm', 'wmv'] :
                raise ValueError("not legal video format")
                
        except Exception:
            print(Exception)
            sys.exit(1)
        
        seq = test_with_video(model, args.video_path, args.output_path)
        
        end_time = time.time()
        running_time = end_time - start_time
        print(f"duration: {running_time}")
                
    elif args.image_path1 and args.image_path2 is not None:
        try:
            if args.image_path1.split('.')[-1] not in ['bmp', 'jpg', 'jpeg', 'png'] and args.image_path2.split('.')[-1] not in ['bmp', 'jpg', 'jpeg', 'png']:
                raise ValueError("not legal image format")
        except Exception:
            print(Exception)
            sys.exit(1)
        
        seq = test_with_img(model, args.image_path1, args.image_path2)
        for idx, item in enumerate(seq):
            path = os.path.join(args.output_path, "image_" + str(idx) + ".jpg")
            print(item.shape)
            cv2.imwrite(path, img_postprocess(item, True) )

        end_time = time.time()
        running_time = end_time - start_time
        print(f"duration: {running_time}")
    
    elif args.image_path1 and args.image_path2 is not None:
        try:
            raise ValueError("Only one image input")
        except Exception:
            print(Exception)
            sys.exit(1)
    else:
        print("did nothing🤣")
    

