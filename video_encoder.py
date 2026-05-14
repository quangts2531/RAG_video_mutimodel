import whisper
import os
from scenedetect import detect, ContentDetector
import torch
from lavis.models import load_model_and_preprocess
from moviepy import VideoFileClip
import shutil
import cv2 as cv
from PIL import Image
import numpy as np
import io
import modal
import json

from pprintpp import pprint

app = modal.App("video-encoder-app")


torch.set_grad_enabled(True)


import time
start = time.time()

device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)


class Video_encoder():
    def __init__(self):
        self.blip_model, self.vis_processors, _ = load_model_and_preprocess(name="blip_caption",
                                                         model_type="base_coco",
                                                         is_eval=True,
                                                         device=device_name)
        self.whisper_model = whisper.load_model("small")
        OllamaModel = modal.Cls.from_name("ollama-llava-server", "OllamaServer")
        self.llava_client = OllamaModel()

    def video_encoder(self, video_path, save_path = "cache"):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        video_clip = VideoFileClip(video_path)
        fps = video_clip.fps
        audio_path = "{}/{}.wav".format(save_path, "speech")
        if video_clip.audio is not None:
            video_clip.audio.write_audiofile(audio_path, logger=None)
            resul_transcribe = self.whisper_model.transcribe(audio=audio_path, fp16=False)
        else:
            resul_transcribe = {"segments": []}


        transcribe_segments = [[segment["start"], segment["end"], segment["text"]] for segment in resul_transcribe["segments"]]
        split_scenes = detect(video_path, ContentDetector(threshold=50.0, min_scene_len= int(fps*2)))
        response = {}
        for i, scene in enumerate(split_scenes):
            audio_text = "".join([segment[2] if self.range_float(scene[0].get_seconds(), scene[1].get_seconds(),
                                                    segment[0],segment[1]) else "" for segment in transcribe_segments])
            audio_text = audio_text.strip()
            subclip = video_clip.subclipped(scene[0].get_seconds(), min(scene[1].get_seconds(),video_clip.duration))
            subclip = subclip.without_audio()
            length_video = scene[1].get_seconds() - scene[0].get_seconds()
            num_frame = int(length_video * fps/150) if int(length_video * fps/150)>3 else  3
            print ("num_frame {}/{} in subclip: ".format(i, len(split_scenes)), int(length_video * fps/150),  num_frame)
            frame_text = self.frame_encoder(subclip, num_frame)
            response[i]={
                "start_time": scene[0].get_seconds(),
                "end_time": scene[1].get_seconds(),
                "audio_text": audio_text,
                "frame_text": frame_text
            }
        shutil.rmtree(save_path)
        return response

    def frame_encoder(self, subclip, num_frame):
        frames_list = list(subclip.iter_frames())
        if not frames_list:
            return ""

        key_frames_list = self.frame_sampler(frames_list, k=num_frame)
        result_text= []
        for j, key_frames in enumerate(key_frames_list):
            frame_rgb = cv.cvtColor(key_frames, cv.COLOR_BGR2RGB)
            pil_image = Image.fromarray(np.asarray(frame_rgb))
            image = self.vis_processors["eval"](pil_image).unsqueeze(0).to(device)
            base_cap =  self.blip_model.generate({"image": image})[0]
            query = f"""You are an AI assistant specialized in video summarization. Your task is to update the summary based on the flow of events.
                [PREVIOUS CONTEXT]
                I have analyzed the previous {j} keyframes. The summary of events up to this point is:
                {result_text}
                [NEW INFORMATION]
                The basic content of the current frame is:
                {base_cap}
                [INSTRUCTION]
                Based on the previous context and integrating the new information from this frame, please summarize the video's narrative concisely, logically, and seamlessly."""

            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='JPEG')
            img_bytes = img_byte_arr.getvalue()
            result_text = self.llava_client.chat_vision.remote(query, img_bytes)
            print("frames: ", j)
        return result_text


    def frame_sampler(self, frames_list, k=10):
        if not frames_list:
            return []

        right_frame = frames_list[0]
        shape_hight = 224
        shape_width = 224

        list_mse = []

        for i, frame in enumerate(frames_list):
            frame_a = np.resize(right_frame, (shape_hight, shape_width, 3))
            frame_b = np.resize(frame, (shape_hight, shape_width, 3))
            mse = self.mean_square_error(frame_a, frame_b)
            list_mse.append(mse)

        list_mse[0] = max(list_mse)

        array_mse = np.array(list_mse)

        top_indices = np.argsort(array_mse)[-k:][::-1]
        top_indices = sorted(index for index in top_indices.tolist())

        key_frame = [frames_list[index] for index in top_indices]

        return key_frame

    def mean_square_error(self, frame_a, frame_b):
        if not frame_a.shape == frame_b.shape:
            return 99999
        sum_square = np.sum((frame_a.astype("float") - frame_b.astype("float")) ** 2)
        mean_square_error = sum_square / float(frame_a.shape[0] * frame_a.shape[1] * frame_a.shape[2])

        return mean_square_error

    def range_float(self, start, end, second_start, second_end):
        second_center = (second_start+ second_end)/2
        in_left = start<=second_start and second_center<=end
        in_right= start<=second_start and second_center<=end
        return in_left or in_right


@app.local_entrypoint()
def main():
    encoder = Video_encoder()

    # video_path = "dataset/MMBench-Video/unwrap/video/G4DPefY6-NM.mp4"
    # encoder = encoder.video_encoder(video_path)
    # pprint(encoder)

    path = "dataset/MMBench-Video/unwrap/video/"
    lisst_dir = os.listdir(path)
    text_document_path = "document.json"
    if not os.path.exists(text_document_path):
        with open(text_document_path, "w", encoding="utf-8") as f:
            f.write("[]")

    for i, dir in enumerate(lisst_dir):
        if i<=262:
            continue
        print(("Video {}/{} tên {}".format(i+1, len(lisst_dir), dir)))
        video_path = os.path.join(path, dir)
        encoder_result = encoder.video_encoder(video_path)

        data = {
            "index": i,
            "name": dir,
            "encoder": encoder_result
        }

        with open(text_document_path, "r+", encoding="utf-8") as file:
            old_data = json.load(file)
            old_data.append(data)

            f.seek(0)
            json.dump(old_data, f, ensure_ascii=False, indent=2)
            f.truncate()
            print("Save sucessful")




if __name__ == "__main__":
    main()


# 17