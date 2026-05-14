---
license: cc-by-4.0
task_categories:
- visual-question-answering
language:
- en
modalities:
- Video
- Text
tags:
- video understanding
- evaluation
- large vision-language model
size_categories:
- 1K<n<10K
---
# MMBench-Video: A Long-Form Multi-Shot Benchmark for Holistic Video Understanding

- **Homepage:** [https://mmbench-video.github.io/](https://mmbench-video.github.io/)
- **Repository:** [https://huggingface.co/datasets/opencompass/MMBench-Video](https://huggingface.co/datasets/opencompass/MMBench-Video)
- **Paper:** [MMBench-Video: A Long-Form Multi-Shot Benchmark for Holistic Video Understanding](https://arxiv.org/abs/2406.14515).

## Table of Contents

- [MMBench-Video: A Long-Form Multi-Shot Benchmark for Holistic Video Understanding](#mmbench-video-a-long-form-multi-shot-benchmark-for-holistic-video-understanding)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Leaderboard](#leaderboard)
  - [Data](#data)
    - [How to get video data](#how-to-get-video-data)
  - [Citation](#citation)
  - [License](#license)

## Introduction

MMBench-Video is a quantitative benchmark designed to rigorously evaluate LVLMs' proficiency in video understanding.
MMBench-Video incorporates approximately 600 web videos with rich context from YouTube, spanning 16 major categories, including News, Sports, etc., covering most video topics people watch in their daily lives. Each video ranges in duration from 30 secs to 6 mins, to accommodate the evaluation of video understanding capabilities on longer videos. The benchmark
includes roughly 2,000 original question-answer (QA) pairs, contributed by volunteers, covering a total of 26 fine-grained capabilities. And it also implement a GPT-4-based evaluation paradigm, which offers superior accuracy, consistency, and a closer alignment with human judgments.

## Leaderboard

Latest leaderboard is in our [openvlm_video_leaderboard](https://huggingface.co/spaces/opencompass/openvlm_video_leaderboard).

## Data

The dataset includes 1,998 question-answer (QA) pairs, with each QA assessing one or multiple capabilities of a vision-language model. Each question in the dataset is a question-answer questions with groundtruth.

Here is a example:

```
    index: 177	
    video: DmUgQzu3Z4U	
    video_type: Food & Drink	
    question: Did the mint-style guy in the video drink his mouthwash?	
    answer: Yes, he drank it. This is very strange. Under normal circumstances we are not allowed to drink mouthwash, but this boy may be doing it to attract viewers.	
    dimensions: ['Counterfactual Reasoning']	
    video_path: ./video/DmUgQzu3Z4U.mp4
```

### How to get video data

Using this function to unwrap pkl files to get original video data.

```python
def unwrap_hf_pkl(pth, suffix='.mp4'):
    base_dir = os.path.join(pth, 'video_pkl/')
    target_dir = os.path.join(pth, 'video/')
    pickle_files = [os.path.join(base_dir, file) for file in os.listdir(base_dir)]
    pickle_files.sort()

    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        for pickle_file in pickle_files:
            with open(pickle_file, 'rb') as file:
                video_data = pickle.load(file)
            # For each video file in the pickle file, write its contents to a new mp4 file
            for video_name, video_content in video_data.items():
                output_path = os.path.join(target_dir, f'{video_name}{suffix}')
                with open(output_path, 'wb') as output_file:
                    output_file.write(video_content)
        print('The video file has been restored and stored from the pickle file.')
    else:
        print('The video file already exists.')
```

For full dataset evaluation, you can use [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) to use MMBench-Video with single command.

```bash
python run.py --model GPT4o --data MMBench-Video --nframe 8 --verbose
```

## Citation

```
@misc{fang2024mmbenchvideolongformmultishotbenchmark,
      title={MMBench-Video: A Long-Form Multi-Shot Benchmark for Holistic Video Understanding}, 
      author={Xinyu Fang and Kangrui Mao and Haodong Duan and Xiangyu Zhao and Yining Li and Dahua Lin and Kai Chen},
      year={2024},
      eprint={2406.14515},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.14515}, 
}
```

If you using VLMEvalKit for model evaluation, please cite this:

```
@misc{duan2024vlmevalkitopensourcetoolkitevaluating,
      title={VLMEvalKit: An Open-Source Toolkit for Evaluating Large Multi-Modality Models}, 
      author={Haodong Duan and Junming Yang and Yuxuan Qiao and Xinyu Fang and Lin Chen and Yuan Liu and Amit Agarwal and Zhe Chen and Mo Li and Yubo Ma and Hailong Sun and Xiangyu Zhao and Junbo Cui and Xiaoyi Dong and Yuhang Zang and Pan Zhang and Jiaqi Wang and Dahua Lin and Kai Chen},
      year={2024},
      eprint={2407.11691},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.11691}, 
}
```

## License

The MMBench-Video dataset is licensed under a
[Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
