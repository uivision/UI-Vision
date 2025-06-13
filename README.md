# UI-Vision: A Desktop-centric GUI Benchmark for Visual Perception and Interaction

[![arXiv](https://img.shields.io/badge/arXiv-2503.15661-b31b1b.svg)](https://arxiv.org/abs/2503.15661)
[![Website](https://img.shields.io/badge/Website-UI--Vision-blue)](https://uivision.github.io/)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-UI--Vision-yellow)](https://huggingface.co/datasets/ServiceNow/ui-vision)

## 📢 News
- **[15 May 2025]** UI-Vision [grounding dataset](https://huggingface.co/datasets/ServiceNow/ui-vision) (Element and Layout Grounding) and evaluation code released
- **[1 May 2025]** UI-Vision got accepted to ICML 2025 🔥
- **[19 March 2025]** Project website is live at [uivision.github.io](https://uivision.github.io/)
- **[19 March 2025]** UI-Vision paper is available on [arXiv](https://arxiv.org/abs/2503.15661) 🔥 🔥 


## Introduction
UI-Vision is a comprehensive, license-permissive benchmark for offline, fine-grained evaluation of computer use agents in real-world desktop environments across 83 software applications spanning 6 categories. The benchmark includes three tasks:

- Element Grounding
- Layout Grounding  
- Action Prediction

The benchmark aims to advance the development of more capable agents for real-world desktop tasks.

## Evaluation
### Element Grounding

| Model                 | Basic Overall | Functional Overall | Spatial Overall | Final Avg |
|-----------------------|--------------:|-------------------:|----------------:|----------:|
| **Closed-Source VLMs**|               |                    |                 |           |
| GPT-4o                |          1.58 |               1.52 |            1.03 |      1.38 |
| Gemini-1.5-pro        |          0.79 |               0.28 |            0.57 |      0.55 |
| Gemini-Flash-2.0      |          0.45 |               0.40 |            0.05 |      0.30 |
| Claude-3.5-Sonnet     |          5.08 |               5.19 |            3.15 |      4.47 |
| Claude-3.7-Sonnet     |          9.48 |               7.73 |            7.60 |      8.27 |
| **Open-Source VLMs**  |               |                    |                 |           |
| Qwen-2.5VL-7B         |          1.24 |               0.79 |            0.51 |      0.85 |
| InternVL2-8B          |          0.11 |               0.11 |            0.00 |      0.09 |
| InternVL2.5-8B        |          2.48 |               2.82 |            0.98 |      2.09 |
| Qwen-2VL-7B           |          3.44 |               3.22 |            1.45 |      2.70 |
| MiniCPM-V-8B          |          7.11 |               5.30 |            3.57 |      4.34 |
| **Open-Source GUI Agents**|           |                    |                 |           |
| ShowUI-2B             |          8.07 |               7.67 |            2.07 |      5.94 |
| AriaUI-25.3B          |         12.20 |              14.00 |            3.98 |     10.10 |
| UGround-v1-7B         |         15.40 |              17.10 |            6.25 |     12.90 |
| OSAtlas-7B            |         12.20 |              11.20 |            3.67 |      9.02 |
| UGround-7B            |         11.50 |              12.20 |            2.79 |      8.83 |
| Aguvis-7B             |         17.80 |              18.30 |            5.06 |     13.70 |
| UI-TARS-7B            |         20.10 |              24.30 |            8.37 |     17.60 |
| CogAgent-9B           |         12.00 |              12.20 |            2.63 |      8.94 |
| SeeClick-9.6B         |          9.42 |               4.68 |            2.07 |      5.39 |
| UGround-v1-72B        |         27.90 |              26.70 |           14.90 |     23.20 |
| UI-TARS-72B           |         31.40 |              30.50 |           14.70 |     25.50 |
| TongUI-3B             |         22.40 |              17.40 |            6.50 |     15.43 |
| TongUI-7B             |         24.40 |              22.50 |            7.20 |     18.03 |
| Jedi-3B               |         22.29 |              25.23 |            9.35 |     18.96 |
| Jedi-7B               |         32.34 |              30.47 |           12.76 |     25.19 |

### Layout Grounding

| Model               | IoU ↑ | Precision ↑ | Recall ↑ |
|---------------------|------:|------------:|---------:|
| **Closed-Source VLMs**|||||
| GPT-4o             |  20.0 |        59.6 |     24.1 |
| Claude-3.5-Sonnet  |  22.4 |        64.3 |     26.8 |
| Claude-3.7-Sonnet  |  17.6 |        31.5 |     34.1 |
| Gemini-1.5-pro     |  30.8 |        67.8 |     36.9 |
| Gemini-2.0-flash   |  28.3 |        63.0 |     34.2 |
| **Open-Source VLMs**|||||
| Qwen-2VL-7B        |  24.3 |        65.7 |     33.4 |
| MiniCPM-V-8B       |  16.3 |        25.7 |     43.6 |
| **Open-Source GUI Agents**|||||
| CogAgent-9B        |   6.22|         7.99|     42.9 |
| SeeClick-9.6B      |   5.11|         6.32|     30.1 |
| OSAtlas-7B         |  28.2 |        66.4 |     41.6 |

### Action Prediction

| Model             | Click/Move Dist. ↓ | Click/Move Recall@d ↑ | Drag Dist. ↓ | Drag Recall@d ↑ | Typing Corr. ↑ | Hotkey Corr. ↑ | SSR ↑  |
|-------------------|-------------------:|----------------------:|-------------:|----------------:|---------------:|---------------:|-------:|
| **Naive Baselines**|||||||||
| Random            |               81.6 |                  0.0  |         94.2 |            0.0  | N/A            | N/A            | N/A    |
| GPT-4o w/o image  |               52.0 |                  3.3  |         72.4 |            0.0  | 22.7           | 34.0           | 7.64   |
| **Closed-Source VLMs**|||||||||
| GPT-4o            |               41.2 |                  4.4  |         63.9 |            1.5  | 32.1           | 56.5           | 11.5   |
| Gemini-1.5-Pro    |               38.7 |                 13.0  |         61.1 |            1.6  | 24.7           | 45.3           | 16.0   |
| Claude-3.5-Sonnet |               41.0 |                  4.8  |         61.4 |            1.1  | 29.0           | 39.2           | 9.9    |
| **Open-Source GUI Agents**|||||||||
| ShowUI-2B         |               42.8 |                 11.8  | N/A          | N/A             | 15.2           | 62.5           | 15.7   |
| UI-TARS-7B        |               47.0 |                 19.7  |         64.8 |            3.1  | 33.8           | 40.5           | 21.4   |


## Repository Structure
```
├── eval/
│   └── grounding/   # Scripts for element and layout grounding evaluation
│   └── action_prediction/   # Scripts for action prediction evaluation
├── .gitignore          # Git ignore file
└── README.md           # Project documentation
```

## Citation

If you find UI-Vision useful in your research, please consider citing our paper:

```bibtex
@misc{nayak2025uivisiondesktopcentricguibenchmark,
  title={UI-Vision: A Desktop-centric GUI Benchmark for Visual Perception and Interaction},
  author={Shravan Nayak and Xiangru Jian and Kevin Qinghong Lin and Juan A. Rodriguez and
  Montek Kalsi and Rabiul Awal and Nicolas Chapados and M. Tamer Özsu and
  Aishwarya Agrawal and David Vazquez and Christopher Pal and Perouz Taslakian and
  Spandana Gella and Sai Rajeswar},
  year={2025},
  eprint={2503.15661},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2503.15661},
}
```
