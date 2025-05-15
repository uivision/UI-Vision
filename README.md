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
