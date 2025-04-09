# MDK12EvalHub

[**🏗️ Handbook**](docs/Quickstart.md) | [**🛠️ Config Guide**](#-config) | [**📝 Report**](https://arxiv.org/abs/2504.05782) | [**🖊️ Citation**](#-citation)

**MDK12EvalHub** is an **open-source evaluation toolkit** of and **multimodal large language models (MLLMs)**. It enables **one-command evaluation** on MDK12-Bench. In MDK12EvalHub, we adopt **generation-based evaluation** for all MLLMs, and provide the evaluation results obtained with both **exact matching** and **LLM-as-a-Judge**.

## 🏗️ Document

See [Handbook](docs/Quickstart.md) for a quick start guide.

Other documents that facilitate the further optional exploratory development are provided in `docs`. For example, for developers interested in contributing to the code of framework, please check [Development Guide](MDK12EvalHub/docs/Development.md).

<a id="-config"></a>
## 🔧 Config

**Config Tips**

- `pip install -r requirements.txt` to install the necessary environment.

- `pip install -e .` to install the vlmeval (with major changes in `dataset` and `vlm/qwen2_vl` compared with the original vlmeval).

- Please see [Document](docs/Quickstart.md) before the experiment.

- Please set up your evaluation condition following this script: `run_example.sh`

- Change the path of model in `vlmeval/config.py`.

- Use the global variable `LMUData` to set the data root.

**Transformers Version Recommendation:**

Note that some MLLMs may not be able to run under certain transformer versions, we recommend the following settings to evaluate each MLLM:

- **Please use** `transformers==4.33.0` **for**: `Qwen series`, `Monkey series`, `InternLM-XComposer Series`, `mPLUG-Owl2`, `OpenFlamingo v2`, `IDEFICS series`, `VisualGLM`, `MMAlaya`, `ShareCaptioner`, `MiniGPT-4 series`, `InstructBLIP series`, `PandaGPT`, `VXVERSE`.
- **Please use** `transformers==4.36.2` **for**: `Moondream1`.
- **Please use** `transformers==4.37.0` **for**: `LLaVA series`, `ShareGPT4V series`, `TransCore-M`, `LLaVA (XTuner)`, `CogMLLM Series`, `EMU2 Series`, `Yi-VL Series`, `MiniCPM-[V1/V2]`, `OmniLMM-12B`, `DeepSeek-VL series`, `InternVL series`, `Cambrian Series`, `VILA Series`, `Llama-3-MixSenseV1_1`, `Parrot-7B`, `PLLaVA Series`.
- **Please use** `transformers==4.40.0` **for**: `IDEFICS2`, `Bunny-Llama3`, `MiniCPM-Llama3-V2.5`, `360VL-70B`, `Phi-3-Vision`, `WeMM`.
- **Please use** `transformers==4.44.0` **for**: `Moondream2`, `H2OVL series`.
- **Please use** `transformers==4.45.0` **for**: `Aria`.
- **Please use** `transformers==latest` **for**: `LLaVA-Next series`, `PaliGemma-3B`, `Chameleon series`, `Video-LLaVA-7B-HF`, `Ovis series`, `Mantis series`, `MiniCPM-V2.6`, `OmChat-v2.0-13B-sinlge-beta`, `Idefics-3`, `GLM-4v-9B`, `VideoChat2-HD`, `RBDash_72b`, `Llama-3.2 series`, `Kosmos series`.

**Torchvision Version Recommendation:**

Note that some MLLMs may not be able to run under certain torchvision versions, we recommend the following settings to evaluate each MLLM:

- **Please use** `torchvision>=0.16` **for**: `Moondream series` and `Aria`

**Flash-attn Version Recommendation:**

Note that some MLLMs may not be able to run under certain flash-attention versions, we recommend the following settings to evaluate each MLLM:

- **Please use** `pip install flash-attn --no-build-isolation`

<a id="-citation"></a>
## 🖊️ Citation 
If you feel MDK12 useful in your project or research, please kindly use the following BibTeX entry to cite our paper. Thanks!
```bibtex
@misc{zhou2025mdk12,
      title={MDK12-Bench: A Multi-Discipline Benchmark for Evaluating Reasoning in Multimodal Large Language Models}, 
      author={Pengfei Zhou and Fanrui Zhang and Xiaopeng Peng and Zhaopan Xu and Jiaxin Ai and Yansheng Qiu and Chuanhao Li and Zhen Li and Ming Li and Yukang Feng and Jianwen Sun and Haoquan Zhang and Zizhen Li and Xiaofeng Mao and Wangbo Zhao and Kai Wang and Xiaojun Chang and Wenqi Shao and Yang You and Kaipeng Zhang},
      year={2025},
      eprint={2504.05782},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.05782}, 
}
