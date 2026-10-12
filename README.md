# LoRA-DermaSks: LoRA Fine-Tuning for Synthetic Dermoscopic Image Generation

## Overview

**LoRA-DermaSks-Project** explores parameter-efficient fine-tuning of Stable Diffusion 1.5 for generating synthetic dermoscopic images conditioned on clinical descriptions.

The project uses clinical metadata associated with skin-lesion images to construct descriptive text prompts. These prompts are used to fine-tune a pretrained text-to-image diffusion model through **Low-Rank Adaptation (LoRA)**. The resulting adapter is then used to generate synthetic dermoscopic images and analyze their visual characteristics.

The workflow also includes training-log analysis, quantitative image comparisons, and visualization of image statistics.

## Objectives

- Fine-tune a pretrained text-to-image diffusion model using LoRA.
- Incorporate clinical metadata into text-conditioned dermoscopic image generation.
- Generate synthetic skin-lesion images from clinical descriptions.
- Evaluate synthetic images using image-distribution and structural similarity metrics.
- Analyze training convergence and image characteristics through visualizations.
- Explore synthetic image generation as a potential resource for medical imaging research.

## Key Features

- **Clinical metadata processing:** Convert structured CSV records into image-caption pairs.
- **LoRA fine-tuning:** Adapt Stable Diffusion 1.5 using a trainable low-rank adapter.
- **Checkpoint management:** Resume training from saved checkpoints when available.
- **Clinical prompt generation:** Generate dermoscopic images using descriptions of lesion characteristics, anatomical location, age, sex, and other available metadata.
- **Image evaluation:** Calculate Fréchet Inception Distance (FID) and Structural Similarity Index Measure (SSIM).
- **Training analysis:** Extract TensorBoard logs and visualize loss and training progression.
- **Visual analysis:** Produce intensity histograms, simulated feature heatmaps, edge-complexity distributions, and color-channel distributions.
- **Automated reporting:** Save evaluation summaries and visualizations to Google Drive.

## Technologies Used

- Python
- PyTorch
- Hugging Face Diffusers
- Hugging Face Transformers
- Accelerate
- Pandas and NumPy
- Matplotlib
- SciPy
- scikit-image
- Torchvision
- OpenCV
- TensorBoard
- Google Colab and Google Drive

## Model and Methodology

### 1. Clinical Metadata Preparation

The project reads clinical information from `metadata.csv` and associates each record with its corresponding image using the `isic_id` field.

For images that exist in the dataset directory, it constructs a text description using fields such as:

- Image type
- Melanocytic status
- Diagnosis
- Anatomical location
- Approximate patient age
- Sex
- Dermoscopic features
- Diagnostic confirmation
- Concomitant biopsy information

Missing CSV values are replaced with `unspecified`, and the generated captions are converted to lowercase.

The metadata is saved in JSON Lines format as `metadata.jsonl`.

Each record follows this structure:

```json
{"file_name": "ISIC_example.jpg", "text": "dermoscopic photo of a ..."}
```

The example above illustrates the format; the actual filename and caption depend on the dataset.

### 2. LoRA Fine-Tuning

The project fine-tunes the pretrained `runwayml/stable-diffusion-v1-5` model using the Hugging Face Diffusers training script, `train_text_to_image_lora.py`.

Rather than updating the entire model, LoRA introduces trainable low-rank updates to selected model layers, reducing the number of trainable parameters.

The main training configuration is:

| Parameter | Value |
|---|---|
| Base model | Stable Diffusion 1.5 |
| Training resolution | 512 × 512 |
| LoRA rank | 16 |
| Training batch size | 1 |
| Gradient accumulation steps | 4 |
| Maximum training steps | 1,000 |
| Learning rate | 0.0002 |
| Learning-rate scheduler | Cosine |
| Random seed | 42 |
| Checkpoint interval | Every 500 steps |
| Logging backend | TensorBoard |

Training checkpoints are saved in the output directory. If checkpoints are detected, the script selects the latest checkpoint by its numeric step identifier and passes it to the training script for resumption.

### 3. Synthetic Image Generation

After loading the base model and available LoRA weights, the pipeline generates images from clinical text prompts.

The inference configuration includes:

- DPM-Solver multistep scheduler
- 35 inference steps for the initial five-prompt generation batch
- Guidance scale of 7.5
- LoRA inference scale of 0.85
- A negative prompt to discourage blurry images, watermarks, unwanted text, and other specified features

Generated images are saved as PNG files under `generated_samples/`.

The current examples focus on melanoma-related descriptions and vary selected contextual attributes, such as anatomical location, age, sex, and dermoscopic acquisition characteristics.

**Important:** These prompts demonstrate the implemented generation workflow. They do not establish that the model generates clinically valid, diagnostically accurate, or representative melanoma images.

## Evaluation Methodology

The project implements quantitative and qualitative analyses of generated images.

### Quantitative Metrics

**Fréchet Inception Distance (FID)**

FID compares feature distributions extracted from real and synthetic images using an Inception network. Lower FID values generally indicate greater similarity between the two feature distributions when the metric is implemented and evaluated appropriately.

The implementation extracts features from resized images and calculates a Fréchet-distance estimate using their means and covariance matrices.

**Structural Similarity Index Measure (SSIM)**

SSIM compares luminance, contrast, and structural patterns between two images. Higher values indicate greater structural similarity for the particular image pairs being compared.

The current implementation compares generated images against a selected real image, rather than establishing paired ground-truth correspondences for each generated sample.

### Qualitative and Statistical Analysis

The project also generates visualizations for examining image characteristics:

| Analysis | Output |
|---|---|
| Simulated feature heatmaps | `feature_heatmaps.png` |
| Real-versus-synthetic intensity histograms | `intensity_histogram.png` |
| SSIM trend visualization | `ssim_trend.png` |
| Evaluation summary | `summary_report.txt` |
| Resized evaluation report | `detailed_metrics.txt` |
| LoRA training report | `lora_report_no_lr.png` |
| Merged training report | `merged_lora_report.png` |
| Edge-complexity distribution | `lora_edge_complexity.png` |
| Color-channel distribution | `lora_color_fidelity.png` |

The two evaluation workflows use different sample sizes: an initial batch of 8 images and a second batch of 20 images.

The heatmaps in the current code are calculated from a sine transformation of grayscale pixel intensities. They are illustrative texture maps, not genuine model-attention maps or validated clinical saliency maps.

## Project Structure

The following structure summarizes the intended Google Drive organization and the outputs produced by the supplied code.

```text
LoRA_DermaSks_Project/
├── metadata.csv
├── dataset/
│   ├── metadata.jsonl
│   ├── ISIC_example_1.jpg
│   └── ...
├── output/
│   ├── checkpoint-500/
│   ├── checkpoint-1000/
│   ├── pytorch_lora_weights.safetensors
│   ├── lora_report_no_lr.png
│   ├── merged_lora_report.png
│   └── logs/
├── generated_samples/
│   ├── clinical_batch_0.png
│   ├── clinical_batch_1.png
│   └── ...
├── evaluation_results/
│   ├── feature_heatmaps.png
│   ├── intensity_histogram.png
│   ├── ssim_trend.png
│   └── summary_report.txt
└── evaluation_results_resize/
    ├── feature_heatmaps.png
    ├── intensity_histogram.png
    ├── ssim_trend.png
    └── detailed_metrics.txt
```

*Note: This is a logical representation of expected outputs, not a guarantee that every listed file exists. Checkpoint contents, exported LoRA weights, and report filenames depend on successful execution of the corresponding cells.*

## Getting Started

### Prerequisites

- A Google account with Google Colab access
- A CUDA-enabled GPU runtime
- Access to the dataset images and `metadata.csv`
- Sufficient Google Drive storage for images, checkpoints, and generated outputs

### 1. Open Google Colab

Open the project notebook in Google Colab and select a GPU runtime.

### 2. Mount Google Drive

Run the following code:

```python
from google.colab import drive
drive.mount("/content/drive")
```

### 3. Configure the Project Paths

```python
PROJECT_DIR = "/content/drive/MyDrive/LoRA_DermaSks_Project"
DATA_DIR = f"{PROJECT_DIR}/dataset"
OUTPUT_DIR = f"{PROJECT_DIR}/output"
```

Create the directories if necessary:

```python
import os

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
```

### 4. Prepare the Dataset

Place `metadata.csv` in the project directory and ensure that the corresponding skin-lesion images are available in `dataset/`.

Run the metadata-generation cell to create `dataset/metadata.jsonl`.

The metadata-generation function includes only records for which the expected image file exists.

### 5. Install Dependencies

The notebook installs the primary training dependencies using:

```python
!pip install -q datasets transformers accelerate ftfy bitsandbytes xformers
!pip install -q git+https://github.com/huggingface/diffusers
```

Additional dependencies used in the evaluation and visualization cells include:

```python
!pip install -q pandas numpy scipy scikit-image matplotlib seaborn opencv-python tensorboard
```

Use compatible package versions for the selected Colab runtime.

### 6. Train the LoRA Adapter

Execute the training cell to launch `train_text_to_image_lora.py` with the configured hyperparameters.

Training logs are directed to TensorBoard, and checkpoints are saved in `output/`.

### 7. Generate Synthetic Images

Load the Stable Diffusion pipeline, load the exported LoRA weights when available, and execute the prompt-generation cell.

Generated samples are saved to:

```text
LoRA_DermaSks_Project/generated_samples/
```

### 8. Evaluate the Results

Execute the evaluation cells to calculate FID and SSIM estimates and create the associated plots and text reports.

Run the training-log analysis cells to inspect loss progression and generate training summaries.

## Reproducibility and Limitations

Several considerations are important when interpreting the current implementation:

- **Dataset availability:** The number of usable records depends on image availability, metadata completeness, and matching filenames.
- **Checkpoint resumption:** The code detects existing checkpoints, but resumed training should be verified to ensure that the intended training state and configuration are restored.
- **LoRA weight export:** The inference code expects `pytorch_lora_weights.safetensors`. Ensure that weights are exported to the expected location before inference.
- **FID sample size:** The initial evaluation uses only eight generated images and eight real images. The resized evaluation uses up to twenty of each. Such small sample sizes make FID estimates unreliable for drawing strong conclusions.
- **FID implementation:** The supplied implementation uses the Inception classification output as features, rather than the conventional intermediate Inception feature representation commonly used for FID. Its values should therefore not be treated as directly comparable with standard FID benchmark results.
- **SSIM comparison:** Comparing unrelated generated images with the same real reference does not establish image-level reconstruction quality.
- **Clinical validity:** Visual similarity and image statistics do not establish diagnostic correctness, clinical utility, or safety.

Future work could include larger evaluation cohorts, standardized FID computation, expert dermatological review, dataset-level diversity assessment, and validation of whether synthetic samples preserve clinically relevant features.

## Responsible Use

This project is intended for research and experimentation in medical image generation. Synthetic images are not substitutes for clinical examination, diagnosis, or medical advice.

Generated images should be clearly identified as synthetic and independently evaluated before being considered for use in medical research, training, or decision-support systems.

## Author

**Laiba Imran**

GitHub: [@LaibaImran1500](https://github.com/LaibaImran1500)

## License

Refer to the repository's license information and the applicable terms for the underlying dataset, pretrained Stable Diffusion model, and software dependencies.
