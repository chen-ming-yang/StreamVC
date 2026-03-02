# StreamVC
An unofficial pytorch implementation of [STREAMVC: REAL-TIME LOW-LATENCY VOICE CONVERSION](https://arxiv.org/pdf/2401.03078.pdf) created for learning purposes.

This is not an official product, and while the code should work, I don't have a trained model checkpoint to share.
If you successfully trained the model, I encourage you to share it on Hugging Face, and ping me so I can link it here.

The streaming inference as it is in the paper isn't fully implemented, and I have no plans to implement it.

```mermaid
flowchart LR 
    TS[Training Sample] -.-> SP
    SP -.-> HB[["(8) HubBert based\npseudo labels"]]
    CE -.-> LN[[LN+Linear+Softmax]] -.->L1((cross\nentropy\nloss))
    HB -.->L1
    subgraph Online Inference
        SP[Source\nSpeech] --> CE[["(1) Content\nEncoder"]] -->|"// (grad-stop)"| CL[Content\nLatent] --> CAT(( )) -->D[["(3) Decoder"]]
        SP --> f0[["(4) f0 estimation"]] --> f0y[["(5) f0 whitening"]] --> CAT
        SP --> FE[["(6) Frame Energy\nEstimation"]] --> CAT
    end
    subgraph "Offline Inference (Target Speaker)"
        TP[Target\nSpeech] --> SE[["(2) Speech\nEncoder"]] --- LP[["(7) Learnable\nPooling"]]-->SL[Speaker\nLatent]
    end
    TS -.-> TP
    SL --> |Conditioning| D
    D -.-> Dis[["(9) Discriminator"]]
    TS2[Training Sample] -.->Dis
    Dis -.-> L2((adversarial\nloss))
    Dis -.-> L3((feature\nloss))
    D -.-> L4((reconstruction\nloss))
    TS2 -.-> L4 
    classDef train fill:#337,color:#ccc;
    class TS,TS2,HB,LN,Dis train;
    classDef off fill:#733,color:#ccc;
    class TP,SE,LP,SL off;
    classDef else fill:#373,color:#ccc;
    class SP,CE,CL,D,f0,f0y,FE,CAT else;
```

## Training Flow (High-Level)
Training is performed in a single unified stage. In each step, three phases are executed sequentially:

1. **Content Encoder** — classification loss (cross-entropy with HuBERT pseudo labels)
2. **Generator (Decoder + Speech Encoder)** — adversarial + feature matching + reconstruction loss
3. **Discriminator** — adversarial loss

```mermaid
flowchart LR
    subgraph Unified Training
        A[Audio Batch] --> CE[Content Encoder]
        CE --> H[LayerNorm + Dropout + Linear]
        H --> L[Class Logits]
        HLBL[HuBERT Labels] --> CE_L((Cross-Entropy\nLoss))
        L --> CE_L

        A --> CE2[Content Encoder] --> CL[Content Latent]
        A --> F0[f0 Estimator] --> F0W[f0 whitened]
        A --> EN[Energy Estimator] --> ENF[Energy]
        CL --> CAT((Concat))
        F0W --> CAT
        ENF --> CAT
        A --> SE[Speech Encoder] --> LP[Learnable Pooling] --> SL[Speaker Latent]
        CAT --> DEC[Decoder]
        SL -->|Conditioning| DEC
        DEC --> G[Generated Audio]
        A --> R[Real Audio]
        G --> D[Multi-Scale Discriminator]
        R --> D
        D --> ADV_L((Adversarial Loss))
        D --> FEAT_L((Feature Matching Loss))
        R --> REC_L((Reconstruction Loss))
        G --> REC_L
        ADV_L --> GEN_L((Generator Loss))
        FEAT_L --> GEN_L
        REC_L --> GEN_L
    end
```

## Example Usage
### Training
#### Requirements
To install the requirements for training run:
```bash
pip install -r requirements-training.txt
```
#### Preprocess the datasets
`preprocess_dataset.py` is the python script for dataset preprocessing.
This script downloads the specified LibriTTS split, compress it locally into `ogg` files,
and creates the HuBert labels for it.
To  view the dataset and see the available splits, go to [mythicinfinity/libritts](https://huggingface.co/datasets/mythicinfinity/libritts).
To launch the script, run:
```bash
python preprocess_dataset.py --split [SPLIT-NAME]
```
It is recommended to download all the train splits as well as the clean dev & test at least.
To see additional available option:
```bash
python preprocess_dataset.py --help
```

#### Running the training script

The training of StreamVC is done in a single unified stage where the content encoder, generator (decoder + speech encoder), and discriminator are all trained jointly. 

An example of launching the training script:
```bash
accelerate launch \
    train.py \
    --run-name myrun_228 \
    --batch_size 24 \
    --num-epochs 30 \
    --lr 1e-4 \
    --datasets.train-dataset-path "./dataset/train-clean-100" "./dataset/train-clean-360" \
    --model-checkpoint-interval 500 \
    --log-gradient-interval 500 \
    lr-scheduler:cosine-annealing-warm-restarts \
    --lr-scheduler.T-0 3000
```
### Inference
#### Requirements
To install the requirements for inference run:
```bash
pip install -r requirements-inference.txt
```
#### Running the script
 `inference.py` is the python script for inference on a single source & target combo.


To launch the script, run:
```bash
python inference.py -c <model_checkpoint> -s <source_speech> -t <target_speech> -o <output_file>
```
For eaxmaple
```bash
python inference.py \
    -c /root/autodl-tmp/cmy/StreamVC/checkpoints/myrun_228_state_epoch8/pytorch_model_1.bin \
    -s /root/autodl-tmp/cmy/StreamVC/LibriTTS/test-clean/1089/134686/1089_134686_000001_000001.wav \
    -t /root/autodl-tmp/cmy/StreamVC/LibriTTS/test-clean/672/122797/672_122797_000002_000002.wav \
    -o output.wav
```



## Acknowledgements
This project was made possible by the following open source projects:

 - For the encoder-decoder architecture (based on SoundStream) we based our code on [AudioLM's official implementation](https://github.com/lucidrains/audiolm-pytorch).
 - For the multi-scale discriminator and the discriminator losses we based our code on [MelGan's official implementation](https://github.com/descriptinc/melgan-neurips).
 -  For the HuBert discrete units computation we used the HuBert + KMeans implementation from [SoftVC's official implementation](https://github.com/bshall/soft-vc).
 - For the Yin algorithm we based our implementation on the [torch-yin package](https://github.com/brentspell/torch-yin).