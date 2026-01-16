# Memories of Malice: Emergent Memory Poisoning Vulnerabilities in Personality-Modified AI Agents

This repository contains the code, configurations, and scripts for the paper **"Memories of Malice: Emergent Memory Poisoning Vulnerabilities in Personality-Modified AI Agents"**.

## Repository Structure

The repository is organized into functional modules:

- `agent-configuration/`: YAML configuration files for the Letta agent and the experiment.
- `email-dataset-setup/`: Scripts to parse, sort, and organize the Enron email dataset.
- `personality-framework/`: Tools for personality analysis and model fine-tuning.
  - `scoring/`: Scripts to score dialogues on Big Five personality traits.
  - `subset-generation/`: Scripts to create personality-specific training subsets (Stability/Plasticity).
  - `finetuning/`: Scripts and configs for fine-tuning Qwen3 models.

## Agent Architecture & Setup

This project uses **[Letta](https://github.com/letta-ai/letta)** (formerly MemGPT) as the agent framework.

### Documentation
- [Letta Documentation](https://docs.letta.com/)

### Configurations
**Path:** `agent-configuration/`
- **Agent Config**: [`agent-configuration/agent-config.yaml`](./agent-configuration/agent-config.yaml) - Defines the agent's memory blocks, tools, and system prompts.
- **Experiment Config**: [`agent-configuration/experiment-config.yaml`](./agent-configuration/experiment-config.yaml) - Defines the parameters for the memory poisoning experiment (email injection, date ranges, etc.).
*Note: Paths in these configs have been sanitized. You will need to update them to match your local environment.*

## Data Preparation

### 1. JIC Dataset Tools (Personality Framework)
We utilize the **Journal Intensive Conversations (JIC) Dataset** for personality fine-tuning.
- **Repository**: [Beyond-Discrete-Personas](https://github.com/Sayantan-world/Beyond-Discrete-Personas)

#### Personality Modules
Located in `personality-framework/`:

1.  **Scoring (`personality-framework/scoring/`)**:
    - Scripts to infer Big Five traits from text (`big5_infer.py`) and analyze distributions.
    
2.  **Subset Generation (`personality-framework/subset-generation/`)**:
    - Scripts to generate training subsets based on Stability and Plasticity meta-traits.
    - `system_prompts.json`: The specific system prompts used for each personality profile.

3.  **Fine-Tuning (`personality-framework/finetuning/`)**:
    - `train_sft.py`: The main supervised fine-tuning script.
    - `train.dgx.yaml`: Configuration for training on DGX infrastructure.

### 2. Enron Email Dataset (Experiment Environment)
The agent operates within a simulated email client populated with real data from the Enron Email Dataset (Vince Kaminski's mailbox).

- **Download**: You can obtain the dataset from [CMU Enron Email Dataset](http://www.cs.cmu.edu/~enron/).

#### Database Setup Instructions
Located in `email-dataset-setup/`:
1.  **Parse Raw Emails**: Use `email-dataset-setup/parse_kaminski_emails.py` to convert raw Enron email files into a JSONL format.
2.  **Sort Emails**: Use `email-dataset-setup/sort_kaminski_emails.py` to sort the parsed emails chronologically.
3.  **Create Database**: Use `email-dataset-setup/create_email_database.py` to ingest the sorted JSONL into a SQLite database (`kaminski_emails.sqlite`).

## Citation

If you use this code or data in your research, please cite our paper:

```bibtex
@article{memoriesofmalice2026,
  title={Memories of Malice: Emergent Memory Poisoning Vulnerabilities in Personality-Modified AI Agents},
  author={Naren and Sikdar, Biplab},
  year={2026}
}
```

## License

This project is licensed under a Strict Academic License (Citation & Permission Required for All Uses) - see the [LICENSE](LICENSE) file for details.
