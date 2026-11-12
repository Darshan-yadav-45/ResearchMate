"""Populates the database with real sample academic research papers, chunks, and citations."""

import os
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import json
from app import create_app, db
from database.models import Paper, PaperChunk, PaperSummary, ReferenceItem, LiteratureReview, Project
from rag.chunker import TextChunker
from rag.vector_store import global_vector_store
from services.reference_formatter import ReferenceFormatterService

app = create_app()

SAMPLE_PAPERS = [
    {
        "title": "Attention Is All You Need",
        "authors": "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin",
        "year": 2017,
        "journal": "Advances in Neural Information Processing Systems (NeurIPS 2017)",
        "doi": "10.5555/3295222.3295349",
        "url": "https://arxiv.org/abs/1706.03762",
        "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.",
        "full_text": """Attention Is All You Need
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin

Abstract:
We propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.

1. Introduction:
Recurrent neural networks, particularly LSTM and GRU, have been established as state of the art approaches in sequence modeling. The inherently sequential nature of recurrence precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples.

2. Model Architecture:
The Transformer follows an encoder-decoder structure. The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers: a multi-head self-attention mechanism, and a simple, position-wise fully connected feed-forward network. We employ residual connections around each of the two sub-layers, followed by layer normalization.
Multi-Head Attention allows the model to jointly attend to information from different representation subspaces at different positions.

3. Results:
On the WMT 2014 English-to-German translation task, the big transformer model achieves 28.4 BLEU, outperforming existing best models by more than 2.0 BLEU. On the WMT 2014 English-to-French task, our model establishes a new single-model state-of-the-art BLEU score of 41.8.

4. Limitations:
A notable limitation of the standard dot-product attention mechanism is its quadratic O(N^2) computational and memory complexity with respect to sequence length N. This quadratic scaling poses severe constraints on deploying Transformers on low-power edge processors and handling very long context sequences.
""",
        "summary": {
            "objectives": "Eliminate recurrent and convolutional dependencies in sequence transduction by introducing an attention-only architecture (the Transformer) to enable full sequence parallelization during training.",
            "abstract": "The authors introduce the Transformer, an architecture that replaces recurrence with multi-head self-attention. Tested on WMT translation tasks, it achieved new state-of-the-art BLEU scores (28.4 BLEU on English-to-German) with vastly superior training efficiency.",
            "methodology": "Encoder-decoder layout with stacked multi-head self-attention and position-wise feedforward networks, residual connections, layer normalization, and sinusoidal positional encodings.",
            "findings": "Surpassed existing RNN and CNN benchmarks in translation tasks while training in only 3.5 days on 8 GPUs. Multi-head attention allows concurrent subspace alignment.",
            "limitations": "Quadratic computational and memory complexity O(N^2) with respect to input sequence length, making long-document and low-power deployment challenging.",
            "future_directions": "Extending attention to non-text modalities (vision, audio), sparse attention patterns to mitigate quadratic memory, and local context self-attention.",
        }
    },
    {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "authors": "Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova",
        "year": 2019,
        "journal": "NAACL-HLT 2019",
        "doi": "10.18653/v1/N19-1423",
        "url": "https://arxiv.org/abs/1810.04805",
        "abstract": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks.",
        "full_text": """BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova

Abstract:
BERT demonstrates that pre-trained language models can be fine-tuned to achieve state-of-the-art results across eleven diverse NLP tasks including GLUE, MultiNLI, and SQuAD without heavy task-specific architecture engineering.

1. Introduction:
Language model pre-training has been shown to be effective for improving many natural language processing tasks. There are two existing strategies: feature-based (such as ELMo) and fine-tuning (such as OpenAI GPT). We argue that current techniques restrict the power of the pre-trained representations because standard language models are unidirectional.

2. BERT Architecture & Pre-training:
BERT's model architecture is a multi-layer bidirectional Transformer encoder. To train deep bidirectional representations, we use two unsupervised tasks: Masked LM (MLM) and Next Sentence Prediction (NSP). In MLM, 15% of input tokens are masked at random and the model predicts their vocabulary IDs.

3. Empirical Evaluation:
BERT achieves remarkable gains: GLUE score of 80.5% (7.7% absolute improvement), MultiNLI accuracy of 86.7%, and SQuAD v1.1 F1 score of 93.2, outperforming human performance.

4. Limitations:
Pre-training is computationally demanding, requiring 64 TPU chips for 4 days for BERT-Large. The static [MASK] token introduces a pre-train versus fine-tune discrepancy because [MASK] never appears during downstream inference. Furthermore, BERT does not generate text autoregressively.
""",
        "summary": {
            "objectives": "Develop deep bidirectional contextual language representations by pre-training Transformers on masked token reconstruction and next-sentence prediction.",
            "abstract": "BERT introduces bidirectional self-attention pre-training. Fine-tuning BERT with simple task-specific layers set new records across 11 NLP benchmarks, including GLUE and SQuAD.",
            "methodology": "Multi-layer bidirectional Transformer encoder pre-trained on BooksCorpus and English Wikipedia with Masked Language Modeling (MLM) and Next Sentence Prediction (NSP).",
            "findings": "BERT-Large achieved 80.5% on GLUE (+7.7% gain) and 93.2 F1 on SQuAD v1.1. Bidirectionality is proven critical over unidirectional left-to-right baselines.",
            "limitations": "Substantial computational training costs, inference latency, static [MASK] token mismatch between training and test distributions, and inability to perform generative text completion.",
            "future_directions": "Autoregressive bidirectional formulation, model distillation (e.g. DistilBERT), and cross-lingual pre-training.",
        }
    },
    {
        "title": "LoRA: Low-Rank Adaptation of Large Language Models",
        "authors": "Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen",
        "year": 2022,
        "journal": "ICLR 2022",
        "doi": "10.48550/arXiv.2106.09685",
        "url": "https://arxiv.org/abs/2106.09685",
        "abstract": "An important paradigm of natural language processing consists of large-scale pre-training on general domain data and adaptation to particular tasks or domains. As we pre-train larger models, full fine-tuning becomes increasingly infeasible. We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks.",
        "full_text": """LoRA: Low-Rank Adaptation of Large Language Models
Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen

Abstract:
LoRA allows an existing pre-trained LLM (like GPT-3 175B) to be adapted with 10,000 times fewer trainable parameters and a 3x reduction in GPU memory requirements, while matching or exceeding full fine-tuning performance.

1. Problem Statement:
Fine-tuning every weight parameter in models with hundreds of billions of parameters requires exorbitant GPU memory for optimizer states and gradients, and creates massive deployment storage bottlenecks when serving independent task checkpoints.

2. Proposed Method:
Inspired by the observation that weight updates during adaptation have a low 'intrinsic dimension', LoRA parameterizes the weight update matrix Delta W as the product of two low-rank matrices B and A: Delta W = B * A, where B in R^(d x r) and A in R^(r x k), with rank r << min(d, k). During inference, Delta W can be merged directly into the base weights with zero additional latency.

3. Results:
On GPT-3 175B, LoRA matches full fine-tuning performance across WikiSQL and MNLI benchmarks while reducing trainable parameters from 175 billion down to just 37.7 million (a 99.98% reduction).

4. Limitations:
LoRA requires choosing hyperparameter rank r. While zero inference latency is possible when weights are merged, serving multiple tasks concurrently with different LoRA adapters in a single batch requires specialized batching kernels (like S-LoRA).
""",
        "summary": {
            "objectives": "Enable parameter-efficient fine-tuning of multi-billion parameter LLMs by decomposing weight updates into low-rank factorization matrices.",
            "abstract": "LoRA freezes pre-trained foundation weights and injects trainable rank decomposition matrices into Transformer attention layers. It achieves comparable accuracy to full fine-tuning while reducing trainable parameters by up to 10,000x.",
            "methodology": "Low-rank matrix decomposition Delta W = B * A, rank r << d, scaling factor alpha, frozen original weights W0 with zero added inference latency upon weight folding.",
            "findings": "Matches full fine-tuning on GPT-3 175B while training only 37.7M parameters. Reduces VRAM usage from 1.2TB to under 350GB on 175B models.",
            "limitations": "Fixed rank selection per layer, potential performance degradation if task intrinsic dimensionality exceeds rank r, and concurrent multi-adapter serving complexity without custom kernels.",
            "future_directions": "Dynamic adaptive rank allocation per layer (AdaLoRA), quantization integration (QLoRA), and multi-task weight blending.",
        }
    }
]

def seed():
    with app.app_context():
        # Clear existing data to avoid duplicates
        PaperChunk.query.delete()
        PaperSummary.query.delete()
        ReferenceItem.query.delete()
        Paper.query.delete()
        LiteratureReview.query.delete()
        db.session.commit()

        chunker = TextChunker(chunk_size=700, chunk_overlap=120)
        ref_service = ReferenceFormatterService()
        store_chunks = []

        for p_data in SAMPLE_PAPERS:
            paper = Paper(
                title=p_data["title"],
                authors=p_data["authors"],
                year=p_data["year"],
                journal=p_data["journal"],
                doi=p_data["doi"],
                url=p_data["url"],
                abstract=p_data["abstract"],
                full_text=p_data["full_text"],
                page_count=len(p_data["full_text"].split("\n\n")),
                source="pdf_upload",
                is_analyzed=True,
            )
            db.session.add(paper)
            db.session.commit()

            # Add Summary
            s_data = p_data["summary"]
            summary = PaperSummary(
                paper_id=paper.id,
                research_objectives=s_data["objectives"],
                abstract_summary=s_data["abstract"],
                methodology=s_data["methodology"],
                key_findings=s_data["findings"],
                limitations=s_data["limitations"],
                future_directions=s_data["future_directions"],
                model_used="ibm/granite-13b-chat-v2",
            )
            db.session.add(summary)

            # Add Reference
            formatted_ref = ref_service.format_all(p_data)
            ref_item = ReferenceItem(
                paper_id=paper.id,
                title=paper.title,
                authors=paper.authors,
                year=paper.year,
                journal=paper.journal,
                doi=paper.doi,
                url=paper.url,
                citation_apa=formatted_ref["apa"],
                citation_ieee=formatted_ref["ieee"],
                bibtex=formatted_ref["bibtex"],
            )
            db.session.add(ref_item)

            # Chunks for RAG
            pages_list = [
                {"page_number": idx + 1, "text": section}
                for idx, section in enumerate(p_data["full_text"].split("\n\n"))
            ]
            chunks = chunker.chunk_page_texts(pages_list, paper_id=paper.id)
            for c in chunks:
                db_chunk = PaperChunk(
                    paper_id=paper.id,
                    chunk_index=c["chunk_index"],
                    content=c["content"],
                    page_number=c["page_number"],
                    section_hint=c["section_hint"],
                )
                db.session.add(db_chunk)
                store_chunks.append({
                    "paper_id": paper.id,
                    "paper_title": paper.title,
                    "chunk_index": c["chunk_index"],
                    "content": c["content"],
                    "page_number": c["page_number"],
                    "section_hint": c["section_hint"],
                })

            db.session.commit()

        # Update Vector Store
        global_vector_store.set_chunks(store_chunks)
        print(f"Successfully seeded {len(SAMPLE_PAPERS)} papers and {len(store_chunks)} RAG chunks!")

if __name__ == "__main__":
    seed()
