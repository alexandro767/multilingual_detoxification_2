!pip install sacrebleu sentence_transformers -qqq

import argparse
import logging
import os
from typing import Dict, List, Optional
import numpy as np
import numpy.typing as npt
import pandas as pd
import transformers
from sacrebleu import CHRF
from sentence_transformers import SentenceTransformer

from scipy.spatial.distance import cosine
from tqdm.auto import trange
import torch

def evaluate_sim(
    model: SentenceTransformer,
    original_texts: List[str],
    rewritten_texts: List[str],
    batch_size: int = 32,
    efficient_version: bool = False,
) -> npt.NDArray[np.float64]:
    """
    Evaluate the semantic similarity between original and rewritten texts.

    Args:
        model (SentenceTransformer): The sentence transformer model.
        original_texts (List[str]): List of original texts.
        rewritten_texts (List[str]): List of rewritten texts.
        batch_size (int): Batch size for inference.
        efficient_version (bool): To use efficient calculation method.

    Returns:
        npt.NDArray[np.float64]: Array of semantic similarity scores between \
              original and rewritten texts.
    """
    similarities = []

    batch_size = min(batch_size, len(original_texts))
    # for i in range(0, len(original_texts), batch_size):
    for i in trange(0, len(original_texts), batch_size, desc="Calculating SIM scores"):
        original_batch = original_texts[i : i + batch_size]
        rewritten_batch = rewritten_texts[i : i + batch_size]

        embeddings = model.encode(original_batch + rewritten_batch, show_progress_bar=False)
        original_embeddings = embeddings[: len(original_batch)]
        rewritten_embeddings = embeddings[len(original_batch) :]

        if efficient_version:
            similarity_matrix = np.dot(original_embeddings, rewritten_embeddings.T)
            original_norms = np.linalg.norm(original_embeddings, axis=1)
            rewritten_norms = np.linalg.norm(rewritten_embeddings, axis=1)
            similarity_matrix = 1 - similarity_matrix / (
                np.outer(original_norms, rewritten_norms) + 1e-9
            )
            similarities.extend(similarity_matrix.diagonal())
        else:
            t = [
                1 - cosine(original_embedding, rewritten_embedding)
                for original_embedding, rewritten_embedding in zip(
                    original_embeddings, rewritten_embeddings
                )
            ]
            similarities.extend(t)

    return np.array(similarities, dtype=np.float64)


def calculate_toxicities(
    tox_pipe: transformers.Pipeline,
    batch: List[str],
    method: str = "pipeline",
    lang: str = "ru",
):
    try:
        if method == "pipeline":
            return [
                torch.softmax(
                    tox_pipe.model(
                        **tox_pipe.tokenizer(
                            _,
                            return_tensors="pt",
                            max_length=512,
                            truncation=True,
                        ).to("cuda")
                    ).logits[0],
                    dim=-1,
                )
                .to("cpu")
                .detach()[0]
                if _ is not np.nan else 1.0
                for _ in batch
            ]
        elif method == "api":
            res = []
            for sentence in batch:
                toxicity_score = -5
                while toxicity_score <= 0:
                    try:
                        resp = get_sta_scores(
                            sentence,
                            return_spans=False,
                            return_full_response=True,
                            lang=lang,
                        )
                        toxicity_score = resp["response"]["attributeScores"]["TOXICITY"][
                            "summaryScore"
                        ]["value"]
                        break
                    except KeyError:
                        print(resp)
                        toxicity_score += 1
                else:
                    toxicity_score = 1
                res.append(1 - toxicity_score)
        
    
        return res
        
    except ValueError:
        print(batch, type(batch), all(map(lambda _: isinstance(_, str), batch)))
        raise


def evaluate_sta(
    tox_pipe: transformers.pipeline,
    texts: List[str],
    batch_size: int = 32,
    method: str = "pipeline",
    lang: str = "ru",
) -> npt.NDArray[np.float64]:
    toxicities = []

    batch_size = min(batch_size, len(texts))
    for i in trange(0, len(texts), batch_size, desc="Calculating STA scores"):
        batch = texts[i : i + batch_size]

        t = calculate_toxicities(tox_pipe, batch, method=method, lang=lang)
        toxicities.extend(t)

    return np.array(toxicities, dtype=np.float64)


def ensure_dir(directory: str):
    """Ensure that the directory exists, if not, create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def evaluate_style_transfer(
    original_texts: List[str],
    rewritten_texts: List[str],
    sta_pipeline: transformers.Pipeline,
    meaning_model: SentenceTransformer,
    references: Optional[List[str]] = None,
    batch_size: int = 32,
) -> Dict[str, npt.NDArray[np.float64]]:

    accuracy = evaluate_sta(
        tox_pipe=sta_pipeline,
        texts=rewritten_texts,
    )

    similarity = evaluate_sim(
        model=meaning_model,
        original_texts=original_texts,
        rewritten_texts=rewritten_texts,
        batch_size=batch_size,
    )

    result = {
        "STA": accuracy,
        "SIM": similarity,
    }

    if references is not None:

        chrf = CHRF()

        result["CHRF"] = np.array(
            [
                chrf.sentence_score(hypothesis=rewritten, references=[reference]).score
                / 100
                for rewritten, reference in zip(rewritten_texts, references)
            ],
            dtype=np.float64,
        )

        result["J"] = result["STA"] * result["SIM"] * result["CHRF"]
    
    return result


def find_language_column(df: pd.DataFrame) -> str:
    for col in ["lang", "language", "Lang"]:
        if col in df.columns:
            return col
    raise ValueError("No language column found in the predictions file.")


def evaluate_by_language(
    test_data: pd.DataFrame,
    predictions: pd.DataFrame,
    sta_pipeline: transformers.Pipeline,
    batch_size: int = 32,
    output_base = None,
):
    lang_col = find_language_column(predictions)
    languages = predictions[lang_col].unique()
    for language in set(test_data[lang_col].unique()) - set(languages):
        test_data = test_data[test_data[lang_col] != language]

    meaning_model = SentenceTransformer("sentence-transformers/LaBSE")
    all_aggregated_results = []
    all_non_aggregated_results = []

    for lang in languages:

        test_subset = test_data[test_data[lang_col] == lang]
        pred_subset = predictions[predictions[lang_col] == lang]

        results = evaluate_style_transfer(
            original_texts=test_subset["toxic_sentence"].values.tolist(),
            rewritten_texts=pred_subset["neutral_sentence"].values.tolist(),
            sta_pipeline=sta_pipeline,
            meaning_model=meaning_model,
            references=test_subset["neutral_sentence"],
            batch_size=batch_size,
        )

        aggregated_results = {k: np.mean(v) for k, v in results.items()}

        all_aggregated_results.append(
            {
                "language": lang,
                "STA": aggregated_results["STA"],
                "SIM": aggregated_results["SIM"],
                "CHRF": aggregated_results["CHRF"],
                "J": aggregated_results["J"],
            }
        )

        for i, original_text in enumerate(
            test_subset["toxic_sentence"].values.tolist()
        ):
            all_non_aggregated_results.append(
                {
                    "source_text": original_text,
                    "detoxified_text": pred_subset["neutral_sentence"].values.tolist()[
                        i
                    ],
                    "reference_text": test_subset["neutral_sentence"].values.tolist()[
                        i
                    ],
                    "STA": results["STA"][i],
                    "SIM": results["SIM"][i],
                    "CHRF": results["CHRF"][i],
                    "J": results["J"][i],
                    "language": lang,
                }
            )

    aggregated_output_file = f"{output_base}_aggregated.csv"
    
    print('Results will be here: ', aggregated_output_file)
    ensure_dir(os.path.dirname(aggregated_output_file))

    aggregated_df = pd.DataFrame(all_aggregated_results)
    aggregated_df.to_csv(aggregated_output_file, index=False)

def evaluate(test_data_path, predictions_path, output_base="eval_results/evaluation_results"):
    """
    Basic usage:

    python evaluate_script.py \
        --toxic_data_path "data/predictions_for_paper/multiparadetox_test.csv" \
        --detox_data_path "data/predictions_for_paper/predictions.csv" \
        --output_output_dirbase "results/evaluation_results"

    will by default will save the results (aggregated into `results/evaluation_results_aggregated.csv`)
    """

    parser = argparse.ArgumentParser(description='Train a TwoLossesModel with detoxification and classification tasks.')
    parser.add_argument('--toxic_data_path', type=str, required=True, help='Should contain [toxic_sentence, neutral_sentence, lang] columns')
    parser.add_argument('--detox_data_path', type=str, required=True, help='Should contain [neutral_sentence, lang] columns')
    parser.add_argument('--output_dir', type=str, default="results/evaluation_results", help='Where to find the results')

    args = parser.parse_args()

    toxic_data_path = args.toxic_data_path
    detox_data_path = args.detox_data_path
    output_dir = args.output_dir

    toxic_data = pd.read_csv(toxic_data_path)
    detox_data = pd.read_csv(detox_data_path)

    tox_pipe = transformers.pipeline(
        "text-classification",
        model="textdetox/xlmr-large-toxicity-classifier",
        device="cuda",
    )

    evaluate_by_language(
        test_data=toxic_data,
        predictions=detox_data,
        sta_pipeline=tox_pipe,
        batch_size=32,
        output_base=output_dir,
    )

