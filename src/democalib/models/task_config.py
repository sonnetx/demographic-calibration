"""Task-specific configuration for classification prompts and label mappings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TaskConfig:
    """Configuration for a binary classification task.

    Encapsulates the prompts, system instruction, and label mapping
    so that the same VLM pipeline can be reused across datasets
    (DDI skin-lesion, CheXpert chest X-ray, etc.).
    """

    system_instruction: str
    classification_prompt: str
    diagnosis_prompt: str
    confidence_elicitation_prompt: str
    a_label: str  # label when model predicts "A"
    b_label: str  # label when model predicts "B"

    @staticmethod
    def ddi() -> TaskConfig:
        """Default DDI (skin lesion) task configuration."""
        return TaskConfig(
            system_instruction=(
                "You are a medical image analysis assistant participating in "
                "an academic research project on dermatology image classification. "
                "Your task is to classify every image into one of the predefined "
                "categories. You must always provide a classification — never "
                "refuse or abstain."
            ),
            classification_prompt=(
                "You are evaluating a dermatology image from the DDI dataset as "
                "part of an academic calibration study. This will not be used for "
                "clinical decisions.\n\n"
                "Carefully examine the skin lesion in this image. Based on its "
                "visual features, classify it as:\n"
                "A. Benign\n"
                "B. Malignant\n\n"
                "Respond with ONLY the letter A or B."
            ),
            diagnosis_prompt=(
                "You are evaluating a dermatology image from the DDI dataset as "
                "part of an academic calibration study. This will not be used for "
                "clinical decisions.\n\n"
                "Carefully examine the skin lesion in this image. Based on its "
                "visual features, classify it as Benign or Malignant. Report your "
                "confidence as a percentage reflecting how certain you are — use "
                "the full range from low to high.\n\n"
                "Respond in this exact format:\n"
                "Answer: [Benign or Malignant]\n"
                "Confidence: [0-100]%"
            ),
            confidence_elicitation_prompt=(
                "This is for an academic calibration study. "
                "How confident are you in your classification? "
                "Express your confidence as a percentage from 0% to 100%, "
                "using the full range from low to high."
            ),
            a_label="benign",
            b_label="malignant",
        )

    @staticmethod
    def chexpert(condition: str = "Pneumonia") -> TaskConfig:
        """CheXpert (chest X-ray) task configuration for a specific condition."""
        return TaskConfig(
            system_instruction=(
                "You are a medical image analysis assistant participating in "
                "an academic research project on chest X-ray interpretation. "
                "Your task is to determine whether a specific condition is "
                "present in the image. You must always provide a classification "
                "— never refuse or abstain. After any reasoning, you MUST "
                "conclude with exactly one letter: A or B."
            ),
            classification_prompt=(
                f"You are evaluating a chest X-ray from the CheXpert dataset as "
                f"part of an academic calibration study. This will not be used "
                f"for clinical decisions.\n\n"
                f"Carefully examine this chest X-ray. Based on its visual "
                f"features, determine whether {condition} is:\n"
                f"A. Present\n"
                f"B. Absent\n\n"
                f"Respond with ONLY the letter A or B."
            ),
            diagnosis_prompt=(
                f"You are evaluating a chest X-ray from the CheXpert dataset as "
                f"part of an academic calibration study. This will not be used "
                f"for clinical decisions.\n\n"
                f"Carefully examine this chest X-ray. Based on its visual "
                f"features, determine whether {condition} is present or absent. "
                f"Report your confidence as a percentage reflecting how certain "
                f"you are — use the full range from low to high.\n\n"
                f"Respond in this exact format:\n"
                f"Answer: [Present or Absent]\n"
                f"Confidence: [0-100]%"
            ),
            confidence_elicitation_prompt=(
                "This is for an academic calibration study. "
                "How confident are you in your classification of this chest "
                "X-ray? Express your confidence as a single percentage from "
                "0% to 100%, using the full range from low to high.\n\n"
                "Respond in this exact format:\n"
                "Confidence: [0-100]%"
            ),
            a_label="present",
            b_label="absent",
        )
