import textwrap
import matplotlib.pyplot as plt
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from deepeval.metrics import (
    AnswerRelevancyMetric, FaithfulnessMetric, HallucinationMetric, ToxicityMetric, BiasMetric,
    SummarizationMetric, ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric, GEval
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from dotenv import load_dotenv
import pandas as pd
import os
from langchain_openai import AzureChatOpenAI
from deepeval.models.base_model import DeepEvalBaseLLM
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, Frame, PageTemplate
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
import io
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import math

class AzureOpenAI(DeepEvalBaseLLM):
    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "Custom Azure OpenAI Model"

# Configuration of OpenAI API
api_key = "255e75cd9f2643faa9db9d073d65bce1"  # Use env variable for security
api_base = "https://teams-center-azure-openai.openai.azure.com/"
api_version = "2024-06-01"
deployment = "teams-center-gpt-4-o"

custom_model = AzureChatOpenAI(
    openai_api_version=api_version,
    azure_deployment=deployment,
    azure_endpoint=api_base,
    openai_api_key=api_key
)

azure_openai = AzureOpenAI(model=custom_model)

# Function to plot metrics bar graph
def plot_metrics(metrics_data):
    metrics = []
    scores = []

    for metric_name, result in metrics_data.items():
        if 'score' in result:
            metrics.append(metric_name)
            scores.append(result['score'])
        else:
            print(f"Warning: No score for metric {metric_name}. Skipping...")

    if metrics and scores:
        plt.figure(figsize=(8, 6))
        plt.barh(metrics, scores, color='skyblue')
        plt.xlabel('Scores')
        plt.title('Evaluation Metrics')
        plt.xlim(0, 1)
        plt.grid(axis='x')
        plt.tight_layout()

        # Save the plot as an image
        plt.savefig('metrics_bar_graph.png')
        plt.close()
    else:
        print("No valid scores to plot.")

# Function to create spider (radar) graph
def create_spider_graph(metrics_data):
    categories = list(metrics_data.keys())
    values = [metrics_data[cat]['score'] if isinstance(metrics_data[cat], dict) and 'score' in metrics_data[cat] else 0 for cat in categories]

    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    plt.xticks(angles[:-1], categories)

    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=7)
    plt.ylim(0, 1)

    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid')
    ax.fill(angles, values, alpha=0.1)

    plt.title("Metrics Spider Graph", size=11, y=1.1)

    plt.savefig('metrics_spider_graph.png')
    plt.close()

# Function for calculating perplexity
def perplexity_metric_score(predictions):
    def tokenize(text):
        return text.split()

    def calculate_perplexity(sentence):
        N = len(tokenize(sentence))
        probability = 1 / N
        perplexity = math.pow(1/probability, 1/N)
        return perplexity

    perplexities = [calculate_perplexity(pred) for pred in predictions]
    perplexity_score = sum(perplexities) / len(perplexities)
    perplexity_score = round(perplexity_score / 10**len(str(int(perplexity_score))), 2)
    return perplexity_score

# Function for calculating BLEU score
def calculate_bleu(actual_output, expected_output):
    reference = [expected_output.split()]
    candidate = actual_output.split()
    smoothie = SmoothingFunction().method4
    weights = (0.25, 0.25, 0.25, 0.25)
    bleu_score = round(sentence_bleu(reference, candidate, weights=weights, smoothing_function=smoothie), 2)
    return bleu_score

# Function for calculating ROUGE scores
def calculate_rouge_scores(llm_answer, ground_truth):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(ground_truth, llm_answer)
    return scores

# Function for calculating METEOR score
def calculate_meteor(llm_answer, ground_truth):
    llm_answer_tokens = llm_answer.split()
    ground_truth_tokens = ground_truth.split()
    score = round(meteor_score([ground_truth_tokens], llm_answer_tokens), 2)
    return score

# Function to add logo and date-time to the PDF
def add_page_elements(canvas, doc):
    # Add logo
    logo_path = r"C:\Users\z004zn2u\Desktop\CODED\DeepEval\templates\logo.png"
    canvas.drawImage(logo_path, 0.5 * inch, letter[1] - 1.1 * inch, width=1.5 * inch, height=1 * inch, preserveAspectRatio=True)

    # Add date and time
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d %H:%M:%S")
    canvas.setFont("Helvetica", 10)
    canvas.drawRightString(letter[0] - 0.5 * inch, letter[1] - 0.5 * inch, date_time)

# Evaluation function using DeepEval metrics
def evaluate_with_deepeval(df):
    def measure_metric(metric_name_metric):
        metric_name, metric = metric_name_metric
        result = {}
        try:
            metric.measure(test_cases[metric_name])
            print(f"Metric {metric_name} Score: {getattr(metric, 'score', 'No score found')}")
            print(f"Metric {metric_name} Reason: {getattr(metric, 'reason', 'No reason found')}")

            result = {"score": round(metric.score, 2), "reason": metric.reason}
        except Exception as e:
            print(f"Error with metric {metric_name}: {e}")
            result = {"error": str(e)}
        return {metric_name: result}

    metrics = {
        "contextual_precision": ContextualPrecisionMetric(model=azure_openai, async_mode=False),
        "contextual_recall": ContextualRecallMetric(model=azure_openai, async_mode=False),
        "contextual_relevancy": ContextualRelevancyMetric(model=azure_openai, async_mode=False),
        "answer_relevancy": AnswerRelevancyMetric(model=azure_openai, async_mode=False),
        "faithfulness": FaithfulnessMetric(model=azure_openai, async_mode=False),
        "hallucination": HallucinationMetric(model=azure_openai, async_mode=False),
        "bias": BiasMetric(model=azure_openai, async_mode=False),
        "toxicity": ToxicityMetric(model=azure_openai, async_mode=False),
        "g_eval": GEval(
            model=azure_openai,
            name="Coherence",
            criteria="Coherence - determine if the actual output is coherent with the input.",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.EXPECTED_OUTPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            async_mode=False
        ),
        "summarization": SummarizationMetric(
            threshold=0.5,
            model=azure_openai,
            strict_mode=False,
            async_mode=False
        )
    }

    test_case_template = LLMTestCase(
        input=df["questions"].tolist()[0],
        actual_output=df["llm_answer"].tolist()[0],
        expected_output=df["answers"].tolist()[0],
        retrieval_context=[str(i) for i in df["contexts"].tolist()[0]],
        context=[str(i) for i in df["contexts"].tolist()[0]]
    )

    test_cases = {
        "summarization": LLMTestCase(
            input=df["contexts"].tolist()[0],
            actual_output=df["llm_answer"].tolist()[0]
        ),
        "g_eval": test_case_template,
        "contextual_precision": test_case_template,
        "contextual_recall": test_case_template,
        "contextual_relevancy": test_case_template,
        "answer_relevancy": test_case_template,
        "faithfulness": test_case_template,
        "hallucination": test_case_template,
        "bias": test_case_template,
        "toxicity": test_case_template
    }

    json_data = []
    temp_dict = {}

    with ThreadPoolExecutor() as executor:
        timeout_seconds = 60
        results = executor.map(measure_metric, metrics.items(), timeout=timeout_seconds)
        for result in results:
            dict_key = list(result.keys())[0]
            temp_dict.update({dict_key: result[dict_key]})
        json_data.append(temp_dict)

    perplexity_score = perplexity_metric_score(predictions=[df["llm_answer"].tolist()[0]])
    json_data[0].update({'perplexity_score': {'score': perplexity_score, 'reason': 'Measures model uncertainty in predicting text sequences.'}})

    bleu_score = calculate_bleu(df["llm_answer"].tolist()[0], df["answers"].tolist()[0])
    json_data[0].update({'bleu_score': {'score': bleu_score, 'reason': 'Measures n-gram precision against reference translations.'}})

    rouge_scores = calculate_rouge_scores(df["llm_answer"].tolist()[0], df["answers"].tolist()[0])
    json_data[0].update({
        'rouge1_score': {'score': rouge_scores['rouge1'].fmeasure, 'reason': ''},
        'rouge2_score': {'score': rouge_scores['rouge2'].fmeasure, 'reason': ''},
        'rougeL_score': {'score': rouge_scores['rougeL'].fmeasure, 'reason': ''}
    })

    meteor_score_value = calculate_meteor(df["llm_answer"].tolist()[0], df["answers"].tolist()[0])
    json_data[0].update({'meteor_score': {'score': meteor_score_value, 'reason': 'Measures precision, recall, synonyms, and word order.'}})

    # Generate PDF report
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    # Create a PageTemplate with the header function
    page_template = PageTemplate(frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='normal')],
                                 onPage=add_page_elements)
    doc.addPageTemplates([page_template])

    story = [Spacer(1, 10), Paragraph("INTUITIVE - LLM Evaluation Report", styles['Heading1']), Spacer(1, 24)]

    user_question = df["questions"].tolist()[0]
    story.append(Paragraph(f"User Question: {user_question}", styles['Normal']))
    story.append(Spacer(1, 12))

    input_data = [
        ["Input Question:", textwrap.fill(user_question, width=50)],
        ["LLM Answer:", textwrap.fill(df["llm_answer"].tolist()[0], width=50)],
        ["Expected Answer:", textwrap.fill(df["answers"].tolist()[0], width=50)],
        ["Retrieval Context:", textwrap.fill(", ".join(df["contexts"].tolist()[0]), width=50)]
    ]

    input_table = Table(input_data, colWidths=[150, 400])
    input_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'TOP')
    ]))

    story.append(input_table)
    story.append(Spacer(1, 24))

    metrics_data = [["Metric", "Score", "Reason"]]
    for metric_name, result in json_data[0].items():
        metrics_data.append([metric_name, str(result.get('score', 'N/A')), textwrap.fill(str(result.get('reason', '')), width=60)])

    metrics_table = Table(metrics_data, colWidths=[150, 50, 350])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'TOP')
    ]))

    story.append(metrics_table)

    # Generate and add the metrics bar graph
    plot_metrics(temp_dict)
    bar_graph_path = 'metrics_bar_graph.png'
    bar_img = Image(bar_graph_path)
    bar_img.hAlign = 'CENTER'
    story.append(bar_img)
    story.append(Spacer(1, 24))

    # Generate and add the spider graph
    create_spider_graph(temp_dict)
    spider_graph_path = 'metrics_spider_graph.png'
    spider_img = Image(spider_graph_path)
    spider_img.hAlign = 'CENTER'
    story.append(spider_img)

    doc.build(story)
    buffer.seek(0)
    return buffer