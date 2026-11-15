import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from rag.config import PROJECT_ROOT, RAG_TOP_K


FULL_EVAL_CACHE_PATH = PROJECT_ROOT / "storage" / "answer_evaluation_cache.json"
EVALUATION_HISTORY_PATH = PROJECT_ROOT / "storage" / "evaluation_history.jsonl"
FULL_EVAL_DELAY = max(float(os.getenv("RAG_FULL_EVAL_DELAY", "0.25")), 0.0)
FULL_EVAL_RETRIES = max(int(os.getenv("RAG_FULL_EVAL_RETRIES", "2")), 0)


def _runtime_evaluation_config(k):
    from rag import retriever

    return {
        "search_mode": retriever.DEFAULT_SEARCH_MODE,
        "hybrid_semantic_weight": retriever.HYBRID_SEMANTIC_WEIGHT,
        "min_score": float(os.getenv("RAG_MIN_SCORE", "0.3")),
        "comprehensive_min_score": retriever.COMPREHENSIVE_MIN_SCORE,
        "rerank_enabled": retriever.RERANK_ENABLED,
        "rerank_model": retriever.RERANK_MODEL_NAME,
        "rerank_candidates": retriever.RERANK_CANDIDATES,
        "rerank_min_score": retriever.RERANK_MIN_SCORE,
        "rerank_relative_cutoff": retriever.RERANK_RELATIVE_CUTOFF,
        "max_chunks_per_source": retriever.MAX_CHUNKS_PER_SOURCE,
        "chunk_size": int(os.getenv("RAG_CHUNK_SIZE", "180")),
        "overlap": int(os.getenv("RAG_CHUNK_OVERLAP", "30")),
        "k": k,
    }


def record_evaluation_run(report, run_type="retrieval", history_path=None):
    """Append one evaluation report to the local JSONL history."""
    path = Path(history_path or EVALUATION_HISTORY_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "run_id": uuid4().hex[:12],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": run_type,
        "config": _runtime_evaluation_config(report.get("k", RAG_TOP_K)),
        "metrics": {
            key: value
            for key, value in report.items()
            if key not in {"results", "quality_explanation"}
        },
        "results": report.get("results", []),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def load_evaluation_history(history_path=None):
    path = Path(history_path or EVALUATION_HISTORY_PATH)
    if not path.exists():
        return []
    runs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            runs.append(record)
    return runs


def _select_history_run(runs, identifier):
    for index, run in enumerate(runs):
        if str(index) == str(identifier) or run.get("run_id") == identifier:
            return index, run
    raise ValueError(f"Evaluation run not found: {identifier}")


def summarize_failures(runs, limit=10):
    failures = {}
    for run in runs:
        for result in run.get("results", []):
            if result.get("passed"):
                continue
            key = (result.get("question", ""), result.get("expected_source"))
            entry = failures.setdefault(
                key,
                {"question": key[0], "source": key[1], "count": 0, "run_ids": []},
            )
            entry["count"] += 1
            entry["run_ids"].append(run.get("run_id"))
    return sorted(
        failures.values(),
        key=lambda item: (-item["count"], item["question"]),
    )[:limit]


def compare_evaluation_runs(left, right):
    metric_names = (
        "precision_percentage", "capped_precision_percentage",
        "context_precision_percentage", "recall_percentage", "f1_percentage",
        "mrr_percentage", "no_answer_percentage", "pass_percentage",
    )
    left_metrics = left.get("metrics", {})
    right_metrics = right.get("metrics", {})
    return {
        name: {
            "left": left_metrics.get(name),
            "right": right_metrics.get(name),
            "delta": (
                right_metrics.get(name) - left_metrics.get(name)
                if isinstance(left_metrics.get(name), (int, float))
                and isinstance(right_metrics.get(name), (int, float))
                else None
            ),
        }
        for name in metric_names
    }


EVAL_CASES = [
    {
        "question": "What are the prerequisites for AI-101?",
        "source": "course.txt",
        "terms": ["prerequisites for ai-101", "basic python", "introductory statistics"],
        "answer": "Basic Python programming and introductory statistics.",
    },
    {
        "question": "What is the late submission penalty?",
        "source": "university_policy.txt",
        "terms": ["late submission rule", "10 percent", "calendar day"],
        "answer": "10 percent of the mark per calendar day, with a maximum of 3 calendar days. An approved extension removes the penalty.",
    },
    {
        "question": "How many credit hours is AI-101?",
        "source": "course.txt",
        "terms": ["credit hours", "3"],
        "answer": "3 credit hours.",
    },
    {
        "question": "What software is required?",
        "source": "course.txt",
        "terms": ["required software", "python 3.11", "chromadb", "sentence transformers"],
        "answer": "Python 3.11 or later, ChromaDB, and Sentence Transformers.",
    },
    {
        "question": "Can students use AI tools for assignments?",
        "source": "university_policy.txt",
        "terms": ["artificial intelligence rule", "ai tools", "instructor", "assignment"],
        "answer": "AI tools may be used only when the instructor or assignment allows them.",
    },
    {
        "question": "How long do students have to request a grade review?",
        "source": "university_policy.txt",
        "terms": ["grade review rule", "7 calendar days"],
        "answer": "Within 7 calendar days of receiving the grade.",
    },
    {
        "question": "What are the final project requirements?",
        "source": "course.txt",
        "terms": ["final project requirement", "two source documents", "ten questions"],
        "answer": "Use at least two source documents, searchable chunks with metadata, a vector database, at least ten test questions, and a report of results and limitations.",
    },
    {
        "question": "What background knowledge is needed before AI-101?",
        "source": "course.txt",
        "terms": ["prerequisites for ai-101", "basic python", "introductory statistics"],
        "answer": "Basic Python programming and introductory statistics.",
    },
    {
        "question": "How are late assignments handled, including extensions?",
        "source": "university_policy.txt",
        "terms": ["late submission rule", "10 percent", "approved extension"],
        "answer": "Late work loses 10 percent per calendar day; an approved extension removes the penalty.",
    },
    {
        "question": "How are the course assessment percentages divided?",
        "source": "course.txt",
        "terms": ["assessment weight", "25 percent", "20 percent", "100 percent"],
        "answer": "Programming assignments are 25%, laboratories 20%, the midterm 20%, the final project 25%, and participation 10%; the total is 100%.",
    },
    {
        "question": "What is the capital of France?",
        "source": None,
        "terms": [],
        "answer": "I don't have enough information to answer this question.",
        "should_answer": False,
    },
    {
        "question": "What is the tuition fee?",
        "source": None,
        "terms": [],
        "answer": "I don't have enough information to answer this question.",
        "should_answer": False,
    },
    {
        "question": "Explain quantum entanglement.",
        "source": None,
        "terms": [],
        "answer": "I don't have enough information to answer this question.",
        "should_answer": False,
    },
    {
        "question": "What is an Americano?",
        "source": "coffee-08-drinks.pdf",
        "terms": ["americano", "espresso diluted with hot water"],
        "answer": "An Americano is espresso diluted with hot water.",
    },
    {
        "question": "What does a latte contain?",
        "source": "coffee-08-drinks.pdf",
        "terms": ["latte", "steamed milk", "thin layer of foam"],
        "answer": "A latte is espresso with a lot of steamed milk and a thin layer of foam.",
    },
    {
        "question": "How is a cappuccino composed?",
        "source": "coffee-08-drinks.pdf",
        "terms": ["cappuccino", "equal parts espresso", "milk foam"],
        "answer": "A cappuccino has equal parts espresso, steamed milk, and milk foam.",
    },
    {
        "question": "How much foam does a flat white have?",
        "source": "coffee-08-drinks.pdf",
        "terms": ["flat white", "velvety steamed milk", "very little foam"],
        "answer": "A flat white uses smooth, velvety steamed milk and very little foam.",
    },
    {
        "question": "What is a macchiato?",
        "source": "coffee-08-drinks.pdf",
        "terms": ["macchiato", "small spoonful of foam"],
        "answer": "A macchiato is espresso marked with a small spoonful of foam.",
    },
    {
        "question": "How does a latte differ from the espresso brewing process?",
        "source": "coffee-08-drinks.pdf",
        "terms": ["latte", "steamed milk", "thin layer of foam"],
        "answer": "The drinks document defines a latte by its espresso, steamed milk, and thin layer of foam; the brewing document describes how espresso is extracted.",
    },
    {
        "question": "What share of coffee output is Arabica?",
        "source": "coffee_from_farm_to_cup.pdf",
        "terms": ["arabica", "60 percent of world output"],
        "answer": "Arabica accounts for roughly 60 percent of world coffee output.",
    },
    {
        "question": "Where is the Bean Belt and where does Arabica grow best?",
        "source": "coffee_from_farm_to_cup.pdf",
        "terms": ["bean belt", "1,000 and 2,000 meters"],
        "answer": "The Bean Belt is between the Tropic of Cancer and the Tropic of Capricorn; Arabica often prefers 1,000 to 2,000 meters.",
    },
    {
        "question": "What happens during washed coffee processing?",
        "source": "coffee_from_farm_to_cup.pdf",
        "terms": ["washed or wet process", "ferment briefly in water", "washed and dried"],
        "answer": "The skin and pulp are removed, the beans ferment briefly in water, and then they are washed and dried.",
    },
    {
        "question": "How do natural and honey coffee processes differ?",
        "source": "coffee_from_farm_to_cup.pdf",
        "terms": ["natural or dry process", "honey process", "mucilage"],
        "answer": "Natural coffee dries as whole cherries, while the honey process removes the skin but leaves some mucilage during drying.",
    },
    {
        "question": "What pressure and time are typical for espresso?",
        "source": "coffee_from_farm_to_cup.pdf",
        "terms": ["9 bars of pressure", "25 to 30 seconds"],
        "answer": "Espresso is made with water forced through fine grounds at about 9 bars, producing a shot in 25 to 30 seconds.",
    },
    {
        "question": "Which coffee species has more caffeine, Arabica or Robusta?",
        "source": "coffee_from_farm_to_cup.pdf",
        "terms": ["robusta", "twice as much caffeine", "arabica"],
        "answer": "Robusta contains roughly twice as much caffeine as Arabica.",
    },
    {
        "question": "How long is the AI-101 course and how is it delivered?",
        "source": "course.txt",
        "terms": ["course duration", "14 weeks", "two lectures", "one laboratory session"],
        "answer": "AI-101 lasts 14 weeks and has two lectures and one laboratory session each week.",
    },
    {
        "question": "What does the first AI-101 learning outcome cover?",
        "source": "course.txt",
        "terms": ["learning outcome 1", "artificial intelligence, machine learning, and deep learning"],
        "answer": "Students should explain the difference between artificial intelligence, machine learning, and deep learning.",
    },
    {
        "question": "What modules are included in AI-101?",
        "source": "course.txt",
        "terms": ["module 1", "module 4", "module 8", "responsible ai"],
        "answer": "The modules cover AI foundations; supervised and unsupervised learning; Python text processing; chunking and metadata; embeddings; vector databases; RAG; and evaluation, bias, privacy, and responsible AI.",
    },
    {
        "question": "What editor is recommended for the course?",
        "source": "course.txt",
        "terms": ["recommended editor", "visual studio code", "jupyter notebook"],
        "answer": "Visual Studio Code or Jupyter Notebook is recommended.",
    },
    {
        "question": "What should a technical support question include?",
        "source": "course.txt",
        "terms": ["student support", "error message", "reproduction steps"],
        "answer": "Technical questions should include the error message and reproduction steps.",
    },
    {
        "question": "How much sleep do most adults need?",
        "source": "healthy.txt",
        "terms": ["7 to 9 hours", "circadian rhythm"],
        "answer": "Most adults need 7 to 9 hours of sleep per night, and consistent sleep times support the circadian rhythm.",
    },
    {
        "question": "How much water is a common daily guideline?",
        "source": "healthy.txt",
        "terms": ["2 to 3 liters", "dark urine", "headaches", "fatigue"],
        "answer": "A common guideline is 2 to 3 liters per day; dark urine, headaches, and fatigue can signal mild dehydration.",
    },
    {
        "question": "What weekly exercise is recommended?",
        "source": "healthy.txt",
        "terms": ["150 minutes", "muscle-strengthening exercises", "two or more days"],
        "answer": "The guide recommends at least 150 minutes of moderate aerobic activity weekly plus muscle-strengthening exercises on two or more days.",
    },
    {
        "question": "What proportions make up the balanced plate?",
        "source": "healthy.txt",
        "terms": ["half vegetables and fruit", "quarter lean protein", "quarter whole grains"],
        "answer": "Half the plate should be vegetables and fruit, with a quarter lean protein and a quarter whole grains.",
    },
    {
        "question": "How does the Pomodoro technique work?",
        "source": "healthy.txt",
        "terms": ["25 minutes", "5-minute break", "four rounds"],
        "answer": "It uses 25 minutes of focused work followed by a 5-minute break; after four rounds, take a longer break.",
    },
    {
        "question": "What tools measured time before mechanical clocks?",
        "source": "mechanical_clock_history.pdf",
        "terms": ["sundials", "water clocks", "candle clocks"],
        "answer": "People used sundials, water clocks, and candle clocks.",
    },
    {
        "question": "What is the central invention of mechanical timekeeping?",
        "source": "mechanical_clock_history.pdf",
        "terms": ["central invention of mechanical timekeeping", "escapement"],
        "answer": "The escapement releases energy from a weight or spring at a steady rate.",
    },
    {
        "question": "When did Huygens build the first pendulum clock?",
        "source": "mechanical_clock_history.pdf",
        "terms": ["1656 and 1657", "christiaan huygens", "first pendulum clock"],
        "answer": "Christiaan Huygens built the first pendulum clock in 1656 and 1657.",
    },
    {
        "question": "What was John Harrison's H4?",
        "source": "mechanical_clock_history.pdf",
        "terms": ["h4", "1759", "sea voyage"],
        "answer": "Harrison's H4 was a marine timekeeper completed in 1759 that kept accurate time during a sea voyage.",
    },
    {
        "question": "How is the second defined by atomic timekeeping?",
        "source": "mechanical_clock_history.pdf",
        "terms": ["second", "9,192,631,770 cycles", "cesium-133"],
        "answer": "The second is defined by 9,192,631,770 cycles of radiation associated with cesium-133.",
    },
    {
        "question": "How many octopus species are described and how many arms do they have?",
        "source": "octopus_biology_and_intelligence.pdf",
        "terms": ["300 described species", "eight arms"],
        "answer": "There are roughly 300 described octopus species, and octopuses have eight arms.",
    },
    {
        "question": "How many hearts does an octopus have?",
        "source": "octopus_biology_and_intelligence.pdf",
        "terms": ["three hearts", "two branchial hearts", "systemic heart"],
        "answer": "An octopus has three hearts: two branchial hearts and one systemic heart.",
    },
    {
        "question": "Where are most of an octopus's neurons located?",
        "source": "octopus_biology_and_intelligence.pdf",
        "terms": ["two thirds of an octopus's neurons", "arms"],
        "answer": "Roughly two thirds of an octopus's neurons are located in its arms.",
    },
    {
        "question": "How do octopuses defend themselves?",
        "source": "octopus_biology_and_intelligence.pdf",
        "terms": ["change color", "dark ink", "jet propulsion"],
        "answer": "They change color and texture, release dark ink, and can escape by jet propulsion.",
    },
    {
        "question": "What is semelparity in octopuses?",
        "source": "octopus_biology_and_intelligence.pdf",
        "terms": ["breed once", "die soon afterward", "semelparity"],
        "answer": "Semelparity is the pattern in which most octopus species breed once and die soon afterward.",
    },
    {
        "question": "What does a photovoltaic cell do?",
        "source": "photovoltaic_solar_energy.pdf",
        "terms": ["converts sunlight directly into electricity", "photovoltaic effect"],
        "answer": "A photovoltaic cell converts sunlight directly into electricity through the photovoltaic effect, usually using silicon.",
    },
    {
        "question": "What efficiency range is common for monocrystalline panels?",
        "source": "photovoltaic_solar_energy.pdf",
        "terms": ["monocrystalline silicon", "20 to 24 percent"],
        "answer": "Commercial monocrystalline silicon panels commonly have efficiencies from 20 to 24 percent.",
    },
    {
        "question": "What components are in a typical grid-connected residential system?",
        "source": "photovoltaic_solar_energy.pdf",
        "terms": ["solar panels", "inverter", "mounting hardware", "battery"],
        "answer": "It includes panels, an inverter, mounting hardware, wiring, and a meter, and it may include a battery.",
    },
    {
        "question": "How does heat affect crystalline silicon panels?",
        "source": "photovoltaic_solar_energy.pdf",
        "terms": ["0.3 to 0.5 percent", "above 25 degrees Celsius"],
        "answer": "Crystalline silicon panels lose roughly 0.3 to 0.5 percent of output for each degree Celsius above 25 degrees Celsius.",
    },
    {
        "question": "How long are solar panels usually warranted and what is their energy payback time?",
        "source": "photovoltaic_solar_energy.pdf",
        "terms": ["warranty panels for 25 years", "one and three years"],
        "answer": "Manufacturers commonly warranty panels for 25 years, and modern silicon panels usually repay their embodied energy in one to three years.",
    },
    {
        "question": "How should a Python file be opened safely?",
        "source": "python_cheatsheet.md",
        "terms": ["files close automatically", "encoding"],
        "answer": "Use a with statement so the file closes automatically, for example with open(..., encoding=...).",
    },
    {
        "question": "What is the difference between a list and a dictionary?",
        "source": "python_cheatsheet.md",
        "terms": ["list", "ordered, mutable collection", "dictionary", "key-value pairs"],
        "answer": "A list is an ordered, mutable collection, while a dictionary stores key-value pairs.",
    },
    {
        "question": "What does pandas read_csv return?",
        "source": "python_cheatsheet.md",
        "terms": ["pandas.read_csv", "dataframe"],
        "answer": "pandas.read_csv returns a DataFrame.",
    },
    {
        "question": "How should a ValueError be handled in the cheatsheet?",
        "source": "python_cheatsheet.md",
        "terms": ["try", "except ValueError", "please enter a number"],
        "answer": "Put the conversion in a try block and catch ValueError, then ask the user to enter a number.",
    },
    {
        "question": "How do you create and activate a virtual environment on Windows?",
        "source": "python_cheatsheet.md",
        "terms": ["python -m venv venv", "venv\\scripts\\activate"],
        "answer": "Create it with python -m venv venv and activate it on Windows with venv\\Scripts\\activate.",
    },
    {
        "question": "Who coined the name Silk Road and when?",
        "source": "silk_road_history.pdf",
        "terms": ["coined in 1877", "ferdinand von richthofen"],
        "answer": "The German geographer Ferdinand von Richthofen coined the name in 1877.",
    },
    {
        "question": "What did Zhang Qian's mission contribute to the Silk Road?",
        "source": "silk_road_history.pdf",
        "terms": ["zhang qian", "138 bce", "regular diplomatic and commercial contact"],
        "answer": "Zhang Qian traveled west around 138 BCE, and his reports opened the way for regular diplomatic and commercial contact.",
    },
    {
        "question": "What goods traveled along the Silk Road?",
        "source": "silk_road_history.pdf",
        "terms": ["other eastbound and westbound commodities", "spices, glassware", "paper, porcelain"],
        "answer": "Goods included silk, spices, glassware, precious stones, paper, porcelain, and many others.",
    },
    {
        "question": "Which religions spread along the Silk Road?",
        "source": "silk_road_history.pdf",
        "terms": ["buddhism", "manichaeism", "islam"],
        "answer": "Buddhism, Manichaeism, Nestorian Christianity, and later Islam traveled along the network.",
    },
    {
        "question": "What was the Pax Mongolica and what later reduced overland trade?",
        "source": "silk_road_history.pdf",
        "terms": ["pax mongolica", "maritime routes around africa", "portuguese navigators"],
        "answer": "The Mongol Empire made overland travel safer in an effect called Pax Mongolica; later maritime routes around Africa reduced the network's importance.",
    },
    {
        "question": "What is the difference between photovoltaic and solar thermal energy?",
        "source": "solar_energy_basics.pdf",
        "terms": ["panels convert sunlight directly into electricity", "solar thermal systems"],
        "answer": "Photovoltaic panels convert sunlight into electricity, while solar thermal systems use sunlight to heat water or fluids.",
    },
    {
        "question": "What current does a solar panel produce and what does the inverter do?",
        "source": "solar_energy_basics.pdf",
        "terms": ["this current is direct current", "inverter converts it to alternating current"],
        "answer": "A solar panel produces direct current, and an inverter converts it to alternating current.",
    },
    {
        "question": "What are the listed solar panel types and efficiencies?",
        "source": "solar_energy_basics.pdf",
        "terms": ["monocrystalline", "20-23%", "polycrystalline", "15-18%", "thin-film", "10-13%"],
        "answer": "The guide lists monocrystalline at 20–23%, polycrystalline at 15–18%, and thin-film at 10–13%.",
    },
    {
        "question": "How much sun and solar irradiation does Jordan receive?",
        "source": "solar_energy_basics.pdf",
        "terms": ["300 sunny days", "5 to 7 kwh", "per square meter per day"],
        "answer": "Jordan receives roughly 300 sunny days per year and about 5 to 7 kWh per square meter per day of solar irradiation.",
    },
    {
        "question": "What are the main solar advantages, limitations, and maintenance tasks?",
        "source": "solar_energy_basics.pdf",
        "terms": ["low running costs", "output drops at night", "clean panels", "inspect wiring annually"],
        "answer": "Advantages include low running costs and no fuel emissions; output drops at night or in clouds, and maintenance includes cleaning panels and inspecting wiring annually.",
    },
    {
        "question": "Compare the solar panel types described in the two solar documents.",
        "source": "solar_energy_basics.pdf",
        "terms": ["monocrystalline", "20-23%", "polycrystalline", "thin-film"],
        "answer": "The solar basics document compares monocrystalline, polycrystalline, and thin-film panels with different efficiency, cost, and use profiles.",
    },
    {
        "question": "What attendance notification deadline applies after a missed class?",
        "source": "university_policy.txt",
        "terms": ["attendance exception", "within 48 hours"],
        "answer": "A student who misses a class should notify the instructor within 48 hours.",
    },
    {
        "question": "What actions violate academic integrity?",
        "source": "university_policy.txt",
        "terms": ["plagiarism rule", "copying words", "cheating rule", "unauthorized help"],
        "answer": "Copying words, code, data, or ideas without credit and using unauthorized help are prohibited.",
    },
    {
        "question": "What privacy rules apply to AI tools?",
        "source": "university_policy.txt",
        "terms": ["artificial intelligence privacy", "confidential university", "personal information"],
        "answer": "Students must not upload confidential university or personal information to public AI tools.",
    },
    {
        "question": "What must a grade review request include?",
        "source": "university_policy.txt",
        "terms": ["grade review requirement", "rubric criterion", "reason for the review"],
        "answer": "It must identify the rubric criterion and explain the reason for the review.",
    },
    {
        "question": "What information should academic communication include?",
        "source": "university_policy.txt",
        "terms": ["official university email", "student name", "course code", "clear request"],
        "answer": "Students should use their official university email and include their name, course code, and a clear request.",
    },
    {
        "question": "How many hearts does a squid have?",
        "source": None,
        "terms": [],
        "answer": "I don't have enough information to answer this question.",
        "should_answer": False,
    },
    {
        "question": "How many sunny days does Lebanon receive each year?",
        "source": None,
        "terms": [],
        "answer": "I don't have enough information to answer this question.",
        "should_answer": False,
    },
    {
        "question": "What late penalty applies to graduate students?",
        "source": None,
        "terms": [],
        "answer": "I don't have enough information to answer this question.",
        "should_answer": False,
    },
]


def recall_at_k(ranked_ids, relevant_ids, k):
    relevant_ids = set(relevant_ids)

    if not relevant_ids:
        return None

    retrieved = set(ranked_ids[:k])

    return len(retrieved & relevant_ids) / len(relevant_ids)


def precision_at_k(ranked_ids, relevant_ids, k):
    relevant_ids = set(relevant_ids)

    if k < 1:
        raise ValueError("k must be at least 1")

    retrieved = ranked_ids[:k]

    relevant_count = sum(
        chunk_id in relevant_ids
        for chunk_id in retrieved
    )

    return relevant_count / k


def context_precision(context_ids, relevant_ids):
    """Precision using the chunks actually sent to answer generation."""
    relevant_ids = set(relevant_ids)
    if not context_ids or not relevant_ids:
        return None
    return len(set(context_ids) & relevant_ids) / len(context_ids)


def r_precision_at_k(ranked_ids, relevant_ids, k):
    """Capped precision with a denominator of min(k, number of relevant chunks)."""
    relevant_ids = set(relevant_ids)
    if not relevant_ids:
        return None
    r = min(k, len(relevant_ids))
    if r < 1:
        return None
    retrieved = set(ranked_ids[:r])
    return len(retrieved & relevant_ids) / r


def reciprocal_rank(ranked_ids, relevant_ids):
    relevant_ids = set(relevant_ids)

    for rank, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in relevant_ids:
            return 1 / rank

    return 0


def _source_matches(chunk, expected_source):
    return bool(
        expected_source
        and chunk.get("source", "").replace("\\", "/").endswith(expected_source)
    )


def _relevant_chunk_ids(chunks, case):
    """Build relevance labels from the expected source and answer terms."""
    expected_source = case.get("source")
    all_terms = [term.lower() for term in case.get("terms", [])]
    # Standalone numbers are too broad to label a chunk as relevant (for
    # example, many course chunks may contain the number "3").
    expected_terms = [term for term in all_terms if not term.strip().isdigit()] or all_terms
    if not expected_source or not expected_terms:
        return []

    return [
        chunk["id"]
        for chunk in chunks
        if _source_matches(chunk, expected_source)
        and any(term in chunk.get("text", "").lower() for term in expected_terms)
    ]


def _percentage(value):
    return round(value * 100, 2) if value is not None else None


def _f1_score(precision, recall):
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


_QUESTION_STOP_WORDS = {
    "a", "an", "and", "are", "be", "can", "does", "do", "for", "from",
    "has", "have", "how", "in", "is", "it", "many", "of", "on", "the",
    "to", "was", "what", "when", "where", "which", "who", "why", "with",
    "year", "years",
}


def _question_terms(question):
    return [
        token for token in re.findall(r"[a-z0-9]+", question.lower())
        if token not in _QUESTION_STOP_WORDS and len(token) > 2
    ]


def _has_direct_support(question, chunks):
    """Treat related-but-incomplete evidence as insufficient for abstention tests."""
    terms = _question_terms(question)
    if not terms:
        return bool(chunks)
    for chunk in chunks:
        sentences = re.split(r"[.!?]+", chunk.get("text", "").lower())
        if any(
            all(term in re.findall(r"[a-z0-9]+", sentence) for term in terms)
            for sentence in sentences
        ):
            return True
    return False


def _case_explanation(case, passed, relevant_retrieved, relevant_total, retrieved_count):
    if not case.get("should_answer", True):
        if passed:
            if retrieved_count:
                return "Related chunks were retrieved, but none directly supported the complete question, so the system correctly abstained."
            return "No matching evidence was retrieved, so the system correctly abstained."
        return "Evidence was retrieved for an unsupported question, which can lead to an invented answer."

    if not relevant_total:
        return "No labeled relevant chunk was found for this test case, so it needs manual review."
    if relevant_retrieved == 0:
        return "None of the relevant chunks reached the top results, so the answer would lack evidence."
    if relevant_retrieved < relevant_total:
        return (
            f"The top results contain {relevant_retrieved} of {relevant_total} relevant chunks; "
            "some supporting facts may be missed."
        )
    if retrieved_count > relevant_retrieved:
        return (
            f"All {relevant_total} relevant chunks were found, but "
            f"{retrieved_count - relevant_retrieved} extra chunks were also retrieved."
        )
    return "All labeled relevant chunks were retrieved without extra evidence."


def run_evaluation(retrieve_fn, k=None):
    """Check whether the top evidence chunks contain the expected answer."""
    if k is None:
        k = RAG_TOP_K
    results = []

    for case in EVAL_CASES:
        question = case["question"]
        expected_source = case["source"]
        expected_terms = case["terms"]
        should_answer = case.get("should_answer", True)
        # Keep the actual context window separate from the larger ranked set
        # used to discover all labeled relevant chunks.
        chunks = retrieve_fn(question, k=k)
        ranked_chunks = retrieve_fn(question, k=max(k, 1000))
        relevant_ids = _relevant_chunk_ids(ranked_chunks, case)
        context_ids = [chunk["id"] for chunk in chunks]
        ranked_ids = [chunk["id"] for chunk in ranked_chunks]
        precision = precision_at_k(context_ids, relevant_ids, k) if relevant_ids else None
        capped_precision = r_precision_at_k(context_ids, relevant_ids, k)
        context_precision_value = context_precision(context_ids, relevant_ids)
        recall = recall_at_k(context_ids, relevant_ids, k) if relevant_ids else None
        top_k_ids = set(context_ids[:k])
        relevant_retrieved = len(top_k_ids & set(relevant_ids))
        reciprocal = reciprocal_rank(ranked_ids, relevant_ids) if relevant_ids else None
        top_chunk = chunks[0] if chunks else {}
        relevant_chunks = [
            chunk for chunk in chunks
            if _source_matches(chunk, expected_source)
        ]
        evidence_text = "\n".join(chunk.get("text", "") for chunk in relevant_chunks).lower()
        found_terms = [term for term in expected_terms if term.lower() in evidence_text]
        missing_terms = [term for term in expected_terms if term.lower() not in evidence_text]
        correct = (
            not _has_direct_support(question, chunks)
            if not should_answer
            else bool(relevant_chunks) and not missing_terms
        )
        results.append({
            "question": question,
            "expected_source": expected_source,
            "expected_answer": case["answer"],
            "should_answer": should_answer,
            "expected_terms": expected_terms,
            "found_terms": found_terms,
            "missing_terms": missing_terms,
            "passed": correct,
            "top_source": top_chunk.get("source"),
            "top_chunk": top_chunk.get("chunk_index"),
            "retrieved_text": "\n".join(chunk.get("text", "") for chunk in relevant_chunks) or top_chunk.get("text", ""),
            "relevant_chunks": len(relevant_ids),
            "precision_at_k": round(precision, 4) if precision is not None else None,
            "capped_precision_at_k": round(capped_precision, 4) if capped_precision is not None else None,
            "context_precision_at_k": round(context_precision_value, 4) if context_precision_value is not None else None,
            "recall_at_k": round(recall, 4) if recall is not None else None,
            "precision_percentage": _percentage(precision),
            "capped_precision_percentage": _percentage(capped_precision),
            "context_precision_percentage": _percentage(context_precision_value),
            "recall_percentage": _percentage(recall),
            "relevant_retrieved": relevant_retrieved,
            "relevant_total": len(relevant_ids),
            "retrieved_count": len(chunks),
            "first_relevant_rank": round(1 / reciprocal, 2) if reciprocal else None,
            "explanation": _case_explanation(
                case,
                correct,
                relevant_retrieved,
                len(relevant_ids),
                len(chunks),
            ),
        })

    passed = sum(result["passed"] for result in results)
    precision_values = [
        result["precision_at_k"]
        for result in results
        if result["precision_at_k"] is not None
    ]
    capped_precision_values = [
        result["capped_precision_at_k"]
        for result in results
        if result["capped_precision_at_k"] is not None
    ]
    context_precision_values = [
        result["context_precision_at_k"]
        for result in results
        if result["context_precision_at_k"] is not None
    ]
    recall_values = [
        result["recall_at_k"]
        for result in results
        if result["recall_at_k"] is not None
    ]
    average_precision = (
        sum(precision_values) / len(precision_values)
        if precision_values else None
    )
    average_capped_precision = (
        sum(capped_precision_values) / len(capped_precision_values)
        if capped_precision_values else None
    )
    average_context_precision = (
        sum(context_precision_values) / len(context_precision_values)
        if context_precision_values else None
    )
    average_recall = (
        sum(recall_values) / len(recall_values)
        if recall_values else None
    )
    f1 = _f1_score(average_precision, average_recall)
    context_f1 = _f1_score(average_context_precision, average_recall)
    reciprocal_values = [
        result["first_relevant_rank"]
        for result in results
        if result["first_relevant_rank"] is not None
    ]
    mean_reciprocal_rank = (
        sum(1 / rank for rank in reciprocal_values) / len(reciprocal_values)
        if reciprocal_values else None
    )
    no_answer_results = [result for result in results if not result["should_answer"]]
    no_answer_passed = sum(result["passed"] for result in no_answer_results)
    pass_rate = passed / len(results) if results else None
    no_answer_rate = (
        no_answer_passed / len(no_answer_results)
        if no_answer_results else None
    )

    if (
        pass_rate is not None
        and pass_rate >= 0.9
        and average_recall is not None
        and average_recall >= 0.9
        and average_context_precision is not None
        and average_context_precision >= 0.6
    ):
        quality_label = "Strong retrieval"
    elif average_recall is not None and average_recall >= 0.8:
        quality_label = "Good coverage; improve focus"
    else:
        quality_label = "Needs retrieval improvement"

    quality_reasons = []
    if average_recall is not None:
        quality_reasons.append(
            f"Recall is {_percentage(average_recall)}%, so the retriever finds most labeled evidence."
        )
    if average_precision is not None:
        quality_reasons.append(
            f"Precision@{k} is {_percentage(average_precision)}%, using the fixed evaluation denominator."
        )
    if average_capped_precision is not None:
        quality_reasons.append(
            f"Capped precision is {_percentage(average_capped_precision)}%, normalizing by the number of relevant chunks."
        )
    if average_context_precision is not None:
        quality_reasons.append(
            f"Context precision is {_percentage(average_context_precision)}%, measuring the chunks sent to the generator."
        )
    if no_answer_rate is not None:
        quality_reasons.append(
            f"Unsupported-question accuracy is {_percentage(no_answer_rate)}%, showing whether the system knows when to abstain."
        )

    return {
        "total": len(results),
        "passed": passed,
        "needs_review": len(results) - passed,
        "k": k,
        "precision_at_k": round(average_precision, 4) if average_precision is not None else None,
        "capped_precision_at_k": round(average_capped_precision, 4) if average_capped_precision is not None else None,
        "context_precision_at_k": round(average_context_precision, 4) if average_context_precision is not None else None,
        "recall_at_k": round(average_recall, 4) if average_recall is not None else None,
        "precision_percentage": _percentage(average_precision),
        "capped_precision_percentage": _percentage(average_capped_precision),
        "context_precision_percentage": _percentage(average_context_precision),
        "recall_percentage": _percentage(average_recall),
        "f1_at_k": round(f1, 4) if f1 is not None else None,
        "f1_percentage": _percentage(f1),
        "context_f1_at_k": round(context_f1, 4) if context_f1 is not None else None,
        "context_f1_percentage": _percentage(context_f1),
        "mean_reciprocal_rank": round(mean_reciprocal_rank, 4) if mean_reciprocal_rank is not None else None,
        "mrr_percentage": _percentage(mean_reciprocal_rank),
        "pass_percentage": _percentage(pass_rate),
        "no_answer_passed": no_answer_passed,
        "no_answer_total": len(no_answer_results),
        "no_answer_percentage": _percentage(no_answer_rate),
        "quality_label": quality_label,
        "quality_explanation": " ".join(quality_reasons),
        "metric_cases": len(precision_values),
        "results": results,
    }


def _answer_cache_key(model, question, context):
    context_hash = hashlib.sha256(context.encode("utf-8")).hexdigest()
    payload = json.dumps(
        {"model": model or "", "question": question, "context_hash": context_hash},
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), context_hash


def _load_answer_cache(cache_path):
    try:
        payload = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_answer_cache(cache_path, cache):
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _is_rate_limit_error(error):
    status_code = getattr(error, "status_code", None)
    if status_code == 429:
        return True
    response = getattr(error, "response", None)
    if getattr(response, "status_code", None) == 429:
        return True
    return "429" in str(error)


def _call_with_backoff(generate_fn, question, context, retries):
    for attempt in range(retries + 1):
        try:
            return generate_fn(question, context)
        except Exception as error:
            if not _is_rate_limit_error(error) or attempt >= retries:
                raise
            time.sleep(min(8.0, 0.5 * (2 ** attempt)))
    raise RuntimeError("Generation retry loop ended unexpectedly")


def _generation_metadata(generate_fn):
    module = sys.modules.get(getattr(generate_fn, "__module__", ""))
    if module is None:
        return {}
    getter = getattr(module, "get_last_generation_metadata", None)
    if callable(getter):
        metadata = getter()
        return metadata if isinstance(metadata, dict) else {}
    metadata = getattr(module, "LAST_GENERATION_METADATA", {})
    return dict(metadata) if isinstance(metadata, dict) else {}


def _match_text(value):
    value = (value or "").lower().replace("%", " percent ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def _term_matches(term, answer):
    normalized_term = _match_text(term)
    normalized_answer = _match_text(answer)
    return bool(normalized_term) and normalized_term in normalized_answer


def _is_list_question(question):
    normalized = question.lower()
    return (
        bool(re.search(r"\b(and|all|each|every|list|compare|comparison|difference|types|requirements|modules|percentages)\b", normalized))
        or normalized.startswith(("what are", "which "))
    )


def _percentage_from_count(numerator, denominator):
    if not denominator:
        return None
    return round(100 * numerator / denominator, 2)


def run_answer_evaluation(
    retrieve_fn,
    build_context_fn,
    generate_fn,
    k=None,
    limit=None,
    cache_path=None,
    model=None,
    delay=None,
    retries=None,
):
    """Run opt-in answer checks with a disk cache and quota-safe retries."""
    if k is None:
        k = RAG_TOP_K
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    cases = EVAL_CASES[:limit] if limit is not None else EVAL_CASES
    cache_path = Path(cache_path or FULL_EVAL_CACHE_PATH)
    cache = _load_answer_cache(cache_path)
    model = model or os.getenv("GEMINI_MODEL") or os.getenv("OPENAI_MODEL") or ""
    delay = FULL_EVAL_DELAY if delay is None else max(float(delay), 0.0)
    retries = FULL_EVAL_RETRIES if retries is None else max(int(retries), 0)
    results = []
    cache_hits = 0
    api_calls = 0

    for case in cases:
        question = case["question"]
        chunks = retrieve_fn(question, k=k)
        expected_terms = case["terms"]
        should_answer = case.get("should_answer", True)
        answer = ""
        metadata = {}
        cached = False
        error_message = None

        if not chunks:
            answer = "I don't have enough information to answer this question."
            metadata = {"finish_reason": "not_applicable", "truncated": False, "attempts": 0}
        else:
            built_context = build_context_fn(chunks)
            context = built_context[0] if isinstance(built_context, tuple) else built_context
            context = context or ""
            cache_key, context_hash = _answer_cache_key(model, question, context)
            cached_entry = cache.get(cache_key)
            if isinstance(cached_entry, dict) and "answer" in cached_entry:
                answer = cached_entry.get("answer", "")
                metadata = cached_entry.get("metadata", {}) or {}
                cached = True
                cache_hits += 1
            else:
                if api_calls and delay:
                    time.sleep(delay)
                try:
                    answer = _call_with_backoff(generate_fn, question, context, retries)
                    metadata = _generation_metadata(generate_fn)
                    cache[cache_key] = {
                        "model": model,
                        "question": question,
                        "context_hash": context_hash,
                        "answer": answer,
                        "metadata": metadata,
                    }
                    _save_answer_cache(cache_path, cache)
                    api_calls += 1
                except Exception as generation_error:
                    error_message = str(generation_error)
                    answer = ""
                    metadata = {"finish_reason": "error", "truncated": False, "attempts": 0}

        answer = answer or ""
        citations = re.findall(r"\[S(\d+)\]", answer)
        citation_numbers = [int(citation) for citation in citations]
        valid_citations = (
            bool(citations)
            and all(1 <= citation <= len(chunks) for citation in citation_numbers)
        )
        abstained = "i don't have enough information to answer this question" in answer.lower()
        answer_terms_found = [term for term in expected_terms if _term_matches(term, answer)]
        missing_terms = [term for term in expected_terms if term not in answer_terms_found]
        term_completeness = (
            len(answer_terms_found) / len(expected_terms)
            if expected_terms else None
        )
        list_completeness = term_completeness if _is_list_question(question) else None
        truncated = bool(metadata.get("truncated", False))
        if should_answer:
            passed = (
                bool(chunks)
                and not error_message
                and not truncated
                and term_completeness == 1
                and valid_citations
            )
            unsupported_claims = None
        else:
            passed = not error_message and abstained
            unsupported_claims = not abstained
            # No citation is expected for the exact abstention response.
            valid_citations = not citations

        results.append({
            "question": question,
            "expected_source": case.get("source"),
            "expected_answer": case["answer"],
            "should_answer": should_answer,
            "answer": answer,
            "passed": passed,
            "citations": [f"S{citation}" for citation in citations],
            "valid_citations": valid_citations,
            "answer_terms_found": answer_terms_found,
            "missing_terms": missing_terms,
            "term_completeness": term_completeness,
            "list_completeness": list_completeness,
            "abstained": abstained,
            "unsupported_claims": unsupported_claims,
            "truncated": truncated,
            "finish_reason": metadata.get("finish_reason"),
            "generation_attempts": metadata.get("attempts", 0),
            "cached": cached,
            "error": error_message,
        })

    passed = sum(result["passed"] for result in results)
    answerable_results = [result for result in results if result["term_completeness"] is not None]
    list_results = [result for result in results if result["list_completeness"] is not None]
    citation_results = [result for result in answerable_results]
    unsupported_results = [result for result in results if result["unsupported_claims"] is not None]
    term_values = [result["term_completeness"] for result in answerable_results]
    list_values = [result["list_completeness"] for result in list_results]

    return {
        "total": len(results),
        "passed": passed,
        "needs_review": len(results) - passed,
        "k": k,
        "limit": limit,
        "term_completeness_percentage": round(100 * sum(term_values) / len(term_values), 2) if term_values else None,
        "citation_validity_percentage": round(100 * sum(result["valid_citations"] for result in citation_results) / len(citation_results), 2) if citation_results else None,
        "unsupported_claims": sum(result["unsupported_claims"] for result in unsupported_results),
        "unsupported_claims_percentage": _percentage_from_count(sum(result["unsupported_claims"] for result in unsupported_results), len(unsupported_results)),
        "list_completeness_percentage": round(100 * sum(list_values) / len(list_values), 2) if list_values else None,
        "list_cases": len(list_results),
        "truncated_responses": sum(result["truncated"] for result in results),
        "cache_hits": cache_hits,
        "api_calls": api_calls,
        "results": results,
    }


if __name__ == "__main__":
    from rag.retriever import retrieve

    report = run_evaluation(retrieve)
    print(f"Evidence checks passed: {report['passed']}/{report['total']}")
    print(f"Needs review: {report['needs_review']}")
    print(f"Precision@{report['k']}: {report['precision_percentage']}%")
    print(f"Recall@{report['k']}: {report['recall_percentage']}%")
    for result in report["results"]:
        status = "CORRECT" if result["passed"] else "CHECK"
        print(f"{status}: {result['question']}")
