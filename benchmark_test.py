import os
import sys
import numpy as np
from nlp_engine import calculate_match_scores

# ---------------------------------------------------------------------------
# Benchmark Test Suite: 3 Roles & 12 Diverse Resumes
# ---------------------------------------------------------------------------
TEST_SUITE = [
    {
        "role": "Senior Python & AI Engineer",
        "description": "Looking for a Senior Python & AI Engineer with at least 3 years of experience. Must be proficient in Python, PyTorch or TensorFlow, Natural Language Processing, building REST APIs with FastAPI or Flask, and containerization using Docker and Linux.",
        "candidates": [
            {
                "id": "Cand-1A",
                "name": "Alex Chen (Senior AI Engineer)",
                "resume": "Senior AI & Machine Learning Engineer with 4 years of experience. Proficient in Python, PyTorch, TensorFlow, Natural Language Processing (NLP), transformers, Hugging Face, Flask, FastAPI, Docker, and Linux. Built and deployed scalable LLM applications.",
                "expected_score": 90.0,
                "expected_label": "Highly Suitable"
            },
            {
                "id": "Cand-1B",
                "name": "Sarah Miller (Data Analyst)",
                "resume": "Data Analyst with 3 years experience. Skilled in Python, SQL, Pandas, Tableau, and basic Machine Learning algorithms like Scikit-Learn regression and classification. Experienced in data visualization and reporting.",
                "expected_score": 55.0,
                "expected_label": "Suitable"
            },
            {
                "id": "Cand-1C",
                "name": "David Clark (Junior Web Developer)",
                "resume": "Junior Web Developer with 1 year experience in HTML, CSS, JavaScript, and basic Python scripting. Passionate about AI and eager to learn machine learning frameworks.",
                "expected_score": 30.0,
                "expected_label": "Low Match"
            },
            {
                "id": "Cand-1D",
                "name": "Marcus Vance (Executive Chef)",
                "resume": "Head Chef with 8 years of culinary and kitchen management experience. Expert in menu development, food safety protocols, inventory control, and staff supervision.",
                "expected_score": 5.0,
                "expected_label": "Low Match"
            }
        ]
    },
    {
        "role": "Frontend React Developer",
        "description": "Hiring a Frontend Developer with 2+ years experience building responsive web apps using React, JavaScript, TypeScript, HTML5, CSS3, and Redux. Experience with Git and REST APIs required.",
        "candidates": [
            {
                "id": "Cand-2A",
                "name": "Elena Rostova (React Developer)",
                "resume": "Frontend Developer with 3 years of experience specializing in React, TypeScript, JavaScript, Redux, HTML5, modern CSS, Git, and integrating RESTful APIs.",
                "expected_score": 88.0,
                "expected_label": "Highly Suitable"
            },
            {
                "id": "Cand-2B",
                "name": "Kevin Patel (Backend Java Dev)",
                "resume": "Backend Software Engineer with 4 years experience in Java, Spring Boot, Microservices, and SQL. Has basic working knowledge of JavaScript and HTML for internal admin dashboards.",
                "expected_score": 48.0,
                "expected_label": "Suitable"
            },
            {
                "id": "Cand-2C",
                "name": "Lisa Wong (UI/UX Designer)",
                "resume": "UI/UX Designer with 2 years experience creating wireframes, Figma mockups, and user research. Limited coding experience with basic HTML and CSS.",
                "expected_score": 32.0,
                "expected_label": "Low Match"
            },
            {
                "id": "Cand-2D",
                "name": "Robert Taylor (Chartered Accountant)",
                "resume": "Certified Public Accountant with 6 years experience in tax filing, auditing, financial statements, balance sheets, and QuickBooks.",
                "expected_score": 5.0,
                "expected_label": "Low Match"
            }
        ]
    },
    {
        "role": "DevOps & Cloud Engineer",
        "description": "Seeking a DevOps & Cloud Engineer with 4 years experience. Key requirements include AWS, Kubernetes, Docker, CI/CD automation with Jenkins/GitLab, Terraform, and Linux administration.",
        "candidates": [
            {
                "id": "Cand-3A",
                "name": "Tariq Mansoor (Senior DevOps)",
                "resume": "Lead DevOps Engineer with 5 years experience managing AWS infrastructure, Kubernetes clusters, Docker containers, CI/CD pipelines with GitLab, Terraform, and Linux systems.",
                "expected_score": 92.0,
                "expected_label": "Highly Suitable"
            },
            {
                "id": "Cand-3B",
                "name": "Brian Adams (Linux Sysadmin)",
                "resume": "System Administrator with 4 years experience in Linux server management, bash scripting, network troubleshooting, and introductory Docker experience. Actively studying for AWS solutions architect.",
                "expected_score": 52.0,
                "expected_label": "Suitable"
            },
            {
                "id": "Cand-3C",
                "name": "Emily Watson (Technical Support)",
                "resume": "IT Support Specialist with 2 years experience in desktop support, user account provisioning, Windows/Linux troubleshooting, and ticketing systems.",
                "expected_score": 28.0,
                "expected_label": "Low Match"
            },
            {
                "id": "Cand-3D",
                "name": "James Sullivan (Civil Project Manager)",
                "resume": "Civil Construction Manager with 7 years experience overseeing structural projects, contractor safety, concrete inspections, and CAD blueprints.",
                "expected_score": 5.0,
                "expected_label": "Low Match"
            }
        ]
    }
]

def run_evaluation():
    total_pairs = 0
    correct_labels = 0
    correct_p1 = 0
    errors = []

    print("-" * 65)
    print("Running Candidate Screening Benchmark Evaluation...")
    print("-" * 65)

    for item in TEST_SUITE:
        job_desc = item["description"]
        candidates = item["candidates"]

        scores = []
        for cand in candidates:
            result = calculate_match_scores(job_desc, [cand["resume"]])[0]
            scores.append((cand, result))

            total_pairs += 1
            if result["label"] == cand["expected_label"]:
                correct_labels += 1
            errors.append(abs(result["score"] - cand["expected_score"]))

        scores_sorted = sorted(scores, key=lambda x: x[1]["score"], reverse=True)
        if scores_sorted[0][0]["id"].endswith("A"):
            correct_p1 += 1

    label_accuracy = (correct_labels / total_pairs) * 100
    top_1_accuracy = (correct_p1 / len(TEST_SUITE)) * 100
    mae = float(np.mean(errors))

    print(f"Total Test Pairs Evaluated: {total_pairs}")
    print(f"Classification Label Accuracy: {label_accuracy:.2f}%")
    print(f"Top-1 Candidate Precision: {top_1_accuracy:.2f}%")
    print(f"Mean Absolute Error (MAE): {mae:.2f}%")
    print("-" * 65)

if __name__ == "__main__":
    run_evaluation()