
import ast
from pathlib import Path

import pytest


# Load the actual allowed_file function from app.py
# without starting the AI models or Flask server.
APP_PATH = Path(__file__).resolve().parents[1] / "app.py"

source = APP_PATH.read_text(encoding="utf-8")
tree = ast.parse(source)

function_node = next(
    node for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name == "allowed_file"
)

module = ast.Module(body=[function_node], type_ignores=[])
namespace = {"ALLOWED_EXTENSIONS": {"pdf", "docx"}}

exec(compile(module, str(APP_PATH), "exec"), namespace)
allowed_file = namespace["allowed_file"]


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("resume.pdf", True),
        ("resume.docx", True),
        ("resume.PDF", True),
        ("resume.DOCX", True),
        ("resume.txt", False),
        ("resume.exe", False),
        ("resume.jpg", False),
        ("resume", False),
        ("resume.", False),
    ],
)
def test_allowed_file(filename, expected):
    assert allowed_file(filename) == expected

def test_no_resume_uploaded():
    from app import app

    app.config["TESTING"] = True
    client = app.test_client()

    response = client.post(
        "/screen",
        data={"job_description": "Python Developer"}
    )

    assert response.status_code == 400
    assert b"Please upload at least one valid PDF or DOCX resume" in response.data 