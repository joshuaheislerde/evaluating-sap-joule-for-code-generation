import subprocess
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

PROJECT_ROOT = Path("")
BASE_RESULTS_PATH = PROJECT_ROOT / "results-cleaned"
DATA_FILE = PROJECT_ROOT / "humaneval_js.jsonl"
LLM_MODELS = []

@dataclass
class EvaluationResult:
    """Data class for storing the evaluation results for one LLM"""
    total: int = 0
    correct: int = 0
    failed: int = 0
    assertion_failed: int = 0

    @property
    def exception_failed(self) -> int:
        """Calculates the number of failures caused by exceptions"""
        return self.failed - self.assertion_failed

    def update_counts(self, error_message: str):
        """Updates the counts based on the result of the subprocess"""
        self.total += 1
        if not error_message:
            self.correct += 1
        else:
            self.failed += 1
            if 'Assertion failed' in error_message:
                self.assertion_failed += 1

def execute_js_test(test_code: str) -> str:
    """
    Executes the JavaScript test code in a Node.js subprocess

    Args:
        test_code: The JavaScript code to execute

    Returns:
        The error message (stderr) as a string. An empty string on success
    """
    process = subprocess.run(
        ['node', '-e', test_code],
        capture_output=True,
        text=True,
        encoding='utf-8'
    )
    return process.stderr

def evaluate_model(model_name: str, df: pd.DataFrame) -> EvaluationResult:
    """
    Evaluates a single LLM against the test cases in the DataFrame

    Args:
        model_name: The name of the LLM to evaluate
        df: The DataFrame containing the test data ("task_id", "test")

    Returns:
        An EvaluationResult object with the summarized results
    """
    results = EvaluationResult()
    model_path = BASE_RESULTS_PATH / model_name

    for _, row in df.iterrows():
        task_id = row.task_id.replace("/", "-")
        file_path = model_path / f"{task_id}.js"

        try:
            with file_path.open("r", encoding="utf-8") as f:
                llm_response = f.read()
        except FileNotFoundError:
            print(f"Warning: File not found, skipping: {file_path}")
            continue

        test_code = f"{llm_response}\n\n{row.test}"
        error_output = execute_js_test(test_code)
        results.update_counts(error_output)

    return results

def print_summary(model_name: str, results: EvaluationResult):
    """
    Prints a formatted summary of the results

    Args:
        model_name: The name of the evaluated model
        results: The EvaluationResult object containing the result data
    """
    pass_rate = (results.correct / results.total * 100) if results.total > 0 else 0
    exception_rate = (results.exception_failed / results.failed * 100) if results.failed > 0 else 0

    print("-" * 60)
    print(f"Results for model: {model_name}")
    print("-" * 60)
    print(f"Strict accuracy:           {pass_rate:.2f} %")
    print(f" Total entries validated:  {results.total}")
    print(f" Successful:               {results.correct}")
    print(f" Failed (Total):           {results.failed}")
    print(f"  - Assertion Errors:      {results.assertion_failed}")
    print(f"  - Exception Errors:      {results.exception_failed} ({exception_rate:.2f} % of all failures)")
    print("-" * 60)

def main():
    """
    Main function: Loads the data and starts the evaluation process
    """
    try:
        df = pd.read_json(DATA_FILE, lines=True)
    except FileNotFoundError:
        print(f"Error: The data file '{DATA_FILE}' was not found.")
        return

    for llm in LLM_MODELS:
        model_results = evaluate_model(llm, df)
        print_summary(llm, model_results)

if __name__ == "__main__":
    main()