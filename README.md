# CUDA Python Lab

This repository implements Tasks 1–8 and a contract-level implementation for
Tasks 9–10 from the provided excerpt.

## Run in Google Colab
1. Select **Runtime → Change runtime type → T4 GPU**.
2. Upload/extract this repository or clone it.
3. Install dependencies: `!pip install -r requirements.txt`
4. Verify GPU: `!nvidia-smi`
5. Run tests: `!python autograder.py`

## Important scope notes
- The supplied task details only fully specify Tasks 1–10.
- Task 9's text does not fully define the expected output shape for stride > 1
  or exact performance thresholds. The implementation documents its chosen
  sampling contract; confirm this with the instructor before official grading.
- Tasks 11–50 are not implemented because their detailed contracts were not
  included in the supplied excerpt.
- `generate_submission_token.py` is intentionally a placeholder because the
  HMAC secret/fingerprint protocol was not supplied.
- The GitHub Actions workflow runs CPU-hosted CI. CUDA tests skip when no GPU
  is available; this is not a substitute for a GPU-enabled official grader.
