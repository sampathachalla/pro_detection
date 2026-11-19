"""Compare the local OCR API's spelling results with the sample answer key."""

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen
from uuid import uuid4


RESOURCES = Path(__file__).resolve().parent / "app" / "resources"


def post(path, data=b"", headers=None):
    request = Request(f"http://127.0.0.1:8000/api{path}", data, headers or {})
    try:
        with urlopen(request, timeout=240) as response:
            return json.load(response)
    except HTTPError as error:
        body = error.read()
        try:
            message = json.loads(body)["error"]["message"]
        except (ValueError, KeyError, TypeError):
            message = error.reason
        raise RuntimeError(f"HTTP {error.code}: {message}") from error


def detect(image):
    boundary = uuid4().hex
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{image.name}"\r\n'
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode()
    data = header + image.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    result = post("/images", data, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    if result["status"] == "needs_review":
        warnings = ", ".join(warning["code"] for warning in result["warnings"])
        print(f"  Continuing past image warnings: {warnings}")
        image_id = quote(result["image_id"], safe="")
        result = post(f"/images/{image_id}/confirm")
    if result["status"] != "done":
        raise RuntimeError(f"Unexpected OCR status: {result['status']}")
    return result


def normalize(word):
    return word.strip().casefold()


def word_predictions(predictions):
    for prediction in predictions:
        wrong = prediction["wrong_spelling"]
        correction = prediction["correct_spelling"]
        if len(wrong.split()) <= 1 and len(correction.split()) <= 1:
            yield prediction
            continue

        original_words = re.findall(r"\b\w+\b", wrong)
        corrected_words = re.findall(r"\b\w+\b", correction)
        alignment = SequenceMatcher(
            None, list(map(normalize, original_words)), list(map(normalize, corrected_words)),
            autojunk=False,
        )
        for operation, start, end, corrected_start, corrected_end in alignment.get_opcodes():
            if operation == "equal":
                continue
            originals = original_words[start:end]
            corrections = corrected_words[corrected_start:corrected_end]
            pairs = zip(originals, corrections) if len(originals) == len(corrections) else [
                (" ".join(originals), " ".join(corrections)),
            ]
            for original, corrected in pairs:
                yield {**prediction, "wrong_spelling": original, "correct_spelling": corrected}


def compare(expected, predictions):
    remaining = list(word_predictions(predictions))
    correct = incorrect = missed = 0
    for wrong, correction in expected.items():
        matches = [
            index for index, prediction in enumerate(remaining)
            if normalize(prediction["wrong_spelling"]) == normalize(wrong)
        ]
        if not matches:
            print(f"  Missed: {wrong} -> {correction}")
            missed += 1
            continue
        index = next(
            (index for index in matches
             if normalize(remaining[index]["correct_spelling"]) == normalize(correction)),
            matches[0],
        )
        prediction = remaining.pop(index)
        suggestion = prediction["correct_spelling"]
        box = prediction.get("box_number", "?")
        if normalize(suggestion) == normalize(correction):
            print(f"  Correct: {wrong} -> {suggestion} (box {box})")
            correct += 1
        else:
            print(f"  Wrong correction: {wrong} -> {suggestion} (box {box}, expected {correction})")
            incorrect += 1
    for prediction in remaining:
        wrong = prediction["wrong_spelling"]
        suggestion = prediction["correct_spelling"]
        box = prediction.get("box_number", "?")
        print(f"  Extra flag: {wrong} -> {suggestion} (box {box})")
    return correct, incorrect, missed, len(remaining)


def ratio(numerator, denominator):
    return f"{numerator}/{denominator} ({numerator / denominator:.1%})" if denominator else "N/A"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("start", type=int, help="first image number")
    parser.add_argument("end", type=int, help="last image number (inclusive)")
    args = parser.parse_args()
    if not 1 <= args.start <= args.end <= 30:
        parser.error("use an inclusive range from 1 to 30, such as 1 10")

    with (RESOURCES / "test_results.json").open(encoding="utf-8") as answer_file:
        answers = json.load(answer_file)

    correct = incorrect = missed = extra = failed = 0
    for number in range(args.start, args.end + 1):
        name = f"t{number:02d}.jpg"
        print(f"\n{name}", flush=True)
        try:
            result = detect(RESOURCES / name)
            print(f"  OCR: {result['count']} lines, {result['word_count']} word boxes")
            if result.get("words_annotated_url"):
                print(f"  Boxes: http://127.0.0.1:8000{result['words_annotated_url']}")
            spelling = (result.get("llm") or {}).get("spelling") or {}
            if spelling.get("status") != "ok":
                message = (spelling.get("error") or {}).get("message", "No spelling result")
                raise RuntimeError(message)
            counts = compare(answers[name], spelling["mistakes"])
        except (OSError, RuntimeError, KeyError, TypeError, ValueError) as error:
            print(f"  Failed: {error}")
            failed += 1
            continue
        correct += counts[0]
        incorrect += counts[1]
        missed += counts[2]
        extra += counts[3]

    requested = args.end - args.start + 1
    flagged = correct + incorrect
    expected = flagged + missed
    print(f"\nScored {requested - failed}/{requested} images ({failed} failed)")
    print(f"Correct: {correct} | Wrong correction: {incorrect} | Missed: {missed} | Extra: {extra}")
    print(f"Detection recall: {ratio(flagged, expected)}")
    print(f"Detection precision: {ratio(flagged, flagged + extra)}")
    print(f"Corrected out of expected: {ratio(correct, expected)}")
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
