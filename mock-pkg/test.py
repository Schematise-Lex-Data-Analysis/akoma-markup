"""Test the akoma-markup package."""

from pathlib import Path

from dotenv import load_dotenv

# Load .env so Azure env vars are available as fallback
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

from akoma_markup import convert

PDF_PATH = Path(__file__).parent / "a202345.pdf"
OUTPUT_PATH = Path(__file__).parent / "output" / "test_markup.txt"


def main():
    print(f"PDF:    {PDF_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print()

    result = convert(
        pdf_path=str(PDF_PATH),
        llm_config={
            "provider": "azure",
            "model": "Llama-3.3-70B-Instruct",
        },
        output_path=str(OUTPUT_PATH),
    )

    print(f"\nDone! Markup file: {result}")

    # Print a preview of the output
    output = Path(result)
    if output.exists():
        text = output.read_text()
        lines = text.splitlines()
        print(f"Output size: {len(text)} chars, {len(lines)} lines")
        print("\n--- First 30 lines ---")
        print("\n".join(lines[:30]))


if __name__ == "__main__":
    main()
